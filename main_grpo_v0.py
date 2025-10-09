import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import gc
import random, time, json
from tqdm import tqdm
from queue import Empty

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence
from torch.cuda import OutOfMemoryError
from transformers import AutoTokenizer, AutoModelForCausalLM


from rollout.rollout_QAgent import RolloutAsyn
from refer_llm.refer_client import RefClient
from config import train_config, ds_config


# https://github.com/huggingface/trl/issues/2709#issuecomment-2628814505
def GRPO_step(batch, engine, train_config):
    
    def get_per_token_logps(logits, input_ids):
        per_token_logps = [] # Use a loop to reduce memory peak.
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    prompt_length = batch['prompt_len']
    inputs = batch['inputs'].to(engine.device)
    logits_to_keep = inputs.size(1) - prompt_length

    advantages = batch['rewards'].to(engine.device).unsqueeze(1)
    logits = engine(inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it 
    per_token_logps = get_per_token_logps(logits, input_ids)  
    per_token_logps = per_token_logps[:,prompt_length-1:]
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

    if 'doc_masks' in batch: # batch['doc_masks'] [B, L_complete]
        mask = batch['doc_masks'].to(completion_mask.device)  
        completion_mask = completion_mask * mask[:, :logits_to_keep] # 
    if 'gen_logps' in batch:
        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))
        clipped_ratio = torch.clamp(ratio, 1-train_config.clip_param, 1+train_config.clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else: 
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        assert train_config.compute_gen_logps is False
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    per_token_loss = -(per_token_loss - train_config.beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    del inputs, advantages, logits, input_ids, per_token_logps, ref_per_token_logps, per_token_kl, completion_mask, per_token_loss
    if 'doc_masks' in batch: del mask;
    if 'gen_logps' in batch: del ratio, clipped_ratio;
    # torch.cuda.empty_cache()
    return loss


def rollout_worker(gen_queue, gen_device, main_worker= False):
    rolloutAsyn = RolloutAsyn(gen_queue, gen_device, main_worker)
    rolloutAsyn.rollout_worker()


refClient = RefClient(train_config.ref_server_url)
tokenizer = AutoTokenizer.from_pretrained(train_config.model_path)
if __name__ == '__main__':
    if mp.get_start_method(allow_none=True) != 'spawn': # 
        mp.set_start_method('spawn')

    import deepspeed
    deepspeed.init_distributed()
    
    if dist.get_rank() == 0: 
        res = refClient.upload(-1, None, None, None, None) # 清空ref queue
        print(res)
        gen_Queue_list = []
        is_main = True
        for gen_device in train_config.gen_devices:
            gen_Queue = mp.Queue()      # actor
            print('\nSTART vLLM generation...\n')
            p = mp.Process(target=rollout_worker, args=(gen_Queue, gen_device, is_main))
            is_main = False
            p.start()
            gen_Queue_list.append(gen_Queue)
    time.sleep(10)

    from liger_kernel.transformers import AutoLigerKernelForCausalLM
    # model = AutoModelForCausalLM.from_pretrained(train_config.model_path, torch_dtype=torch.bfloat16, _attn_implementation="sdpa")
    # model = AutoModelForCausalLM.from_pretrained(train_config.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model = AutoLigerKernelForCausalLM.from_pretrained(train_config.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.config.use_cache = False
    engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, model_parameters=model.parameters())
    
    check_point_dir = os.environ.get('check_point_dir')
    if check_point_dir is not None:
        print('\033[33m Load check_point_dir! \033[0m')
        success = engine.load_checkpoint(load_dir=check_point_dir, load_module_strict=True)
        optimizer_state = engine.optimizer.state
        for param_id, state in optimizer_state.items():
            print(f"Parameter ID: {param_id}, State: {state}")
    
    progress = range(1, train_config.all_steps+1)
    if dist.get_rank() == 0: progress = tqdm(progress);
    res = refClient.upload(-1, None, None, None, None)
    print(res)
    for step in progress:
        while True:  # get batch
            batch = refClient.get_batch()
            if batch is None:
                print('waiting for batch...')
                time.sleep(1)
                continue
            try:
                loss = GRPO_step(batch, engine, train_config)
                engine.backward(loss)
                break  
            except OutOfMemoryError as e:
                print(f"\033[33m[Warning] CUDA OOM encountered. Re-trying with a new batch.Exception:{e}\033[0m")
                engine.zero_grad()                    
                optimizer.zero_grad(set_to_none=True) 
                if 'batch' in locals(): del batch;
                if 'loss' in locals(): del loss;
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(1) 
        engine.step() 


        if dist.get_rank() == 0:
            progress.set_description(f"Loss: {loss.item():.8f}")

        if step % train_config.gen_update_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('[TRAINING PROC] sending latest state_dict ...')
                state_dict = engine.module.state_dict() 
                state_dict_cpu = {k: v.cpu() for k, v in state_dict.items()}
                del state_dict 
                for gen_Queue in gen_Queue_list:
                    gen_Queue.put(state_dict_cpu)
                torch.cuda.empty_cache() 
                gc.collect()
                print('\033[33m[TRAINING PROC] send state_dict ok!\033[0m')
            dist.barrier()
            res = refClient.upload(-1, None, None, None, None) 
            print(res)
        
        actual_step = -1
        if step % train_config.gradient_accumulation_steps == 0:
            actual_step = step // train_config.gradient_accumulation_steps
        if actual_step % train_config.save_steps == 0 and actual_step != -1:
            dist.barrier()
            engine.save_checkpoint(train_config.outputs_path)
            dist.barrier()
            if dist.get_rank() == 0:
                print('saving model')
                save_name = train_config.outputs_path + f"/step_{actual_step}"
                state_dict = engine.module.state_dict()
                state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                engine.module.save_pretrained(save_name, state_dict=state_dict)
                tokenizer.save_pretrained(save_name)
            dist.barrier()