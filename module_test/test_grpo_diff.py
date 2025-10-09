import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from liger_kernel.transformers import AutoLigerKernelForCausalLM

from config import train_config, ds_config
from grpo_loss import fused_selective_log_softmax, triton_grpo_loss


def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)



def GRPO_step_v0(engine, batch):
    prompt_length = batch['prompt_len']
    inputs = batch['inputs'].to(engine.device)
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)
    
    logits = engine(inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it 

    per_token_logps = get_per_token_logps(logits, input_ids)  
    per_token_logps = per_token_logps[:,prompt_length-1:]
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

    if 'doc_masks' in batch: # batch['doc_masks'] 形状 [B, L_gen]
        mask = batch['doc_masks'].to(completion_mask.device)  
        completion_mask = completion_mask * mask
    if 'gen_logps' in batch:
        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))
        clipped_ratio = torch.clamp(ratio, 1-train_config.clip_param, 1+train_config.clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else: 
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        assert train_config.compute_gen_logps is False
    per_token_loss = -(per_token_loss - train_config.beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    
    print('x'*100)
    print(per_token_loss.size())
    print(per_token_loss)
    print(loss)
    print(per_token_loss * completion_mask)
    print('x'*100)
    return loss, per_token_loss * completion_mask



def GRPO_step_v1(engine, batch):
    from grpo_loss import triton_grpo_loss 

    prompt_length = batch['prompt_len']
    inputs = batch['inputs'].to(engine.device)
    prompt_ids = inputs[:, :prompt_length]
    completion_ids = inputs[:, prompt_length:].contiguous()
    logits_to_keep = completion_ids.size(1)

    logits = engine(inputs, logits_to_keep=logits_to_keep + 1).logits.contiguous() # (B, L)
    ref_per_token_logps = batch['refs'].to(engine.device).contiguous()
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()
    if 'doc_masks' in batch:
        mask = batch['doc_masks'].to(completion_mask.device)  
        completion_mask = completion_mask * mask[:, :logits_to_keep]
    if 'gen_logps' in batch:
        old_per_token_logps = batch['gen_logps'].to(engine.device)
    else:
        old_per_token_logps = None
        assert train_config.compute_gen_logps is False

    per_token_loss,_,_ = triton_grpo_loss(logits, old_per_token_logps, ref_per_token_logps,
                                    completion_ids, advantages, completion_mask,
                                    temperature=1.0, beta=train_config.beta,
                                    eps_low=train_config.clip_param, eps_high=train_config.clip_param,)

    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()  
    print('x'*100)
    print(per_token_loss.size())
    print(per_token_loss)
    print(loss) 
    print('x'*100)
    return loss, per_token_loss



from refer_llm.refer_client import RefClient
refClient = RefClient(train_config.ref_server_url)
tokenizer = AutoTokenizer.from_pretrained(train_config.model_path)

# model = AutoModelForCausalLM.from_pretrained(train_config.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to('cuda')
model = AutoLigerKernelForCausalLM.from_pretrained(train_config.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to('cuda')
batch = refClient.get_batch()
print(batch)
loss1,per1 = GRPO_step_v0(model,batch)
loss2,per2 = GRPO_step_v1(model,batch)
print(loss1-loss2)
print(per1-per2)