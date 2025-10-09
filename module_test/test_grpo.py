import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


from QueryR1_release.rollout.rollout_CoT2Q import RolloutAsyn
from config import train_config, ds_config



tokenizer = AutoTokenizer.from_pretrained('xxxxxxxx')

########################################################
#####################  From TRL  #######################
########################################################
# https://github.com/huggingface/trl/issues/2709#issuecomment-2628814505
def GRPO_step(batch, engine, train_config):
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
        completion_mask = completion_mask * mask[:, :logits_to_keep] # TODO
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

    print(inputs.size())
    print(prompt_length)
    print(completion_ids.size())
    print(logits.size())
    print(ref_per_token_logps.size())
    print(old_per_token_logps.size())
    print(completion_mask.size())
    print(per_token_loss.size())
    print('x'*100)

    print("\n=== Debug: Masked Predictions ===")
    completion_mask = ~completion_mask.bool()
    for i in range(completion_ids.shape[0]):  # 每个 batch 样本
        print(f"\n-- Sample {i} --")
        
        # 提取被 mask 的位置的预测 id
        masked_pred_ids = completion_ids[i][completion_mask[i]]  # 只取 mask 的位置
        
        if len(masked_pred_ids) == 0:
            print("  No masked tokens.")
        else:
            # 直接解码成可读文本（自动处理 ## 子词）
            decoded = tokenizer.decode(masked_pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            print(f"  Predicted masked text: '{decoded}'")
    


    del inputs, logits, advantages, ref_per_token_logps, completion_mask
    del per_token_loss
    if 'doc_masks' in batch:
        del mask
    if 'gen_logps' in batch:
        del old_per_token_logps
    torch.cuda.empty_cache()
    return loss