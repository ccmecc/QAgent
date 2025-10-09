from rollout.rollout_QAgent import RolloutAsyn

if __name__=='__main__':
    rolloutAsyn = RolloutAsyn(None)
    rolloutAsyn.rollout_worker()
    inputs = [{"Q": "what is python", "answers": ["xxx"]}]*10
    print('x'*100)
    _, answers, ans_token_ids, ans_masks = rolloutAsyn.trajectory_sampling(inputs, num_pre_Q=10)
    print(answers)
    for i in range(len(answers)):
        print(answers[i])
        print(ans_token_ids[i])
        print(ans_masks[i])
        for an_id, ans_mask in zip(ans_token_ids[i],ans_masks[i]):
            if ans_mask == 0:
                decoded_text = rolloutAsyn.tokenizer.decode([an_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                print(decoded_text, end='')
            else:
                print('x', end = '')
        print()
        print('-'*100)