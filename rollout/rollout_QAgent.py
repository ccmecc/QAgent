import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import random, re, time, json
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import train_config
from tools import retrieve
from rollout.base_rollout import BaseRolloutAsyn
from QAgent_release.rewards.reward_QAgent import (
    extract_answer, extract_search, 
    extract_passage, parse_passages,
    em_max_over_ground_truths, f1_max_over_ground_truths
)


############################################################################################
# TODO::::TODO 
# from rewards.reward_CoT2Q import get_reward_Cot2Q as reward_func
# from rewards.reward_CoT2Q import get_eval_Cot2Q as eval_reward_func

from QAgent_release.rewards.reward_QAgent import get_reward_Cot2Q_frozen as reward_func
from QAgent_release.rewards.reward_QAgent import get_eval_Cot2Q_frozen as eval_reward_func
############################################################################################


def search_API(args):
    """ ((i,j),(query_str,retrieval_topk)) """
    (i,j) = args[0]
    (query_str,retrieval_topk) = args[1]
    doc = retrieve(query_str,retrieval_topk)
    doc = doc.strip()
    return ((i,j), doc)

def parallel_run(para_list, method=search_API, max_workers=50):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(method, item) for item in para_list]
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"fail: {e}")
    return results
 

answer_prompt = """Search for information to answer the given question.
You can search as many times as needed if you find you lack some knowledge. 
You will go through a loop of: \"<plan>xxx</plan>\n<search>xxx</search>\n<information>xxx</information>\n<reflection>xxx</reflection>\n<plan>xxx</plan>(if not complete)... \n<reflection>xxx</reflection>\n<answer>xxx<answer>\". 
You must conduct planning inside <plan> and </plan> first every time you call a search engine. 
After planing, you can call a search engine to search multiple queries (no more than 3) by <search>\n<query>query1</query>\n<query>query2</query>\n ... \n<query>queryk</query>\n</search>, and it will return the searched results between <information> and </information>. 
After getting information, you must conduct a reflection on the information and place your reflection between the <reflection> and </reflection> tags. 
Note that you must plan within <plan> and </plan> before searching, and reflect within <reflect> and </reflect> after receiving information.
Note that each query must be enclosed between <query> and </query>, and all queries must be placed between <search> and </search>, such as <search>\n<query>query1</query>\n<query>query2</query>\n ... \n<query>queryk</query>\n</search>
If the task is not yet complete, begin a new <plan>.
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer> without detailed illustrations. 
For example, <answer>\nxxx\n</answer>. The answer, \"xxx\", should be a few short words. Question: {question}. 
"""

class RolloutAsyn(BaseRolloutAsyn):
    def __init__(self, gen_queue=None, gen_device='0', main_worker=False, model_path=None):
        super().__init__(answer_prompt, reward_func, gen_queue, gen_device, main_worker, model_path)


    def trajectory_sampling(self, inputs, num_pre_Q, temperature=1.0, retrieval_topk=1, max_loops=5):
        
        def build_prompt(query):
            return self.tokenizer.apply_chat_template([ {"role": "system", "content":  self.answer_prompt.format(question=query)},], tokenize=False, add_generation_prompt=True)
        from vllm import SamplingParams
        prompts = [build_prompt(item["Q"]) for item in inputs]

        n = num_pre_Q           
        max_loops = max_loops   

 
        generations = [ [p] * n for p in prompts ] 
        finished = [ [False] * n for _ in prompts ]
        search_cnt = [ [0] * n for _ in prompts ]
        query_cnt = [ [0] * n for _ in prompts ]  
        all_answers = [ [''] * n for _ in prompts ]
        all_token_ids = [ [[] for _ in range(n)] for _ in prompts ] 
        all_token_masks = [ [[] for _ in range(n)] for _ in prompts ]
        
        def _update_state(i, j, text, mask = False):
            nonlocal generations, all_answers, all_token_ids, all_token_masks 
            generations[i][j] += text
            all_answers[i][j] += text
            new_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0).tolist()
            all_token_ids[i][j] += new_ids
            all_token_masks[i][j] += [0 if mask else 1] * len(new_ids)


 
        while not all(all(all_finished) for all_finished in finished):
            batch_inputs = []
            mapping = []
            for i, gens in enumerate(generations):
                for j, cur_prompt in enumerate(gens):
                    if not finished[i][j]:
                        batch_inputs.append(cur_prompt)
                        mapping.append((i, j))
            if not batch_inputs:
                break
            
            stop_tags = ["</search>", "</answer>"]
            from vllm import SamplingParams
            sampling_params = SamplingParams(n=1, temperature=temperature, max_tokens=512, stop=stop_tags, skip_special_tokens=False)
            outputs_list = self.vllm_engine.generate(batch_inputs, sampling_params, use_tqdm=False)
            
            search_list = []
            for k, outputs in enumerate(outputs_list):
                text = outputs.outputs[0].text
                i, j = mapping[k]

                if outputs.outputs[0].finish_reason == "length": 
                    _update_state(i, j, text, mask=False)        
                    finished[i][j] = True
                    continue
                if outputs.outputs[0].finish_reason == "stop" and outputs.outputs[0].stop_reason in stop_tags:
                    text += outputs.outputs[0].stop_reason
                _update_state(i, j, text, mask=False)            
        
               
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, text, re.DOTALL)
                if not match:
                    if search_cnt[i][j] <= max_loops:
                        state_text = '\nMy previous action is invalid. If I want to search, I should put the query between <search> and </search>. If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n'
                        _update_state(i, j, state_text, mask=True)
                        search_cnt[i][j] += 1
                    else:
                        finished[i][j] = True
                    continue



                content = match.group(2).strip()  # Return only the content inside the tags
                action = match.group(1)
                if action == 'answer' or search_cnt[i][j] >= max_loops:  
                    finished[i][j] = True
                    continue
                if action == 'search':
                    query_str = content

                    def parse_queries(text):  
                        pattern = r'<query>(.*?)</query>'
                        queries = re.findall(pattern, text, re.DOTALL)
                        return [q.strip() for q in queries]
        
                    queries = parse_queries(query_str)
                    queries = query_str.split('\n') if not queries else queries                    
                    query_cnt[i][j] += len(queries)
                    for q in queries:
                        search_list.append(((i,j),(q, retrieval_topk)))
                    search_cnt[i][j] += 1

        
            results = parallel_run(search_list, search_API, max_workers=5)
            index_set = set()
            for item in results:
                (i,j) = item[0]
                index_set.add((i,j))
            for (i,j) in index_set:
                _update_state(i, j, '\n<information>\n', mask=True)
            for item in results:
                i,j = item[0]
                state_text = item[1]
                _update_state(i, j, state_text + '\n\n', mask=True)
            for (i,j) in index_set:
                _update_state(i, j, '</information>\n', mask=True)

        answers = [ans for group in all_answers for ans in group]
        ans_token_ids = [token_ids for group in all_token_ids for token_ids in group]
        ans_masks = [token_masks for group in all_token_masks for token_masks in group]
        query_nums = [cnt for group in query_cnt for cnt in group]
        return prompts, answers, ans_token_ids, ans_masks, query_nums
    

    def dev_eval(self, retrieval_topk=1, input_path=None):
        QAs = self.load_data(input_path if input_path else train_config.dev_path)
        _, answers, _ , _, query_nums = self.trajectory_sampling(QAs, 1, temperature=0.0, retrieval_topk=retrieval_topk)
        eval_ems, eval_f1s, eval_rewards = eval_reward_func(QAs, answers)

        em,s_em,f1 = 0,0,0
        searches, failure, doc_num = 0,0,0
        for item, res_answer in zip(QAs, answers):
            searches += extract_search(res_answer)
            pred_answer = extract_answer(res_answer)
            if pred_answer == 'and' or pred_answer == None:
                failure +=1
                pattern = r'<information>(.*?)</information>'
                pred_answer = re.sub(pattern, "", res_answer, flags=re.DOTALL)
            
            info = extract_passage(res_answer)
            doc_num += len(parse_passages(info))
                
            em += em_max_over_ground_truths(pred_answer, item['answers'])
            s_em += em_max_over_ground_truths(pred_answer, item['answers'], regex=False)
            f1 += f1_max_over_ground_truths(pred_answer, item['answers'])
        
        return {
            'S_EM': round(s_em/len(QAs)*100,3), 'EM':round(em/len(QAs)*100,3), 'F1':round(f1/len(QAs)*100,3), 
            'eval_EM':round(sum(eval_ems)/len(QAs)*100,3), 'eval_F1':round(sum(eval_f1s)/len(QAs)*100,3), 'eval_rewards':round(sum(eval_rewards)/len(QAs)*100,3),
            'Search':round(searches/len(QAs),3), 'Failure':round(failure/len(QAs)*100,3), 'Query':round(sum(query_nums)/len(QAs),3), 'doc_num': round(doc_num/len(QAs),3),
        }


import time
if __name__ == '__main__':
    from rollout.rollout_QAgent import RolloutAsyn
    ############################################################################
    data_path,gen_device = '/path/data/hotpot/hotpot_test.jsonl', 1
    ############################################################################
    rolloutAsyn = RolloutAsyn(gen_device=gen_device)
    start = time.time()
    results = rolloutAsyn.dev_eval(input_path = data_path)
    print(data_path)
    print(results)
    duration = time.time()-start
    print(f"\n {duration:.6f} s")