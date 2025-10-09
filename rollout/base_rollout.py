import os, random, re, time, json
from queue import Empty
from abc import ABC, abstractmethod

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import train_config
from tools import retrieve
from refer_llm.refer_client import RefClient


def some_log(log_path, text):
    with open(log_path, 'a') as f:
        f.write(text + '\n')  

#########################
###### Deprecation ######
#########################
class DataHandler:
    def __init__(self, QAs_data_path, epochs=1, batch_size=1):
        self.QAs = self.load_data(QAs_data_path)
        self.epochs = epochs
        self.batch_size = batch_size
        self.batches = []
        self.prepare_batches()

    def load_data(self, data_path):
        with open(data_path, 'r') as f:
            QAs = [json.loads(line) for line in f]
        return QAs
        
    def prepare_batches(self):
        for _ in range(self.epochs):
            random.shuffle(self.QAs)
            for i in range(0, len(self.QAs), self.batch_size):
                batch = self.QAs[i:i + self.batch_size]
                if len(batch) < self.batch_size:
                    batch.extend(random.sample(self.QAs, self.batch_size - len(batch)))
                self.batches.append(batch)

    def get_batches(self):
        return self.batches


#########################################################################
######### base rollout，need to implement trajectory_sampling and do_eval
#########################################################################
class BaseRolloutAsyn(ABC):
    
    def __init__(self, answer_prompt, reward_func=None, gen_queue=None, gen_device=0, main_worker=False, model_path=None):
        self.gen_queue = gen_queue
        self.refClient = RefClient(train_config.ref_server_url)
        self.main_worker = main_worker      # main_worker 
        self.gen_device = float(gen_device) if isinstance(gen_device, str) else gen_device
        random.seed(42 + self.gen_device)   
        self.answer_prompt = answer_prompt
        
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gen_device}"
        print(f"Generation worker process uses GPU {gen_device}")
        torch.cuda.set_device(0)

        if model_path:
            train_config.model_path = model_path
        from vllm import LLM, SamplingParams 
        self.vllm_engine = LLM(
            model=train_config.model_path, 
            gpu_memory_utilization=train_config.gen_gpu_memory_utilization, 
            tensor_parallel_size=train_config.gen_tensor_parallel_size, 
            max_model_len=train_config.gen_max_model_len, 
        )
        self.tokenizer = AutoTokenizer.from_pretrained(train_config.model_path)
        # self.dataHandler = DataHandler(train_config.data_path, epochs=1, batch_size=train_config.Q_batch_size)
        # self.data = self.dataHandler.get_batches()
        self.QAs = self.load_data(train_config.data_path)
        self.reward_func = reward_func

    def rollout_worker(self):
        random.seed(42 + self.gen_device)
        print('Seed:', 42 + self.gen_device)

        # for it, inputs in enumerate(self.data):
        for it in range(999999999):
            update = self.try_update_model()
            if (update or it==0) and self.main_worker: # eval
                # res = self.refClient.upload(-1, None, None, None, None) 
                out_dict = self.dev_eval()
                some_log(train_config.log_path, json.dumps(out_dict, ensure_ascii=False)) 
            
            inputs = random.sample(self.QAs, train_config.Q_batch_size)
            tic = time.time()
            prompts, answers, ans_token_ids, ans_masks, query_nums= self.trajectory_sampling(inputs, train_config.num_pre_Q)
            rewards = self.reward_func(inputs, answers, train_config.num_pre_Q, rule=train_config.reward_rule)
            mean_reward = rewards.mean().item()
            print(f'time: {time.time()-tic:.2f}s    ', f'    Mean reward: {mean_reward}')
            if it % 8 == 0:  print('Response:\n', answers[rewards.argmax().item()],'\n', '-'*100);
            
            for i, prompt in enumerate(prompts):
                start = i * train_config.num_pre_Q
                end = (i + 1) * train_config.num_pre_Q
                cur_answers = answers[start:end]
                cur_ans_ids = ans_token_ids[start:end]
                cur_rewards = rewards[start:end]
                cur_doc_masks = ans_masks[start:end]
                self.process_one(prompt, cur_answers, cur_ans_ids, cur_rewards, cur_doc_masks, train_config)

    def process_one(self, prompt, answers, ans_token_ids, rewards, doc_masks, train_config):      
        def generator_logps(merged_ids, prompt_len):
            from vllm import SamplingParams
            gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)
            outputs = self.vllm_engine.generate(prompt_token_ids=merged_ids.tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
            logps_list = [output.prompt_logprobs[prompt_len:] for output in outputs]
            gen_logps = torch.tensor([[list(logprob.values())[0].logprob for logprob in logprobs] for logprobs in logps_list])
            return gen_logps

        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        prompt_len = prompt_ids.shape[1]
        batch_size = train_config.train_batch_size
        for batch_start in range(0, len(answers), batch_size):
            try:
                # group
                batch_answers = answers[batch_start:batch_start+batch_size]
                batch_ans_ids = ans_token_ids[batch_start:batch_start+batch_size]
                batch_rewards = rewards[batch_start:batch_start+batch_size]
                if batch_rewards.max() - batch_rewards.min() == 0 : continue # skip group
                batch_rewards = (batch_rewards - batch_rewards.mean()) / (batch_rewards.std() + 1e-4)
     
                batch_doc_masks = doc_masks[batch_start:batch_start+batch_size]
                tensor_doc_masks = [torch.tensor(doc_masks) for doc_masks in batch_doc_masks]
                tensor_doc_masks = pad_sequence(tensor_doc_masks, batch_first=True, padding_value=1)
                
                tensor_ans_ids = [torch.tensor(ans_ids) for ans_ids in batch_ans_ids]
                output_ids = pad_sequence(tensor_ans_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                prompt_ids = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, prompt_len)
                merged_ids = torch.cat([prompt_ids, output_ids], dim=1)
                ####################### max token #######################
                merged_ids = merged_ids[:,:train_config.max_truncate_length]

                gen_logps = None
                if train_config.compute_gen_logps:
                    gen_logps = generator_logps(merged_ids, prompt_len)

                res = self.refClient.upload(prompt_len, merged_ids, batch_rewards, tensor_doc_masks, gen_logps)
                if not res:
                    print('fail')
            except Exception as e:
                print(f"exception：{e}")

    def try_update_model(self):
        try:
            new_state_dict = self.gen_queue.get_nowait()
            print('\033[32m[VLLM PROC] recving new model ...\033[0m')
            llm_model = self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(new_state_dict.items())
            print('\033[32m[VLLM PROC] model updated\033[0m')
            del new_state_dict
            time.sleep(5)
            return True
        except Empty:
            print("\033[33mempty\033[0m")
            return
        except Exception as e:
            print(f"\033[31m{e}\033[0m")
            print('\033[33m[VLLM PROC] no new model!\033[0m')
            return

    def load_data(self, data_path):
        with open(data_path, 'r') as f:
            QAs = [json.loads(line) for line in f]
        return QAs

    @abstractmethod
    def trajectory_sampling(self, inputs, num_pre_Q):
        """
        :param inputs: 
        :param num_pre_Q: 
        :return: prompts, answers, ans_token_ids, ans_masks, query_nums
        """
        pass  



    def example_trajectory_sampling(self, inputs, num_pre_Q, temperature=1.0, retrieval_topk=1, max_loops=5):

        def build_prompt(query):
            return self.tokenizer.apply_chat_template([ {"role": "system", "content":  self.answer_prompt.format(question=query)},], tokenize=False, add_generation_prompt=True)

        from vllm import SamplingParams
        prompts = [build_prompt(item["Q"]) for item in inputs]
        
        n = num_pre_Q           
        max_loops = max_loops   

        generations = [ [p] * n for p in prompts ] 
        finished = [ [False] * n for _ in prompts ]
        search_cnt = [ [0] * n for _ in prompts ]
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

            sampling_params = SamplingParams(n=1, temperature=temperature, max_tokens=512, stop=["</search>"], skip_special_tokens=False)
            outputs_list = self.vllm_engine.generate(batch_inputs, sampling_params, use_tqdm=False)
            
            for k, outputs in enumerate(outputs_list):
                text = outputs.outputs[0].text
                i, j = mapping[k]

                if outputs.outputs[0].finish_reason == "stop" and outputs.outputs[0].stop_reason == '</search>':
                    text += '</search>' 
                _update_state(i,j,text, mask=False) 
                if outputs.outputs[0].finish_reason == "length":
                    finished[i][j] = True
                    continue


                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, text, re.DOTALL)
                if not match and search_cnt[i][j] >= max_loops:
                    finished[i][j] = True
                    continue
                if not match and search_cnt[i][j] < max_loops: 
                    state_text = '\nMy previous action is invalid. If I want to search, I should put the query between <search> and </search>. If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n'
                    search_cnt[i][j] += 1
                    _update_state(i,j, state_text, mask=True)
                    continue

                content = match.group(2).strip()  
                action = match.group(1)
                if action == 'answer': 
                    finished[i][j] = True
                    continue
                if action == 'search' and search_cnt[i][j] >= max_loops: 
                    finished[i][j] = True
                    continue
                if action == 'search' and search_cnt[i][j] < max_loops:
                    query_str = content
                    doc = retrieve(query_str, retrieval_topk)
                    doc = f"<information>{doc.strip()}</information>"
                    search_cnt[i][j] += 1
                    _update_state(i , j, doc, mask=True)
        
        answers = [ans for group in all_answers for ans in group]
        ans_token_ids = [token_ids for group in all_token_ids for token_ids in group]
        ans_masks = [token_masks for group in all_token_masks for token_masks in group]
        return prompts, answers, ans_token_ids, ans_masks, None