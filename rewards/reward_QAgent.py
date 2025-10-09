import re
import math
import time
import string
from collections import Counter

import torch

#################################################
########        READER EVALUATION        ########
#################################################
def normalize_answer(s):
	"""Lower text and remove punctuation, articles and extra whitespace."""
	def remove_articles(text):
		regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
		return re.sub(regex, ' ', text)

	def white_space_fix(text):
		return ' '.join(text.split())

	def remove_punc(text):
		exclude = set(string.punctuation)
		return ''.join(ch for ch in text if ch not in exclude)

	def lower(text):
		return text.lower()

	return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
	""" EM: ground_truth is a str """
	return normalize_answer(prediction) == normalize_answer(ground_truth)

def regex_match(text, pattern):
	"""Test if a regex pattern is contained within a text."""
	try:
		pattern = re.compile(
		normalize_answer(pattern),
		flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
		)
	except BaseException:
		return False
	return pattern.search(normalize_answer(text)) is not None
    
def f1_score(prediction, ground_truth):
	""" F1: ground_truth is a str """
	prediction_tokens = normalize_answer(prediction).split()
	ground_truth_tokens = normalize_answer(ground_truth).split()
	common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
	num_same = sum(common.values())
	if num_same == 0:
		return 0
	precision = 1.0 * num_same / len(prediction_tokens)
	recall = 1.0 * num_same / len(ground_truth_tokens)
	f1 = (2 * precision * recall) / (precision + recall)
	return f1

def em_max_over_ground_truths(prediction, ground_truths, regex=True):
	""" EM: ground_truth is a list """
	return max([regex_match(prediction, gt) if regex else exact_match_score(prediction, gt) for gt in ground_truths])

def f1_max_over_ground_truths(prediction, ground_truths):
  """ F1: ground_truth is a list """
  return max([f1_score(prediction, gt) for gt in ground_truths])



#######################################################
###########          Reward Compute         ###########
#######################################################
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
  
def get_reward_Cot2Q(inputs, answers, num_pre_Q, rule='EM'):
	# assert rule in ['EM'], "rule   'EM'"
	gold_answers_list = [item['answers'] for item in inputs for _ in range(num_pre_Q)]
	accs = []
	for res,gold_answers in zip(answers, gold_answers_list):
		answer_text = extract_answer(res)
		em = em_max_over_ground_truths(answer_text, gold_answers, regex=False) if answer_text else 0
		accs.append(em)
	
	format_accs = [reward_format(res) for res in answers] 
	rewards = [x*y for x, y in zip(accs, format_accs)]
	return torch.tensor(rewards, dtype=torch.float32)




from config import train_config

def get_eval_Cot2Q(inputs, answers, rule='EM'):
	rewards = get_reward_Cot2Q(inputs, answers, num_pre_Q=1, rule='EM')

	client = OpenAI(api_key="EMPTY", base_url=train_config.vllm_llm_server,)

	def response(query, passage):
		answer_prompt = "{passage}\n\nAnswer the given question. The answer should be a few short words. Question: {question}."
		for i in range(3):  
			try:
				chat_response = client.chat.completions.create( 
					model=train_config.eval_model_name,
					messages=[ {"role": "user", "content": answer_prompt.format(question=query, passage=passage)}, ],
					temperature=0.0
				)
				return chat_response.choices[0].message.content
			except Exception as e:
				print(f"\033[93mAttempt {i+1} - Unexpected Error: {e}\033[0m\n")
			time.sleep(1)  
		return 'N/A'

	def frozen_acc(query, res, gold_answers):
		passage = extract_passage(res)
		answer = response(query, passage)
		em = em_max_over_ground_truths(answer, gold_answers, regex=True)
		f1 = f1_max_over_ground_truths(answer, gold_answers)
		return (em, f1)

	gold_answers_list = [item['answers'] for item in inputs]
	queries = [item['Q'] for item in inputs]	
	with ThreadPoolExecutor(max_workers=100) as executor: 
		em_f1s = list(executor.map(frozen_acc, queries, answers, gold_answers_list))	
	ems = [item[0] for item in em_f1s]
	f1s = [item[1] for item in em_f1s]
	return ems, f1s, rewards.tolist()


#######################################################################
#######################################################################
def get_reward_Cot2Q_frozen(inputs, answers, num_pre_Q, rule='EM'):
	# assert rule in ['EM'], "rule   'EM'"

	client = OpenAI(api_key="EMPTY", base_url=train_config.vllm_llm_server,)
	def response(query, passage):
		answer_prompt = "{passage}\n\nAnswer the given question. The answer should be a few short words. Question: {question}."
		for i in range(3):  
			try:
				chat_response = client.chat.completions.create( 
					model=train_config.supervision_model_name,
					messages=[ {"role": "user", "content": answer_prompt.format(question=query, passage=passage)}, ],
					temperature=0.0
				)
				return chat_response.choices[0].message.content
			except Exception as e:
				print(f"\033[93mAttempt {i+1} - Unexpected Error: {e}\033[0m\n")
			time.sleep(1)  
		return 'N/A'

	def frozen_acc(query, res, gold_answers):
		passage = extract_passage(res)
		passages = parse_passages(passage)
		passage = '\n'.join(passages)
		answer = response(query, passage)
		return em_max_over_ground_truths(answer, gold_answers, regex=True)

	gold_answers_list = [item['answers'] for item in inputs for _ in range(num_pre_Q)]
	queries = [item['Q'] for item in inputs for _ in range(num_pre_Q)]	
	with ThreadPoolExecutor(max_workers=100) as executor: 
		accs = list(executor.map(frozen_acc, queries, answers, gold_answers_list))

	hits = [em_max_over_ground_truths(answer_text, gold_answers, regex=True) for gold_answers, answer_text in zip(gold_answers_list, answers)]
	accs = [acc + 0.5*hit for acc, hit in zip(accs, hits)]  
	rewards = accs
	return torch.tensor(rewards, dtype=torch.float32)



def get_eval_Cot2Q_frozen(inputs, answers, rule='EM'):
	client = OpenAI(api_key="EMPTY", base_url=train_config.vllm_llm_server,)
	
	def response(query, passage):
		answer_prompt = "{passage}\n\nAnswer the given question. The answer should be a few short words. Question: {question}."
		for i in range(3):  
			try:
				chat_response = client.chat.completions.create( 
					model="Qwen2.5-3B-Instruct",
					messages=[ {"role": "user", "content": answer_prompt.format(question=query, passage=passage)}, ],
					temperature=0.0
				)
				return chat_response.choices[0].message.content
			except Exception as e:
				print(f"\033[93mAttempt {i+1} - Unexpected Error: {e}\033[0m\n")
			time.sleep(1)  
		return 'N/A'
	

	def frozen_acc(query, res, gold_answers):
		passage = extract_passage(res)
		passages = parse_passages(passage)
		passage = '\n'.join(passages)
		answer = response(query, passage)
		em = em_max_over_ground_truths(answer, gold_answers, regex=True)
		f1 = f1_max_over_ground_truths(answer, gold_answers)
		return (em, f1)

	gold_answers_list = [item['answers'] for item in inputs ]
	queries = [item['Q'] for item in inputs ]	
	with ThreadPoolExecutor(max_workers=100) as executor:  
		em_f1s = list(executor.map(frozen_acc, queries, answers, gold_answers_list))
	ems = [item[0] for item in em_f1s]
	f1s = [item[1] for item in em_f1s]
	hits = [em_max_over_ground_truths(answer_text, gold_answers, regex=True) for gold_answers, answer_text in zip(gold_answers_list, answers)]
	rewards = [acc + 0.5*hit for acc, hit in zip(ems, hits)]  
	return ems, f1s, rewards




###################################################################
############################## utils  #############################
###################################################################
def extract_answer(text):
	pattern = r"<answer>(.*?)</answer>"
	matches = re.findall(pattern, text, re.DOTALL)
	if matches:
		return matches[-1].strip()
	return None

def extract_search(text):
	# pattern = r"<search>(.*?)</search>"
	pattern = r"<information>(.*?)</information>"
	matches = re.findall(pattern, text, re.DOTALL)
	searches = len([content for content in matches if content.strip().lower() != "and"])
	return searches


def parse_passages(text):
	pattern = r'(Passage #\d+.*?)\s*(Passage #\d+.*?)(?=\s*Passage #\d+|\s*$)'
	matches = re.findall(pattern, text, re.DOTALL)
	result = []
	for title_line, content_line in matches:
		clean_title = title_line.strip()
		clean_content = content_line.strip()
		result.append({ 'title': clean_title, 'content': clean_content})
	passages = [item['content'] for item in result]
	passages = list(set(passages))
	return passages
	

def extract_passage(text):
	pattern = r"<information>(.*?)</information>"
	matches = re.findall(pattern, text, re.DOTALL)
	return '\n'.join(matches)


def has_answer(text):
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    return True if match else False


def reward_format(text, has_answer_form = True):
	def validate_plan_before_each_search(text):
		parts = re.split(r"<search>.*?</search>", text, flags=re.DOTALL)  
		return all(re.search(r"<plan>.*?</plan>\s*$", p, re.DOTALL) for p in parts[:-1])

	def validate_reflect_after_each_info(text):
		parts = re.split(r"<information>.*?</information>", text, flags=re.DOTALL)  
		return all(re.search(r"^\s*<reflection>.*?</reflection>", p, re.DOTALL) for p in parts[1:])

	format_correct = (validate_plan_before_each_search(text) and validate_reflect_after_each_info(text))
	if has_answer_form:
		format_correct = format_correct and has_answer(text)
	return 1 if format_correct else 0


