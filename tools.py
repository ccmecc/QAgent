# coding=utf-8
import os, random, re, requests, time, json
from config import train_config



def retrieve(query, topk=1):
    while True:
        try:
            response = requests.post(
                train_config.retrieval_url,
                json={"query": query, "topk": topk, "return_scores": False}, timeout=5  
            )
            if response.status_code == 200:
                data = response.json()
                result_list = data.get('result', [])
                evidences = [f'Passage #{i+1} {passage["title"]}\nPassage #{i+1} {passage["text"]}'  for i, passage in enumerate(result_list)]
                evidence = '\n\n'.join(evidences)
                return evidence
            else:
                print(f"Request failed, status: {response.status_code}, retrying...")
        except Exception as e:
            print(f"Request error: {e}, retrying...")
        time.sleep(1) 
