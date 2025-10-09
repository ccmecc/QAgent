import json
import torch
import requests
from refer_llm.tensor_utils import *


class RefClient:
    def __init__(self, server_url):
        self.server_url = server_url


    def upload(self, prompt_len, merged_ids, rewards, doc_masks, gen_logps=None):
        try:
            if prompt_len == -1:
                data = [json.dumps({"prompt_len": prompt_len}).encode()]
            else:
                data = [json.dumps({"prompt_len": prompt_len}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(rewards), tensor_to_bytes(doc_masks)]
                if gen_logps is not None:
                    data.append(tensor_to_bytes(gen_logps))

            xdata = make_bytes_list(data)
            r = requests.post(f"{self.server_url}/upload", data=xdata)
            if r.content == b'ok': 
                return 'ok'
            if r.content == b'clear':
                return 'clear'
            return None
        except Exception:
            return None
        
    def get_batch(self):
        try:
            r = requests.get(f"{self.server_url}/get").content
            if r == b'empty':
                return None
        except Exception:
            return None
        
        res = bytes_list_to_list(r)
        data = json.loads(res[0]) 
        data['inputs'] = bytes_to_tensor(res[1])
        data['rewards'] = bytes_to_tensor(res[2])
        data['doc_masks'] = bytes_to_tensor(res[3])
        data['refs'] = bytes_to_tensor(res[4])
        if len(res) == 6:
            data['gen_logps'] = bytes_to_tensor(res[5])
        return data

"""

 - tensor request:
    [ 
        { "prompt_len": prompt_len },
        merged_ids, 
        rewards,
        doc_masks,
        gen_logps_sp [optional]
    ]
 - tensor res:
    [ 
        { "prompt_len": prompt_len },
        merged_ids, 
        rewards,
        doc_masks,
        per_token_logps, (main  return)
        gen_logps_sp [optional]
    ]

"""