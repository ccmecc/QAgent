import json, os
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM

import bottle, threading, queue
from bottle import request, Bottle


from refer_llm.tensor_utils import *
from config import train_config


def clear_queue(q, max_tries=5):
    with q.mutex:   
        q.queue.clear()   
        

# =========================
# infer
# =========================
def get_per_token_logps(model, input_ids: Tensor) -> Tensor:
    """ return exery token  log prob """
    with torch.inference_mode():
        logits = model(input_ids).logits  # (B, L, Vocab)
        logits = logits[:, :-1, :]       # (B, L-1, Vocab)
        input_ids = input_ids[:, 1:]     # (B, L-1)
        log_probs = logits.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=2, index=input_ids.unsqueeze(-1)).squeeze(-1)
        del logits, log_probs
    return token_log_prob  # (B, L-1)


# =========================
# server
# =========================
def create_app(raw_queue, result_queue):
    app = Bottle()

    @app.route('/upload', method='POST')
    def do_upload():
        data_list = bytes_list_to_list(request.body.read())
        if json.loads(data_list[0])['prompt_len'] == -1:
            clear_queue(result_queue)
            clear_queue(raw_queue)
            return b"clear"

        data = {
            'meta': json.loads(data_list[0]),
            'inputs': bytes_to_tensor(data_list[1]),
            'rewards': bytes_to_tensor(data_list[2]),
            'doc_masks': bytes_to_tensor(data_list[3])
        }
        if len(data_list) == 5:
            data['gen_logps'] = bytes_to_tensor(data_list[4])
        raw_queue.put(data)
        print('upload ok!')
        return b"ok"

    @app.route('/get', method='GET')
    def do_get():
        if result_queue.empty(): return b'empty'
        print('get batch ok!')
        return result_queue.get()
        
    return app

# =========================
# infer loop
# =========================
def process_loop(ref_model, raw_queue, result_queue):
    while True:
        try:
            data = raw_queue.get()
            prompt_length = data['meta']['prompt_len']
            input_tensor = data['inputs'].to(ref_model.device)
            per_token_logps = get_per_token_logps(ref_model, input_tensor)
            per_token_logps = per_token_logps[:, prompt_length-1:]
            out_data = [
                json.dumps(data['meta']).encode(),
                tensor_to_bytes(data['inputs']),
                tensor_to_bytes(data['rewards']),
                tensor_to_bytes(data['doc_masks']),
                tensor_to_bytes(per_token_logps)
            ]
            if 'gen_logps' in data:
                out_data.append(tensor_to_bytes(data['gen_logps']))
            xdata = make_bytes_list(out_data)
            result_queue.put(xdata)

            del data, input_tensor, per_token_logps
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[Process Loop Error]: {e}")

import threading
import asyncio
from bottle import run as bottle_run

def start_bottle(app):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    bottle_run(app, host='0.0.0.0', port=59875, server='tornado')


def main():
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{train_config.ref_device}'
    print(f"Reference worker process uses GPU {train_config.ref_device}")
    print("Loading model...")
    ref_model = AutoModelForCausalLM.from_pretrained(train_config.reference_path, torch_dtype=torch.bfloat16, _attn_implementation="sdpa").to('cuda')
    ref_model.eval().requires_grad_(False)

    raw_queue = queue.LifoQueue()
    result_queue = queue.LifoQueue()
    app = create_app(raw_queue, result_queue)
    threading.Thread(target=start_bottle, args=(app,), daemon=True).start()
    process_loop(ref_model, raw_queue, result_queue)

if __name__ == '__main__':
    main()