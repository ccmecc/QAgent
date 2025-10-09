import subprocess
import sys
import time
import os

import socket


def wait_for_port(port, host='localhost', timeout=60.0):
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            raise RuntimeError(f"{port} {timeout}")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        try:
            result = sock.connect_ex((host, port))
            if result == 0: 
                sock.close()
                print(f"✅  {port} start")
                return
        except (OSError, socket.error):
            pass
        finally:
            sock.close()
        time.sleep(1)


def launch():
    print("🚀 retrieval Web ...")
    retr_stdout_log = open(f"xxx.log", "w", encoding="utf-8")
    retrieval_proc = subprocess.Popen(
        [sys.executable, "retrieval/retrieval.py"],
        stdout=retr_stdout_log,
        stderr=subprocess.STDOUT
    )
    wait_for_port(8000, timeout=180)
    time.sleep(2)


    print("🚀 vllm reward ...")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    vllm_stdout_log = open(f"yyy.log", "w", encoding="utf-8")
    vllm_reward_proc = subprocess.Popen([
            'vllm', "serve", 
            "TODO",
            "--host", "0.0.0.0", "--port", "8005",
            "--dtype", "auto", "--trust-remote-code",
            "--served-model-name", "Qwen2.5-3B-Instruct", "--enable-chunked-prefill",
            "--gpu-memory-utilization", "0.35", "--max-model-len", "8192", "--max_num_seqs", "64",
            "--disable-log-requests"
        ],
        stdout=vllm_stdout_log,
        stderr=subprocess.STDOUT,
        env = env
    )
    wait_for_port(8005, timeout=180)
    time.sleep(2)


    print("🚀 refer Web ...")
    refer_proc = subprocess.Popen(
        [sys.executable, "-m", "refer_llm.refer_server"],
        stdout=sys.stdout,
        stderr=subprocess.STDOUT
    )
    wait_for_port(59875, timeout=180)
    time.sleep(2)


    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("🚀 DeepSpeed GRPO ...")
    train_proc = subprocess.Popen(
        ["deepspeed", "--include=localhost:5,6,7", "main_grpo_v1.py"],
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
        env = env
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 terminal...")
        retrieval_proc.terminate()
        refer_proc.terminate()
        vllm_reward_proc.terminate()
        train_proc.terminate()

        # wait 
        retrieval_proc.wait()
        refer_proc.wait()
        train_proc.wait()
        vllm_reward_proc.wait()
        print("✅ ok")
        
        vllm_stdout_log.close()
        retr_stdout_log.close()


if __name__ == "__main__":
    launch()