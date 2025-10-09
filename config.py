import os

class Dict2Obj:
    def __init__(self, d):
        self.__dict__.update(d)

cfg_dict = {

    "corpus_path": "/path/corpus/wiki_bm25",
    "model_path": os.environ['model_path'],
    "reference_path": os.environ['reference_path'],
    "data_path": "/path/data/nq_hotpotqa_train/train.jsonl",
    "dev_path": "/path/data//hotpot/hotpot_dev.jsonl",
    "log_path": f"/path/outputs/logs/{os.environ['log_name']}.log",
    "outputs_path": f"/path/outputs/{os.environ['log_name']}_checkpoints",


    "reward_rule": os.environ['reward_rule'], 


    "ref_server_url": "http://localhost:59875",
    "retrieval_url": "http://localhost:8000/retrieve",


    "vllm_llm_server":"http://localhost:8005/v1",
    "supervision_model_name": "Qwen2.5-3B-Instruct",
    "eval_model_name": "Qwen2.5-3B-Instruct",
    


    "beta": 0.001,
    "clip_param": 0.2,
    "all_steps": 96000,
    "save_steps": 8,
    "gradient_accumulation_steps": 64,
    "gen_update_steps": 64,      

 
    "ref_device": '0',
    "gen_devices": ['1','2','3','4'],
    "gen_tensor_parallel_size": 1,
    "gen_gpu_memory_utilization": 0.3, # 0.25
    "gen_max_model_len": 1024*8,
   
    "Q_batch_size": 1*8,         
    "num_pre_Q": 5,              
    "train_batch_size": 5,      
    "compute_gen_logps": True,
    "max_truncate_length": 1024*8, 
}


train_config = Dict2Obj(cfg_dict)


ds_config = {
    "train_micro_batch_size_per_gpu": train_config.train_batch_size,
    "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 1e-8,
            "warmup_max_lr": 1e-6,
            # "warmup_num_steps": train_config.all_steps*0.285,
            "warmup_num_steps": train_config.all_steps*0.15,
        }
    }, 
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e7,
        "overlap_comm": False,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e7,
        "contiguous_gradients": True,
        # "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu", "pin_memory": True}
    },
    # "gradient_checkpointing": {
    #     "enable": True,
    #     "partition_activations": True,
    #     "cpu_checkpointing": True,
    #     "recompute_activations": True 
    # }
}
