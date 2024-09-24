import os
from src.sharding_model import shard_model_and_tokenizer
from load_sharded_model import get_model_and_tokenizer
from utils import mkdir_p, full_path

hf_token = os.getenv("huggingface_token")  # Provide your HF Token here

def shard_model(model_id, shard_size="2GB",hf=hf_token):
    shard_model_and_tokenizer(
        parent_save_dir=mkdir_p(f"/data/shared/{model_id.split('/')[-1]}"),
        max_shard_size=shard_size,
        model_name=model_id, # Example: model_name = "meta-llama/Llama-2-7b-hf"
        token=hf,
    )

def load_shared_model(model_path, hf_token, device):
    model, tokenizer = get_model_and_tokenizer(model_path, device,hf=hf_token)
    return model, tokenizer


if __name__ == "__main__":
    model_id = "meta-llama/Llama-2-7b-hf"
    
    # shard the model
    shard_model(model_id)
    
    # load the model
    model_path = str(full_path("PATH_TO_SAVE")) # eg : /data/shared/****
    model, tokenizer = load_shared_model(model_path, hf_token, device='cuda')