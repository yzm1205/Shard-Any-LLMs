from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel
import torch
import dotenv
import os
import torch
import time
from pathlib import Path
from typing import Union, List
import sys
sys.path.insert(0, "./src/")
from utils import mkdir_p,full_path
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM

dotenv.load_dotenv(os.getenv("./models/.env"))
hf = os.getenv("huggingface_token")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_model_and_tokenizer(model_dir: str, hf_token: str, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True,torch_dtype=torch.float16,token=hf_token).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True,token=hf_token)
    return model.eval(), tokenizer

def model(sentence: Union[str, List[str]], model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: torch.device):
    sent_encode = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True,max_length=1024,return_token_type_ids=False).to(device=device)
    with torch.no_grad():
        model_embedding = model(**sent_encode,output_hidden_states=True)
    sentence_embeddings = mean_pooling(model_embedding.hidden_states, sent_encode['attention_mask'])
    return sentence_embeddings


if __name__ ==  "__main__":
    model_path = str(full_path("PATH_TO_SAVE")) # eg : /data/shared/olmo/OLMo-7B_shard_size_1GB
    device = 'cuda'
    hf_model_name="" # eg : "meta-llama/Meta-Llama-3-8B"
    model, tokenizer = get_model_and_tokenizer(model_path, hf, device)
    message = ["Language modeling is ","this is sentence 2"]
    model_embedding = model(message, model, tokenizer, device)
    print(model_embedding.shape)
    print("done")
