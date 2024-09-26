from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import dotenv
import os
import torch
import time
from pathlib import Path
from typing import Union
import sys
from argparse import ArgumentParser
import dotenv
from utils import read_files
from hf_olmo import *

try:
    dotenv.load_dotenv(os.getenv("./models/.env"))
except:
    try:
        hf_token = os.getenv("HF_TOKEN")
    except:
        raise ValueError("Please provide the huggingface token")

parser = ArgumentParser()

def shard_model_and_tokenizer(
    save_dir: Path,
    token: str,
    max_shard_size: Union[str, int] = "3GB",
    model_name: str = "",
) -> None:
    """
    :param save_dir: Parent directory under which model shards dir will be saved
    :param token: huggingface token to access the model weights
    :param max_shard_size: shard size. int means number of bytes
    :param model_name: HF Model Name
    :return:
    """
    start = time.time()
    save_directory = save_dir.joinpath(
        f"{model_name.split('/')[-1]}_shard_size_{max_shard_size}"
    )
    print(f"Shard Dir: {str(save_directory)}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=token
    )
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, token=token)
    except:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, token=token
        )
        
    model.save_pretrained(save_directory, max_shard_size=max_shard_size)
    tokenizer.save_pretrained(save_directory)
    print(f"Time Taken: {time.time() - start}")
    
def main():
    hf_token = os.getenv("huggingface_token")
    
    parser.add_argument("--model_name", dest= "model_name",type=str, required=True)
    parser.add_argument("--save_dir",dest="save_dir" ,type=str, required=True,help="Parent directory under which model shards dir will be saved")
    parser.add_argument("--token", dest= "token" ,type=str, help="huggingface token to access the model weights",default=hf_token)
    parser.add_argument("--max_shard_size", dest="max_shard_size",type=str,help="shard size. int means number of bytes",default="2GB")
    
    args = parser.parse_args()
    shard_model_and_tokenizer(
        save_dir=read_files.mkdir_p(str(args.save_dir)+"/"),
        token=args.token,
        max_shard_size=args.max_shard_size,
        model_name=args.model_name,
    )
    
if __name__ == "__main__":
    main()
