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

dotenv.load_dotenv(os.getenv("./models/.env"))
parser = ArgumentParser()



def shard_model_and_tokenizer(
    parent_save_dir: Path,
    token: str,
    max_shard_size: Union[str, int] = "3GB",
    model_name: str = "",
) -> None:
    """
    :param parent_save_dir: Parent directory under which model shards dir will be saved
    :param token: huggingface token to access the model weights
    :param max_shard_size: shard size. int means number of bytes
    :param model_name: HF Model Name
    :return:
    """
    start = time.time()
    save_directory = parent_save_dir.joinpath(
        f"{model_name.split('/')[-1]}_shard_size_{max_shard_size}"
    )
    print(f"Shard Dir: {str(save_directory)}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=token
    )
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, token=token)

    model.save_pretrained(save_directory, max_shard_size=max_shard_size)
    tokenizer.save_pretrained(save_directory)
    print(f"Time Taken: {time.time() - start}")
    
if __name__ == "__main__":
    hf_token = os.getenv("huggingface_token")
    
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--parent_save_dir", type=str, required=True,help="Parent directory under which model shards dir will be saved")
    parser.add_argument("--token", type=str, help="huggingface token to access the model weights",default=hf_token)
    parser.add_argument("--max_shard_size", type=str,help="shard size. int means number of bytes",default="2GB")
    
    args = parser.parse_args()
    shard_model_and_tokenizer(
        Path(args.parent_save_dir),
        args.token,
        args.max_shard_size,
        args.model_name,
    )

