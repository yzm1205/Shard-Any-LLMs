from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

import dotenv
import os
import torch
import time
from pathlib import Path
from typing import Union
import sys
from argparse import ArgumentParser

sys.path.insert(0, "./src/")
from utils import mkdir_p
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

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
    :param token: huggingface token to access the llama weights
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

