# 🧠 Shard LLMs: Unleashing the Power of Large Language Models

## 🌟 Introduction

Welcome to the Shard LLMs project! This repository provides tools and techniques for efficiently managing and deploying Large Language Models (LLMs) through sharding. By breaking down these massive models, we can overcome resource constraints and unlock their full potential.

## 🚀 Why Shard LLMs?

Sharding is a game-changer for working with LLMs. Here's why:

1. 💾 **Memory Optimization**: Fit billion-parameter models across multiple GPUs.
2. ⚡ **Speed Boost**: Parallel processing for faster training and inference.
3. 📈 **Scalability**: Effortlessly scale to larger models and datasets.
4. 💰 **Cost-Effective**: Maximize hardware efficiency and reduce training costs.
5. 🔄 **Enhanced Throughput**: Process more requests simultaneously.
6. 🛡️ **Fault Tolerance**: Improve system resilience with distributed processing.
7. 🔧 **Flexibility**: Train large models on limited hardware or scale to massive clusters.
8. 🌐 **Optimized Communication**: Reduce overhead between model components.

## 🛠️ Getting Started

### Prerequisites

Ensure you have all the necessary dependencies:

```bash
pip install -r requirements.txt
```

This will install:
- python-dotenv==1.0.1
- transformers==4.44.2
- torch==2.4.1

### Installation

1. Clone the repository:
   ```bash
   # SSH
   git clone git@github.com:yzm1205/Shard-Any-LLMs.git

   # HTTPS
   git clone https://github.com/yzm1205/Shard-Any-LLMs.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Shard-Any-LLMs
   ```

## 🔬 Sharding a Model
To shard a HuggingFace model, use the following command:
```bash
python sharding_model.py \
  --model_name MODEL-ID \
  --save_dir SAVE_DIRECTORY \
  --max_shard_size SHARD_SIZE \
  --token YOUR_HUGGINGFACE_TOKEN
```
Example: \
Let's shard the LLaMA-3.1-8B-Instruct model as an example:

```bash
python sharding_model.py \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --save_dir ~/sharded_model \
  --max_shard_size 2GB \
  --token YOUR_HUGGINGFACE_TOKEN
```

Replace `YOUR_HUGGINGFACE_TOKEN` with your actual HuggingFace token.

## 🔧 Loading a Sharded Model

To use your sharded model, you can load it using HuggingFace's `AutoModelForCausalLM` and `AutoTokenizer`:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "./sharded_model/Meta-Llama-3.1-8B-Instruct/"
hf_token = "YOUR_HUGGINGFACE_TOKEN"

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    token=hf_token
)

tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    trust_remote_code=True,
    token=hf_token
)
```

> 💡 **Pro Tip**: You can also use other methods to load sharded models, such as the `pipeline` API from Transformers.

## 🤝 Contributing

We welcome contributions! If you have ideas for improvements or new features, feel free to open an issue or submit a pull request.

<comment>
## 📄 License

<>This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
</comment>

## 🙏 Acknowledgments

- HuggingFace for their amazing Transformers library
- The open-source AI community for continuous inspiration and support

Happy sharding! 🎉
