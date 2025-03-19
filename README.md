<div align="center" id="title"> <img src="./image/TokenSwiftLogo.png" width=400px /> </div>

<h3 align="center">From Hours to Minutes: Lossless Acceleration of Ultra Long Sequence Generation</h3>
<div align="center">
  <a href="https://huggingface.co/TokenSwift"><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/model-on-hf-sm-dark.svg" alt="Model On HF"></a>
 <a href="https://bigai-nlco.github.io/TokenSwift/"><img src="https://img.shields.io/badge/Website-TokenSwift-brightgreen.svg"/></a>
 <a href="https://arxiv.org/abs/2502.18890"><img src="https://img.shields.io/badge/Arxiv-2502.18890-b31b1b.svg?logo=arXiv" alt=""></a>
</div>

TokenSwift is a novel framework designed to substantially accelerate the generation process of ultra-long sequences, up to 100K tokens, while maintaining the target model's inherent quality. 

| Highlights          | Description                                  | Emoji |
|------------------|----------------------------------------------|-------|
| ⚡ **Speed**      | 3× faster than vanilla Transformers      | ⏩    |
| 🎯 **Lossless**   | Matches original model's output quality      | ✅    |
| 📈 **Scalability**| Linear time complexity for 100K+ sequences   | 📏    |
| 🛠️ **Plug & Play**| Works with most HuggingFace models           | 🤗    |

---

## ✨ News

[2025.3.19] 🔥🔥Relase model for finetuned [QwQ-32B](https://huggingface.co/TokenSwift/TokenSwift-QwQ-32B) with 3 $\times$ acceleration. Check out [inference guide](#inference) for deployment. 

[2025.2.28] 🔥🔥Relase model for finetuned [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/TokenSwift/TokenSwift-DeepSeek-R1-Distill-Qwen-32B) with 3 $\times$ acceleration. Check out [inference guide](#inference) for deployment. 

[2025.2.27] Paper Release on Arxiv.

---

## 📦 Demo
https://github.com/user-attachments/assets/5094fca7-0b12-470c-a7b6-456d254855d1

---

## ✨ Star History
[![Star History Chart](https://api.star-history.com/svg?repos=bigai-nlco/TokenSwift&type=Date)](https://www.star-history.com/#bigai-nlco/TokenSwift&Date)

---

## 📖 Table of contents
- [Introduction](#introduction)
- [Installation](#installation)
  - [Method 1: With pip](#method-1-with-pip)
  - [Method 2: From the source (recommended)](#method-2-from-the-source-recommended)
- [Getting Started](#getting-started)
  - [Models Download](#models-download)
  - [Inference](#inference)
- [Training Guide (Optional)](#training-guide-optional)
  - [Datasets Download](#datasets-download)
  - [How to Train](#how-to-train)
- [Citation](#citation)
- [Acknowledgment](#acknowledgment)

---

## Introduction
We propose **TokenSwift**, a novel framework that achieves **lossless acceleration** for ultra-long sequence generation (up to 100K tokens) while **reducing computation time from hours to minutes**. 

<img src='image/framework.png'>

*Illustration of TOKENSWIFT Framework. First, target model (LLM) with partial KV cache and three linear layers outputs 4 logits in a single forward pass. Tree-based attention is then applied to construct candidate tokens. Secondly, top-k candidate 4-grams are retrieved accordingly. These candidates compose draft tokens, which are fed into the LLM with full KV cache to generate target tokens. The verification is performed by checking if draft tokens match exactly with target tokens. Finally, we randomly select one of the longest valid draft tokens, and update n-gram table and KV cache accordingly.*

This repository contains:
- ✅ **100% reproducibility** for all experiments
- 📊 Benchmark scripts for sequence lengths: 20K/40K/60K/80K/100K
- 🤖 Pre-trained model adapters for Any Structure

<img src='image/res1.png'> <img src='image/res2.png'>
*Visualization of our acceleration performance vs. baseline methods*

---

## Installation

### Method 1: With pip
```bash
pip install tokenswift
```

### Method 2: From the source (recommended)
```bash
git clone https://github.com/bigai-nlco/TokenSwift.git
cd TokenSwift
conda create -n tokenswift python=3.11
conda activate tokenswift
conda install nvidia::cuda-nvcc
pip install -r requirements.txt
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

---

## Getting Started

### Models Download
| Model Name | Download Link |
|------------|-------------|
| TokenSwift-Yarn-Llama-2-7b-128k | [HuggingFace](https://huggingface.co/TokenSwift/TokenSwift-Yarn-Llama-2-7b-128k) |
| TokenSwift-Llama-3.1-8B | [HuggingFace](https://huggingface.co/TokenSwift/TokenSwift-Llama-3.1-8B) |
| TokenSwift-Qwen2.5-1.5B | [HuggingFace](https://huggingface.co/TokenSwift/TokenSwift-Qwen2.5-1.5B) |
| TokenSwift-Qwen2.5-7B | [HuggingFace](https://huggingface.co/TokenSwift/TokenSwift-Qwen2.5-7B) |
| TokenSwift-Qwen2.5-14B | [HuggingFace](https://huggingface.co/TokenSwift/TokenSwift-Qwen2.5-14B) |
| TokenSwift-DeepSeek-R1-Distill-Qwen-32B | [HuggingFace](https://huggingface.co/TokenSwift/TokenSwift-DeepSeek-R1-Distill-Qwen-32B) |
| TokenSwift-QwQ-32B | [HuggingFace](https://huggingface.co/TokenSwift/TokenSwift-QwQ-32B) |

### Inference
Take LLaMA3.1-8B as an example:
```bash
torchrun  --master-port 1111 --nproc_per_node=1 main.py \
    --model_type llama3_1 \
    --ckpt_path your_checkpoint_path \
    --prefill_len 4096 \
    --retrival_max_budget 4096 \
    --gen_len 102400 \
    --gamma 4 \
    --min_p 0.1 \
    --temperature 1.0 \
    --tree_decoding \
    --ngram_topk 20 \
    --penalty 1.2 \
    --penalty_length 1024 \
    --prompt_id 0

  <NOTE: Modify the data and model path>
```
For other models, you can run the scripts in ```infer_scripts/``` folder. For example：
```bash
bash infer_scripts/r1_qwen_32b.sh
```

---

## Training Guide (Optional)

### Datasets Download
From the [PG-19](https://huggingface.co/datasets/deepmind/pg19) training set, data larger than 8K are filtered out according to different tokenizer.

Or download processed training datasets from [llama2-pg19](https://huggingface.co/datasets/TokenSwift/llama2_pg19_train_data), [llama3.1-pg19](https://huggingface.co/datasets/TokenSwift/llama3.1_pg19_train_data), [qwen2.5-pg19](https://huggingface.co/datasets/TokenSwift/qwen2.5_pg19_train_data).

### How to Train
Take LLaMA3.1-8B as an example:
```bash
torchrun --master-port 1111 --nproc_per_node=4 train/train_legacy.py \
    --model_name_or_path /your_model_path/Meta-Llama-3.1-8B \
    --llama_type llama3_1 \
    --data_path /your_data_path/llama3_1_pg19_8k_data \
    --output_dir /your_checkpoint_path/adapter_ckpts_llama3_1 \
    --max_steps 200 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 10 \
    --save_steps 200 \
    --learning_rate 5e-3 \
    --weight_decay 0.1 \
    --warmup_steps 50 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --report_to tensorboard \
    --bf16 True \
    --medusa_heads 3 \
    --remove-unused-columns false
  
  <NOTE: Modify the data and model path>
```
For other models, you can run the scripts in ```train/scripts/``` folder. For example：
```bash
cd train
bash scripts/train_R1_qwen2_5_32b.sh
```

---

## Citation
If you are interested in our work or use our library, please cite:
```bibtex
@misc{tokenswift,
      title={From Hours to Minutes: Lossless Acceleration of Ultra Long Sequence Generation up to 100K Tokens}, 
      author={Tong Wu and Junzhe Shen and Zixia Jia and Yuxuan Wang and Zilong Zheng},
      year={2025},
      eprint={2502.18890},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.18890}, 
}
```

---

## Acknowledgment
This codebase is influenced by remarkable projects from the LLM community, including [Medusa](https://github.com/FasterDecoding/Medusa/tree/main) and [TriForce](https://github.com/Infini-AI-Lab/TriForce).




