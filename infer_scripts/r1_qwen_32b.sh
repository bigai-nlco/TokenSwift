#!/bin/bash
#SBATCH --job-name=r1_qwen_32b  # create a short name for your job
#SBATCH --partition=HGX,DGX             # specify the partition name: gpu
#SBATCH --qos=lv0b
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G               # total memory (RAM) per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --cpus-per-task=64        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --output=slurm_output/qwen_32b.out      # output format
#SBATCH --error=slurm_output/qwen_32b_error.out      # error output file
#SBATCH --account=research


torchrun  --master-port 3333 --nproc_per_node=2 main.py \
    --model_path_prefix /mnt/buffer/wutong \
    --target DeepSeek-R1-Distill-Qwen-32B/ \
    --model_type qwen2_5 \
    --ckpt_path /mnt/buffer/wutong/adapter_ckpts_R1_qwen2_5_32b/kv_lr_0.005/checkpoint-200 \
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
    --prompt_id 0 \
    --tp_size 2
