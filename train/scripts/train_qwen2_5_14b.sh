torchrun --master-port 1111 --nproc_per_node=4 train/train_legacy.py \
    --model_name_or_path /your_model_path/Qwen2.5-14B \
    --llama_type qwen2_5 \
    --data_path /your_data_path/qwen2_5_pg19_8k_data \
    --output_dir /your_checkpoint_path/adapter_ckpts_qwen2_5_14b \
    --max_steps 600 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 10 \
    --save_steps 600 \
    --learning_rate 1e-3 \
    --weight_decay 0.1 \
    --warmup_steps 50 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --report_to tensorboard \
    --bf16 True \
    --medusa_heads 3 \
    --remove-unused-columns false
