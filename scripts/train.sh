#!/bin/bash

is_causal=True
is_preprocessed=True
is_tuned="untuned"
strategy="deepspeed_stage_3_offload"
upload_user="meta-llama"
model_type="Meta-Llama-3-8B-Instruct"
quantization_type="quantization"
peft_type="lora"
data_max_length=1024
target_max_length=256
precision="bf16"
batch_size=24

python main.py mode=train \
    is_causal=$is_causal \
    is_preprocessed=$is_preprocessed \
    is_tuned=$is_tuned \
    strategy=$strategy \
    upload_user=$upload_user \
    model_type=$model_type \
    quantization_type=$quantization_type \
    peft_type=$peft_type \
    data_max_length=$data_max_length \
    target_max_length=$target_max_length \
    precision=$precision \
    batch_size=$batch_size
