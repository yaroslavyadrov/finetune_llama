#!/bin/sh

HF_USER=openlm-research
BASE_MODEL=open_llama_7b
DATA=sexting
EPOCH=`date '+%s'`
RUN=`expr $EPOCH - 1677862104`
LORA_CHKPTS="$BASE_MODEL-$DATA-$RUN"

CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --hf_user $HF_USER \
  --hf_model $BASE_MODEL \
  --micro_batch_size 16 \
  --batch_size 128 \
  --epochs 12 \
  --save_steps 1 \
  --eval_steps 1 \
  --learning_rate 4e-5 \
  --cutoff_len 264 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --val_set_size 180 \
  --target_modules q_proj,k_proj,v_proj,down_proj,gate_proj,up_proj \
  --data_path /content/finetune_llama/data/kitsune_sexting_chat_data.json \
  --data_files sexting \
  --output_dir /content/finetune_llama/data/lora/$LORA_CHKPTS
