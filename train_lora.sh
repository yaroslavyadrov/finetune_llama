#!/bin/sh

HF_USER=openlm-research
BASE_MODEL=open_llama_7b
DATA=baize-alp_so_qra
EPOCH=`date '+%s'`
RUN=`expr $EPOCH - 1677862104`
LORA_CHKPTS="$BASE_MODEL-$DATA-$RUN"

CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --hf_user $HF_USER \
  --hf_model $BASE_MODEL \
  --add_eos_token \
  --add_bos_token \
  ----no_use_fast \
  --micro_batch_size 16 \
  --batch_size 64 \
  --epochs 1 \
  --save_steps 20 \
  --eval_steps 20 \
  --learning_rate 0.0002 \
  --cutoff_len 512 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --val_set_size 2000 \
  --target_modules q_proj,k_proj,v_proj,down_proj,gate_proj,up_proj \
  --data_path data/data_tmp.json \
  --data_files alpaca,stackoverflow,quora \
  --output_dir /data/lora/finetuned_models/$LORA_CHKPTS
