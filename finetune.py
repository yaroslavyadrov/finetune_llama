import os
import sys
import torch
import pickle
import random
import json
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)


# Parameters
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--hf_user", type=str, default="decapoda-research", help="Hugging Face user")
parser.add_argument("--hf_model", type=str, default="llama-7b-hf", help="Hugging Face model")
parser.add_argument("--no_use_fast", action="store_false", help="If set, do not use fast tokenizer")
parser.add_argument("--add_eos_token", action="store_true", help="If set, adds an end-of-sentence token to each input sentence.")
parser.add_argument("--add_bos_token", action="store_true", help="If set, adds a beginning-of-sentence token to each input sentence.")
parser.add_argument("--micro_batch_size", type=int, help="Micro batch size")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
parser.add_argument("--save_steps", type=int, default=100, help="Number of steps between checkpoint saves")
parser.add_argument("--eval_steps", type=int, default=100, help="Number of steps between evaluation set tests")
parser.add_argument("--learning_rate", type=float, help="Learning rate")
parser.add_argument("--cutoff_len", type=int, default=512, help="Cutoff length")
parser.add_argument("--lora_r", type=int, default=8, help="LoRA R value")
parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha value")
parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout value")
parser.add_argument("--val_set_size", type=int, default=2000, help="Validation set size")
parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,down_proj,gate_proj,up_proj", help="Target modules")
parser.add_argument("--data_path", type=str, default="data/data_tmp.json", help="Data path")
parser.add_argument("--data_files", type=str, default="alpaca,stackoverflow,quora", help="Data files")
parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory path")

args = parser.parse_args()

hf_model = "%s/%s" % (args.hf_user, args.hf_model)
target_modules = args.target_modules.split(',')
output_dir = args.output_dir
gradient_acc_steps = args.batch_size // args.micro_batch_size

# Load data
data = []
for x in args.data_files.split(","):
    data += json.load(open("data/{}_chat_data.json".format(x)))
random.shuffle(data)
json.dump(data, open(args.data_path, "w"))
data = load_dataset("json", data_files=args.data_path)

# Load Model
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    gradient_acc_steps = gradient_acc_steps // world_size

model = LlamaForCausalLM.from_pretrained(
    hf_model,
    load_in_8bit=True,
    device_map=device_map,
)
total_params, params = 0, 0

tokenizer = LlamaTokenizer.from_pretrained(
    hf_model,
    add_eos_token=args.add_eos_token,
    add_bos_token=args.add_bos_token,
    use_fast=(not args.no_use_fast)
)

model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=target_modules,
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
config.save_pretrained(output_dir)

model = get_peft_model(model, config)
tokenizer.pad_token_id = 0

for n, p in model.model.named_parameters():
    if any([x in n for x in ["lora"]]):
        total_params += p.numel()
    params += p.numel()

print(
    "Total number of parameters: {}M, rate: {}%".format(
        total_params // 1000 / 1000, round(total_params / params * 100, 2)
    )
)


# Data Preprocess
def generate_prompt(data_point):
    return data_point["input"]


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=args.cutoff_len + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }


def generate_and_tokenize_prompt(data_point):
    prompt = generate_prompt(data_point)
    return tokenize(prompt)


if args.val_set_size > 0:
    train_val = data["train"].train_test_split(
        test_size=args.val_set_size, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
else:
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = None


# Training
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=gradient_acc_steps,
        warmup_steps=100,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=20,
        evaluation_strategy="steps" if args.val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=args.eval_steps if args.val_set_size > 0 else None,
        save_steps=args.save_steps,
        output_dir=output_dir,
        save_total_limit=100,
        load_best_model_at_end=True if args.val_set_size > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
