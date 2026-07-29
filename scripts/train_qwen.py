#train_qwen.py — LoRA fine-tune Qwen2.5-1.5B-Instruct on Skyrim NPC dialogue (chat format).

import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model,TaskType
from trl import SFTTrainer, SFTConfig


import torch

def main():
    parser = argparse.ArgumentParser(description="LoRA fine tune Qwen2.5 Instruction on NPC dialogue")
    parser.add_argument("--train_file",required=True)
    parser.add_argument("--val_file",required=True)
    parser.add_argument("--output_dir",required=True)
    parser.add_argument("--base_model",default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs",type=float,default=3.0)
    parser.add_argument("--batch_size",type=float,default=2)
    parser.add_argument("--grad_accum",type=int,default=8)
    parser.add_argument("--learning_rate",type=float,default=2e-4)
    parser.add_argument("--max_seq_length",type=int,default=256)
    parser.add_argument("--lora_r",type=int,default=8)
    parser.add_argument("--lora_alpha",type=int,default=16)
    args = parser.parse_args()

    print(f"Loading tokenizer/model: {args.base_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    model = AutoModelForCausalLM.from_pretrained(args.base_model,torch_dtype=torch.float16)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        bias="none",
    )

    model = get_peft_model(model,lora_config)
    model.print_trainable_parameters()


