"""
train.py — LoRA fine-tune GPT-2 on Skyrim NPC dialogue.

Designed to run in Google Colab with a T4 GPU. Reads train.jsonl/val.jsonl
(produced by format_data.py) from Google Drive, trains a LoRA adapter on
top of GPT-2, and saves checkpoints + the final adapter back to Drive.

Usage (in a Colab cell, after mounting Drive):
    !python train.py \
        --train_file /content/drive/MyDrive/Finetuning/data/train.jsonl \
        --val_file /content/drive/MyDrive/Finetuning/data/val.jsonl \
        --output_dir /content/drive/MyDrive/Finetuning/checkpoints/gpt2-skyrim-lora

Why LoRA (not full fine-tuning or QLoRA):
  - GPT-2 (124M params) fits comfortably in T4 memory at full precision —
    no need for QLoRA's 4-bit quantization, which trades some accuracy
    for memory savings we don't need here.
  - LoRA still gives the efficiency win: only a small adapter (a few MB)
    is trained/saved, not the full 124M-param model. Faster to train,
    faster to iterate, and the adapter is portable.
"""

import argparse

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune GPT-2 on NPC dialogue.")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model", default="gpt2")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    args = parser.parse_args()

    print(f"Loading tokenizer/model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    # LoRA config: target GPT-2's attention projection layer.
    # r=8 keeps the adapter small; alpha=16 scales updates (2x r is a common default).
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["c_attn"],  # GPT-2's combined QKV projection
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # sanity check: should be a small fraction of total

    print("Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.val_file},
    )
    print(dataset)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none",
        fp16=True,  # T4 supports fp16; speeds up training meaningfully
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final LoRA adapter to {args.output_dir}/final")
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")

    print("Done.")


if __name__ == "__main__":
    main()