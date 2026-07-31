"""merge_lora.py — Merge the qwen-skyrim-lora adapter into the base Qwen2.5-1.5B-Instruct
weights, producing one standalone model folder."""

import argparse


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model weights.")
    parser.add_argument("--adapter_dir", required=True, help="Path to the saved LoRA adapter ('final' folder).")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_dir", required=True, help="Where to save the merged, standalone model.")
    args = parser.parse_args()
 
    print(f"Loading base model: {args.base_model}")
    # fp16 here just to keep the merge itself light on memory — GGUF quantization
    # happens later as a separate step, this is not that.
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16)
 
    print(f"Loading tokenizer from adapter dir (has the chat template): {args.adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
 
    print(f"Loading LoRA adapter from: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)


    print("Merging adapter into base weights...")
    merged_model = model.merge_and_unload()