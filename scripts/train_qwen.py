#train_qwen.py — LoRA fine-tune Qwen2.5-1.5B-Instruct on Skyrim NPC dialogue (chat format).

import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model,TaskType
from trl import SFTTrainer, SFTConfig


