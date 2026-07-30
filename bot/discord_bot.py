import argparse
import os
import discord
from discord import app_commands
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel

def build_prompt(tokenizer, character, situation):
    messages = [
        {"role": "system", "content": f"You are {character}, an NPC in The Elder Scrolls V: Skyrim. Stay fully in character. Respond the way {character} would actually speak — in tone, vocabulary, and attitude — in one or two short lines."},
        {"role": "user", "content": f"[Situation: {situation}]"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

