"""bot_gguf.py — Skyrim NPC Discord bot, running on the quantized GGUF model via
llama-cpp-python instead of transformers/torch."""

import argparse
import os
 
import discord
from discord import app_commands
from llama_cpp import Llama

SYSTEM_TEMPLATE = (
    "You are {character}, an NPC in The Elder Scrolls V: Skyrim. Stay fully in "
    "character. Respond the way {character} would actually speak — in tone, "
    "vocabulary, and attitude — in one or two short lines."
)


class DialogueModel:
    """Wraps the quantized GGUF model, loaded once at startup."""
 
    def __init__(self, model_path, n_ctx=2048, n_threads=None):
        print(f"Loading GGUF model from: {model_path}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,       # context window; small since prompts here are short
            n_threads=n_threads,
            verbose=False,
        )
        print("Model ready.")