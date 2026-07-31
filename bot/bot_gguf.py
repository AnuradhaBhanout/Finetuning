"""bot_gguf.py — Skyrim NPC Discord bot, running on the quantized GGUF model via
llama-cpp-python instead of transformers/torch."""

import argparse
import os
 
import discord
from discord import app_commands
from llama_cpp import Llama