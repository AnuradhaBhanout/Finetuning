"""
bot.py — Discord bot that generates Skyrim NPC dialogue on demand.

Slash command:
    /npc character:<name> situation:<short description>

Setup:
  1. Download your adapter folder from Drive to your local machine, e.g.:
       C:\\Users\\anura\\AI\\Projects\\Finetuning\\checkpoints\\gpt2-skyrim-lora\\final
  2. Create a Discord bot application at https://discord.com/developers/applications
     - Bot tab -> Reset Token -> copy it
     - Bot tab -> enable nothing special needed for slash commands (no privileged intents required)
     - OAuth2 -> URL Generator -> scopes: "bot" + "applications.commands" -> invite to your server
  3. Set your token as an environment variable (don't hardcode it):
       PowerShell:  $env:DISCORD_BOT_TOKEN = "your-token-here"
  4. Run:
       python bot.py --adapter_dir "C:\\path\\to\\checkpoints\\gpt2-skyrim-lora\\final"
"""

import argparse
import os

import discord
from discord import app_commands
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def build_prompt(character, situation):
    return f"### Character: {character}\n### Situation: {situation}\n### Dialogue:"


class DialogueModel:
    """Wraps the base model + LoRA adapter, loaded once at startup."""

    def __init__(self, adapter_dir, base_model="gpt2"):
        print(f"Loading tokenizer + base model ({base_model})...")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        base = AutoModelForCausalLM.from_pretrained(base_model)

        print(f"Loading LoRA adapter from {adapter_dir}...")
        self.model = PeftModel.from_pretrained(base, adapter_dir)
        self.model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model ready on device: {self.device}")

    def generate(self, character, situation, max_new_tokens=40, temperature=0.8):
        prompt = build_prompt(character, situation)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text[len(prompt):].strip()


def main():
    parser = argparse.ArgumentParser(description="Run the Skyrim NPC dialogue Discord bot.")
    parser.add_argument("--adapter_dir", required=True, help="Path to the saved LoRA adapter ('final' folder).")
    parser.add_argument("--base_model", default="gpt2")
    args = parser.parse_args()

    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "DISCORD_BOT_TOKEN environment variable not set.\n"
            'PowerShell: $env:DISCORD_BOT_TOKEN = "your-token-here"'
        )

    dialogue_model = DialogueModel(args.adapter_dir, args.base_model)

    intents = discord.Intents.default()
    client = discord.Client(intents=intents)
    tree = app_commands.CommandTree(client)

    @client.event
    async def on_ready():
        await tree.sync()
        print(f"Logged in as {client.user}. Slash commands synced.")

    @tree.command(name="npc", description="Generate Skyrim NPC dialogue")
    @app_commands.describe(
        character="Who is speaking, e.g. 'Belethor' or 'A guard'",
        situation="What's happening, e.g. 'Greeting a customer'",
    )
    async def npc(interaction: discord.Interaction, character: str, situation: str):
        await interaction.response.defer()  # generation takes a moment; avoids the 3s timeout
        try:
            line = dialogue_model.generate(character, situation)
            if not line:
                line = "(the NPC has nothing to say)"
            embed = discord.Embed(
                title=character,
                description=f'*"{line}"*',
                color=discord.Color.dark_gold(),
            )
            embed.set_footer(text=situation)
            await interaction.followup.send(embed=embed)
        except Exception as e:
            await interaction.followup.send(f"Something went wrong generating that line: {e}")

    client.run(token)


if __name__ == "__main__":
    main()