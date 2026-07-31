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


class DialogueModel:
    """Wraps the base model + LoRA adapter, loaded once at startup."""

    def __init__(self, adapter_dir, base_model="Qwen/Qwen2.5-1.5B-Instruct"):
        print(f"Loading tokenizer + base model ({base_model})...")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)  # tokenizer saved with adapter has the chat template
        base = AutoModelForCausalLM.from_pretrained(base_model)

        print(f"Loading LoRA adapter from {adapter_dir}...")
        self.model = PeftModel.from_pretrained(base, adapter_dir)
        self.model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model ready on device: {self.device}")

    def generate(self, character, situation, max_new_tokens=80, temperature=0.8):
        prompt = build_prompt(self.tokenizer, character, situation)
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

        # Slice by token position (not string length)
        input_length = inputs["input_ids"].shape[1]
        completion_ids = output[0][input_length:]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()



def main():
    parser = argparse.ArgumentParser(description="Run the Skyrim NPC dialogue Discord bot.")
    parser.add_argument("--adapter_dir", required=True, help="Path to the saved LoRA adapter ('final' folder).")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
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
        character="Who is speaking, e.g. 'Belethor' or 'A bandit'",
        situation="What's happening, e.g. 'Greeting a customer'",
    )

    async def npc(interaction: discord.Interaction, character: str, situation: str):
        await interaction.response.defer()
        try:
            line = dialogue_model.generate(character,situation)
            if not line:
                line = "(NPC has nothing to say)"
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

if __name__ =="__main__":
    main()