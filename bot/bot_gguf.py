"""bot_gguf.py — Skyrim NPC Discord bot, running on the quantized GGUF model via
llama-cpp-python instead of transformers/torch."""

import argparse
import os
import asyncio
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

    def generate(self, character, situation, max_tokens=280, temperature=0.8):

        messages = [
            {"role": "system", "content": SYSTEM_TEMPLATE.format(character=character)},
            {"role": "user", "content": f"[Situation: {situation}]"},
        ]

        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repeat_penalty=1.3,     # same intent as transformers' repetition_penalty
            frequency_penalty=0.3,  # closest llama.cpp equivalent to no_repeat_ngram_size
        )

        return response["choices"][0]["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser(description="Run the Skyrim NPC dialogue Discord bot (GGUF/CPU).")
    parser.add_argument("--model_path", required=True, help="Path to the quantized .gguf file.")
    parser.add_argument("--n_ctx", type=int, default=512)
    parser.add_argument("--n_threads", type=int, default=2)
    args = parser.parse_args()

    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "DISCORD_BOT_TOKEN environment variable not set.\n"
            'Linux/EC2: export DISCORD_BOT_TOKEN="your-token-here"'
        )


    dialogue_model = DialogueModel(args.model_path, n_ctx=args.n_ctx, n_threads=args.n_threads)

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
            #line = dialogue_model.generate(character, situation)
            loop = asyncio.get_running_loop()
            line = await loop.run_in_executor(
                None, dialogue_model.generate, character, situation
            )
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