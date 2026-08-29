"""bot_gguf.py — Skyrim NPC Discord bot, running on the quantized GGUF model via
llama-cpp-python instead of transformers/torch."""

import argparse
import os
import asyncio
import discord
from discord import app_commands
#from llama_cpp import Llama
import requests
import os

RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/yoqtj8dg12gscn/runsync"
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")



def generate(character, situation):
    payload = {
        "input": {
            "messages": [
                {"role": "system", "content": f"You are {character}, an NPC in The Elder Scrolls V: Skyrim. Stay fully in character. Respond the way {character} would actually speak — in tone, vocabulary, and attitude — in one or two short lines."},
                {"role": "user", "content": f"[Situation: {situation}]"}
            ],
            "sampling_params": {
                "temperature": 0.8,
                "top_p": 0.9,
                "repetition_penalty": 1.3,
                "max_tokens": 80
            }
        }
    }
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    resp = requests.post(RUNPOD_ENDPOINT, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["output"][0]["choices"][0]["tokens"][0].strip()





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
            # loop = asyncio.get_running_loop()
            # line = await loop.run_in_executor(
            #     None, dialogue_model.generate, character, situation
            # )

            loop = asyncio.get_running_loop()
            line = await loop.run_in_executor(
                None, generate, character, situation
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