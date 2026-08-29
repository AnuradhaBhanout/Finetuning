import argparse
import json
import os
import statistics
import requests

from metrics import refusal_register,length_ok,max_ngram_repeat,leaked_markup,distinct_n,selfcheck

GEN_ENDPOINT = os.environ.get("GEN_ENDPOINT","http://localhost:8000/v1/completions")
GEN_API_KEY = os.environ.get("GEN_API_KEY","")

def build_prompt(character:str,situation:str)->str:
    return (
        f"<|im_start|>system\nYou are {character}, an NPC in The Elder Scrolls V: "
        f"Skyrim. Stay fully in character. Respond the way {character} would "
        f"actually speak - in tone, vocabulary, and attitude - in one or two "
        f"short lines.<|im_end|>\n"
        f"<|im_start|>user\n[Situation: {situation}]<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def generate(character:str,situation:str,model:str,seed:int)-> str:
    resp =requests.post(
        GEN_ENDPOINT,
        headers={"Authorization": f"Bearer {GEN_API_KEY}"} if GEN_API_KEY else {},
        json={
            "model":model,
            "prompt": build_prompt(character,situation),
            "max_tokens":80,
            "temperature":0.8,
            "top_p":0.9,
            "repetition_penalty":1.3,
            "seed":seed,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"].strip()
