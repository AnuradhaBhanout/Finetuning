#Qwen2.5-Instruct

import json
import random
import argparse
from pathlib import Path
 
PLAYER_SPEAKER_NAMES = {
    "dragonborn", "player", "you", "last dragonborn", "the dragonborn"
}
 
SYSTEM_TEMPLATE = (
    "You are {speaker}, an NPC in The Elder Scrolls V: Skyrim. "
    "Stay fully in character. Respond the way {speaker} would actually speak — "
    "in tone, vocabulary, and attitude — in one or two short lines."
)

def load_raw(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries
 
def is_player_speaker(speaker):
    return speaker.strip().lower() in PLAYER_SPEAKER_NAMES

def clean_text(text):
    return " ".join(text.split()).strip()

def passes_filters(entry, min_words = 3, max_words = 60):
    speaker = entry.get("speaker","").strip()
    dialogue = entry.get("dialogue","").strip()

    if not speaker or not dialogue:
        return False
    if is_player_speaker(speaker):
        return False
    word_count = len(dialogue.split())
    if word_count < min_words or word_count > max_words:
        return False
    return True
