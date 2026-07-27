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