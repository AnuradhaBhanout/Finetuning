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



def format_example(entry):
    speaker = clean_text(entry["speaker"])
    dialogue = clean_text(entry["dialogue"])
    context = clean_text(entry.get("context","General"))

    user_turn = f"[Situation: {context}]"

    messages = [
        {"role":"system","content":SYSTEM_TEMPLATE.format(speaker=speaker)},
        {"role":"user","content":user_turn},
        {"role":"assistant","content":dialogue},
    ]

    return {"messages":messages,"speaker":speaker,"context":context}


def main():
    parser = argparse.ArgumentParser(description="Format scraped dialogue into chat training data.")
    parser.add_argument("--input",default="data/raw/dialogue_raw.jsonl")
    parser.add_argument("--outdir",default="data/processed_chat")
    parser.add_argument("--min-words",type=int,default=3)
    parser.add_argument("--max-words",type=int,default=60)
    parser.add_argument("--val-frac",type=float,default=0.05)
    parser.add_argument("--test-frac",type=float,default=0.05)
    parser.add_argument("--seed",type=int,default=42)
    args = parser.parse_args()

    raw = load_raw(args.input)
    print(f"Loaded {len(raw)} raw entries.")

    filtered = [e for e in raw if passes_filters(e,args.min_words,args.max_words)]

    print(f"After filtering: {len(filtered)} entries.")

    seen = set()
    deduped = []

    for e in filtered:
        key = (e["speaker"].strip().lower(),e["dialogue"].strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)
    print(f"After deduplication: {len(deduped)} entries.")

    formatted = [format_example(e) for e in deduped]

    random.seed(args.seed)
    random.shuffle(formatted)

    n = len(formatted)
    n_val = int(n * args.val_frac)
    n_test = int(n * args.test_frac)
    n_train = n - n_val - n_test

    train = formatted[:n_train]
    val = formatted[n_train:n_train+n_val]
    test = formatted[n_train+n_val:]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True,exist_ok=True)

    for name ,split in [("train",train),("val",val),("test",test)]:
        out_path = outdir / f"{name}.jsonl"
        with open(out_path,"w",encoding="utf-8") as f:
            for ex in split:
                f.write(json.dumps(ex,ensure_ascii=False)+"\n")
        print(f"  {name}: {len(split)} examples -> {out_path}")

    print(f"\nDone. {n_train} train / {n_val} val / {n_test} test examples.")

    print("\n--- Sample formatted example ---")

    if train:
        print(json.dumps(train[0],indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
