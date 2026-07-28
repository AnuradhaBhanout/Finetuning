"""
format_data.py — Clean and format scraped Skyrim dialogue into training data.

Pipeline:
  1. Load raw scraped JSONL
  2. Filter out player/generic speakers (we only want NPC voice)
  3. Deduplicate on (speaker, dialogue)
  4. Filter by length (too short = noise, too long = unwieldy for a small model)
  5. Format into a consistent prompt template for causal LM fine-tuning
  6. Split into train/val/test and save

Input:  data/raw/dialogue_raw.jsonl
Output: data/processed/train.jsonl, val.jsonl, test.jsonl
        each line: {"text": "### Character: ...\n### Situation: ...\n### Dialogue: ...<|endoftext|>"}
"""

import json
import random
import argparse
from pathlib import Path

# Speakers that represent the player, not an NPC — exclude these.
PLAYER_SPEAKER_NAMES = {
    "dragonborn", "player", "you", "last dragonborn", "the dragonborn"
}

EOS_TOKEN = "<|endoftext|>"  # GPT-2's native end-of-sequence token


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
    text = " ".join(text.split())  # collapse whitespace/newlines
    return text.strip()


def passes_filters(entry, min_words=3, max_words=60):
    speaker = entry.get("speaker", "").strip()
    dialogue = entry.get("dialogue", "").strip()

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
    context = clean_text(entry.get("context", "General"))
    dialogue = clean_text(entry["dialogue"])

    text = (
        f"### Character: {speaker}\n"
        f"### Situation: {context}\n"
        f"### Dialogue: {dialogue}{EOS_TOKEN}"
    )
    return {"text": text, "speaker": speaker, "context": context}


def main():
    parser = argparse.ArgumentParser(description="Format scraped dialogue into training data.")
    parser.add_argument("--input", default="data/raw/dialogue_raw.jsonl")
    parser.add_argument("--outdir", default="data/processed")
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--max-words", type=int, default=60)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=40)
    args = parser.parse_args()

    raw = load_raw(args.input)
    print(f"Loaded {len(raw)} raw entries.")

    filtered = [e for e in raw if passes_filters(e, args.min_words, args.max_words)]
    print(f"After filtering (min {args.min_words}/max {args.max_words} words, excluding player lines): {len(filtered)} entries.")

    # Deduplicate on (speaker, dialogue) — same line said by same character counted once.
    seen = set()
    deduped = []
    for e in filtered:
        key = (e["speaker"].strip().lower(), e["dialogue"].strip().lower())
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
    val = formatted[n_train:n_train + n_val]
    test = formatted[n_train + n_val:]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        out_path = outdir / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in split:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(split)} examples -> {out_path}")

    print(f"\nDone. {n_train} train / {n_val} val / {n_test} test examples.")

    # Show a couple of formatted examples so it's easy to eyeball quality.
    print("\n--- Sample formatted examples ---")
    for ex in train[:3]:
        print(ex["text"])
        print("---")


if __name__ == "__main__":
    main()