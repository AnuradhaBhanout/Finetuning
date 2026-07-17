"""
scrape.py — Pull Skyrim NPC dialogue from the Elder Scrolls Fandom wiki.

Uses the MediaWiki API (api.php?action=parse&prop=wikitext) instead of
scraping rendered HTML, since dialogue lives inside wiki templates/markup
that don't render cleanly as plain text.

Two dialogue formats found on NPC pages:
  1. {{AudioQuote|text|speaker|game|audiofile}}  -> single attributed lines
  2. '''Speaker:''' ''"dialogue text"''          -> full conversations

Output: data/raw/dialogue_raw.jsonl
Each line: {"source_page": "...", "context": "<section>", "speaker": "...", "dialogue": "..."}
"""

import re
import time
import json
import argparse
from pathlib import Path

import requests

API_URL = "https://elderscrolls.fandom.com/api.php"
HEADERS = {
    "User-Agent": "SkyrimDialogueResearch/1.0 (educational fine-tuning project; contact: your_email@example.com)"
}

# Named NPC pages tend to have real "==Conversations==" sections with dialogue.
DEFAULT_PAGES = [
    "Nazeem",
    "Belethor",
    "Anoriath",
    "Ahlam",
]

# Pattern 1: {{AudioQuote|dialogue text|Speaker|...}}
AUDIOQUOTE_PATTERN = re.compile(
    r'\{\{AudioQuote\|([^|}]+)\|([^|}]+)(?:\|[^}]*)?\}\}'
)

# Pattern 2: '''Speaker:''' ''"dialogue text"''
CONVERSATION_PATTERN = re.compile(
    r"'''([^:'{}\[\]]{1,40}):'''\s*''\"([^\"]{3,400})\"''"
)

# Matches wikitext section headers at any depth, e.g. ==Conversations==, ===Anoriath===
SECTION_PATTERN = re.compile(r'^={2,4}\s*(.+?)\s*={2,4}\s*$', re.MULTILINE)


def fetch_wikitext(page_title):
    """Fetch raw wikitext for a single page via the MediaWiki API."""
    params = {
        "action": "parse",
        "page": page_title,
        "format": "json",
        "prop": "wikitext",
        "formatversion": "2",
    }
    try:
        resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"  [ERROR] Request failed for '{page_title}': {e}")
        return None
    except json.JSONDecodeError:
        print(f"  [ERROR] Non-JSON response for '{page_title}'")
        return None

    if "error" in data:
        print(f"  [ERROR] API error for '{page_title}': {data['error'].get('info')}")
        return None

    try:
        return data["parse"]["wikitext"]
    except KeyError:
        print(f"  [ERROR] Unexpected response shape for '{page_title}'")
        return None


def section_for_position(pos, section_bounds):
    """Given a character offset, find the nearest preceding section header."""
    current = "General"
    for start, name in section_bounds:
        if start <= pos:
            current = name
        else:
            break
    return current


def clean_wikitext_fragment(text):
    """Strip common inline wiki markup like [[link|display]] -> display."""
    text = re.sub(r'\[\[[^\|\]]+\|([^\]]+)\]\]', r'\1', text)  # [[target|display]]
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)             # [[target]]
    text = re.sub(r"'''(.*?)'''", r'\1', text)                  # bold
    text = re.sub(r"''(.*?)''", r'\1', text)                    # italic
    return text.strip()


def extract_dialogue(wikitext, page_title):
    """Pull dialogue lines out of wikitext using both known patterns."""
    entries = []
    section_bounds = [(m.start(), m.group(1)) for m in SECTION_PATTERN.finditer(wikitext)]

    # Pattern 1: AudioQuote templates
    for match in AUDIOQUOTE_PATTERN.finditer(wikitext):
        raw_dialogue, raw_speaker = match.group(1), match.group(2)
        dialogue = clean_wikitext_fragment(raw_dialogue)
        speaker = clean_wikitext_fragment(raw_speaker)
        if len(dialogue.split()) < 2:
            continue
        entries.append({
            "source_page": page_title,
            "context": section_for_position(match.start(), section_bounds),
            "speaker": speaker,
            "dialogue": dialogue,
        })

    # Pattern 2: '''Speaker:''' ''"text"''  conversation lines
    for match in CONVERSATION_PATTERN.finditer(wikitext):
        raw_speaker, raw_dialogue = match.group(1), match.group(2)
        speaker = clean_wikitext_fragment(raw_speaker)
        dialogue = clean_wikitext_fragment(raw_dialogue)
        if len(dialogue.split()) < 2:
            continue
        # Filter out false positives where "speaker" isn't really a name
        # (e.g. captured stray bold text with no name-like shape)
        if len(speaker.split()) > 4:
            continue
        entries.append({
            "source_page": page_title,
            "context": section_for_position(match.start(), section_bounds),
            "speaker": speaker,
            "dialogue": dialogue,
        })

    return entries


def main():
    parser = argparse.ArgumentParser(description="Scrape Skyrim NPC dialogue from Fandom.")
    parser.add_argument(
        "--pages", nargs="*", default=None,
        help="Wiki page titles to scrape directly (underscores instead of spaces)."
    )
    parser.add_argument(
        "--pages-file", default=None,
        help="Path to a text file with one page title per line (e.g. from discover_pages.py)."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only scrape the first N pages from --pages-file (useful for test runs)."
    )
    parser.add_argument(
        "--out", default="data/raw/dialogue_raw.jsonl",
        help="Output JSONL path."
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds to wait between requests (be polite to the wiki)."
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to the output file instead of overwriting (useful for resuming a run)."
    )
    args = parser.parse_args()

    if args.pages_file:
        with open(args.pages_file, "r", encoding="utf-8") as f:
            pages = [line.strip() for line in f if line.strip()]
        if args.limit:
            pages = pages[:args.limit]
    elif args.pages:
        pages = args.pages
    else:
        pages = DEFAULT_PAGES

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_mode = "a" if args.append else "w"
    total_entries = 0
    with open(out_path, write_mode, encoding="utf-8") as f:
        for i, page in enumerate(pages):
            print(f"[{i+1}/{len(pages)}] Fetching '{page}'...")
            wikitext = fetch_wikitext(page)

            if wikitext is None:
                print(f"  Skipping '{page}' — no content retrieved.")
                continue

            entries = extract_dialogue(wikitext, page)
            print(f"  Extracted {len(entries)} dialogue lines.")

            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
            total_entries += len(entries)

            if i < len(pages) - 1:
                time.sleep(args.delay)

    print(f"\nDone. {total_entries} total dialogue lines saved to {out_path}")
    if total_entries == 0:
        print("WARNING: No dialogue extracted. Inspect the debug wikitext output "
              "and adjust the regex patterns before scaling up.")


if __name__ == "__main__":
    main()