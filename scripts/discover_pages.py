import json
import time
import argparse
from pathlib import Path
 
import requests
 
API_URL = "https://elderscrolls.fandom.com/api.php"
HEADERS = {
    "User-Agent": "SkyrimDialogueResearch/1.0 (educational fine-tuning project; contact: your_email@example.com)"
}
 
# The main character category for Skyrim on this wiki.
DEFAULT_CATEGORY = "Category:Skyrim: Characters"
 
 
def fetch_category_members(category, cmtype="page", cmcontinue=None):
    """Fetch one page of category members (up to 500)."""
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": cmtype,
        "cmlimit": "500",
        "format": "json",
        "formatversion": "2",
    }
    if cmcontinue:
        params["cmcontinue"] = cmcontinue
 
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()
 
 
def get_all_pages_in_category(category, delay=1.0):
    """Walk pagination to collect every page title in a category."""
    titles = []
    cmcontinue = None
 
    while True:
        data = fetch_category_members(category, cmtype="page", cmcontinue=cmcontinue)
 
        if "error" in data:
            print(f"  [ERROR] {data['error'].get('info')}")
            break
 
        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members)
        print(f"  ...collected {len(titles)} page titles so far")
 
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
        time.sleep(delay)
 
    return titles
 
 
def main():
    parser = argparse.ArgumentParser(description="Discover NPC page titles via wiki category.")
    parser.add_argument("--category", default=DEFAULT_CATEGORY,
                         help="Category to walk, e.g. 'Category:Skyrim: Characters'")
    parser.add_argument("--out", default="data/raw/npc_pages.txt",
                         help="Output text file, one page title per line.")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()
 
    print(f"Fetching pages in '{args.category}'...")
    titles = get_all_pages_in_category(args.category, delay=args.delay)
 
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for t in titles:
            f.write(t + "\n")
 
    print(f"\nDone. {len(titles)} page titles saved to {out_path}")
    if len(titles) == 0:
        print("WARNING: 0 pages found. The category name may be wrong — "
              "check the exact category name on the wiki and retry.")
 
 
if __name__ == "__main__":
    main()