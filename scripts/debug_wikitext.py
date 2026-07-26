"""One-off debug script: dump raw wikitext for a page so we can inspect its real structure."""
import requests
import sys

API_URL = "https://elderscrolls.fandom.com/api.php"
HEADERS = {"User-Agent": "SkyrimDialogueResearch/1.0 (educational project; contact: your_email@example.com)"}

page = sys.argv[1] if len(sys.argv) > 1 else "Nazeem"

params = {
    "action": "parse",
    "page": page,
    "format": "json",
    "prop": "wikitext",
    "formatversion": "2",
}

resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
resp.raise_for_status()
data = resp.json()

if "error" in data:
    print(f"API error: {data['error'].get('info')}")
    sys.exit(1)

wikitext = data["parse"]["wikitext"]

out_file = f"data/raw/debug_{page.replace('(', '').replace(')', '')}.txt"
with open(out_file, "w", encoding="utf-8") as f:
    f.write(wikitext)

print(f"Saved {len(wikitext)} characters to {out_file}")
print("\n--- First 2500 characters ---\n")
print(wikitext[:2500])