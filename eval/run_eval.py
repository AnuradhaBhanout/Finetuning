import argparse
import json
import os
import statistics
import requests

from metrics import refusal_register,length_ok,max_ngram_repeat,leaked_markup,distinct_n,selfcheck,DEGENERATE_REPEAT_THRESHOLD

GEN_ENDPOINT = os.getenv("GEN_ENDPOINT")
GEN_API_KEY = os.getenv("GEN_API_KEY")


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
            "input":{
            "model":model,
            "prompt": build_prompt(character,situation),
            "max_tokens":80,
            "temperature":0.8,
            "top_p":0.9,
            "repetition_penalty":1.3,
            "seed":seed,
            }
        },
        timeout=120,
    )
    resp.raise_for_status()
#    return resp.json()["choices"][0]["text"].strip()
    data = resp.json()
   # print(json.dumps(data, indent=2)[:800])
    return data["output"][0]["choices"][0]["tokens"][0].strip()


def run(cases: list,label:str,model:str,samples:int) -> dict:
    rows = []
    for case in cases:
        for s in range(samples):
            text = generate(case["character"],case["situation"],model,seed=1000+s)
            rows.append({
                "id":case["id"],
                "sample":s,
                "character":case["character"],
                "tags": case["tags"],
                "output":text,
                "leaked_markup":leaked_markup(text),
                "refusal_register":refusal_register(text),
                "max_3gram_repeat": max_ngram_repeat(text),
                "length_ok":length_ok(text),
                "chars": len(text),
            })
           # print(f" {case['id']}[{s}] {text[:70]!r}")
    return {"label": label, "model":model, "samples":samples, "rows":rows}


def summarize(result: dict)-> dict:
    rows = result["rows"]
    n = len(rows)

    def rate(key):
        return sum(r[key] for r in rows)/n

    def tag_rate(tag,key):
        sub = [ r for r in rows if tag in r["tags"]]
        return sum(r[key] for r in sub)/ len(sub) if sub else 0.0
    degenerate= [ r for r in rows if r["mean_3gram_repeat"] > DEGENERATE_REPEAT_THRESHOLD]
    clean= [ r for r in rows if  r["mean_3gram_repeat"] <= DEGENERATE_REPEAT_THRESHOLD]
    by_char = {}
    for r in rows:
        by_char.setdefault(r["character"],[]).append(r["output"])

    return{
        "n":n,
        "markup_leak_rate": rate("leaked_markup"),
        "refusal_rate": rate("refusal_register"),
        "refusal_rate_morally_complicated": tag_rate("morally-complicated","refusal_register"),
        "length_ok_rate":rate("length_ok"),
        "mean_3gram_repeat": statistics.mean(r["max_3gram_repeat"] for r in rows),
        "mean_3gram_repeat_excl_degenerate": (statistics.mean(r["max_3gram_repeat"] for r in clean) if clean else 0),
        "degenerate_count":len(degenerate),
        "degenerate_rate":len(degenerate)/n,
        "degenerate_ids":[f"{r['id']}[{r['sample']}]" for r in degenerate],
        "worst_3gram_repeat": max(r["max_3gram_repeat"] for r in rows),
        "mean_chars": statistics.mean(r["chars"] for r in rows),
        "diversity_distinct2_overall": distinct_n([r["output"] for r in rows], 2),
        "diversity_by_character": dict(sorted(
            {c: distinct_n(o, 2) for c, o in by_char.items()}.items(),
            key=lambda kv: kv[1])),
    }

def print_summary(label: str, s: dict):
    print(f"\n=== {label}  (n={s['n']}) ===")
    print(f"  markup leak rate         {s['markup_leak_rate']:.1%}   (want 0%)")
    print(f"  refusal register         {s['refusal_rate']:.1%}   (want <5%)")
    print(f"    morally-complicated    {s['refusal_rate_morally_complicated']:.1%}")
    print(f"  length ok                {s['length_ok_rate']:.1%}   (want >95%)")
    print(f"  mean 3-gram repeat       {s['mean_3gram_repeat']:.2f}   (want ~1.0)")
    print(f"    excl. degenerate rows  {s['mean_3gram_repeat_excl_degenerate']:.2f}")
    print(f"  degenerate generations   {s['degenerate_count']} ({s['degenerate_rate']:.1%})")
    if s["degenerate_ids"]:
        print(f"    -> {', '.join(s['degenerate_ids'])}")
    print(f"  worst 3-gram repeat      {s['worst_3gram_repeat']}")
    print(f"  distinct-2 overall       {s['diversity_distinct2_overall']:.3f}")
    print("  least varied characters:")
    for c, d in list(s["diversity_by_character"].items())[:5]:
        print(f"    {c:24s} {d:.3f}")

def compare(a_path: str, b_path: str):
    a, b =( json.load(open(p)) for p in (a_path, b_path))
    sa, sb = summarize(a),summarize(b)
    print(f"\n{'metric':38s} {a['label']:>12s} {b['label']:>12s}   delta")
    for k in ("markup_leak_rate", "refusal_rate", "refusal_rate_morally_complicated",
              "length_ok_rate",  "mean_3gram_repeat_excl_degenerate", "degenerate_rate", "diversity_distinct2_overall"):
        print(f"{k:38s} {sa[k]:12.3f} {sb[k]:12.3f}   {sb[k] - sa[k]:+.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default="eval/cases.json")
    p.add_argument("--label", help="name for this run, e.g. 'base' or 'lora-v1'")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--samples", type=int, default=3, help="generations per case")
    p.add_argument("--out", default="eval/results.json")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    p.add_argument("--selfcheck", action="store_true")

    args = p.parse_args()

    if args.selfcheck:
        selfcheck()

    elif args.compare:
        compare(*args.compare)

    else:
        cases = json.load(open(args.cases))["cases"]
        result = run(cases,args.label, args.model, args.samples)
        json.dump(result, open(args.out, "w"), indent=2)
        print_summary(args.label, summarize(result))
        print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main()