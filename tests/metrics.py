import re
from collections import Counter


MARKUP_RE = re.compile(r"\[\[|\]\]|\{\{|\}\}|<ref|'''|==|\|display")

REFUSAL_MARKERS = [
    "i'm sorry", "i am sorry", "i apologize", "as an ai", "i cannot",
    "i can't assist", "i can't help", "it's important to", "it is important to",
    "i understand that", "please note", "as a language model",
    "i'm just", "i must clarify", "however, it's worth",
]

MIN_CHARS,MAX_CHARS = 8, 400

def leaked_markup(text:str) ->bool:
    # some of the scraped corpus had raw [[link|display]] in it. scraper's
    # fixed now, this just catches it if it ever comes back.
    return bool(MARKUP_RE.search(text))

def refusal_register(text:str)->bool:
    # bandits going polite instead of threatening
    low = text.lower()
    return any(m in low for m in REFUSAL_MARKERS)

def max_ngram_repeat(text:str,n:int = 3)->int:
    # RunPod endpoint dropped repetition_penalty once and outputs looped.
    words = text.lower().split()
    if len(words)<n:
        return 0
    grams = [ tuple(words[i:i+n]) for i in range(len(words)- n+1)]
    return Counter(grams).most_common(1)[0][1]

def length_ok(text: str)-> bool:
    return MIN_CHARS <= len(text) <= MAX_CHARS

def distinct_n(texts: list,n:int =2)-> float:
    # brynjolf keeps saying the same thing — this is how you catch "flat"
    grams = []
    for t in texts:
        words = t.lower().split()
        grams += [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return len(set(grams))/len(grams) if grams else 0


def selfcheck():
    assert leaked_markup("Hello [[Whiterun|the city]]")
    assert not leaked_markup("Money or your life, traveler.")
    assert refusal_register("I'm sorry, I cannot roleplay violence.")
    assert not refusal_register("You'll regret crossing me.")
    assert max_ngram_repeat("a b c a b c a b c")== 3
    assert max_ngram_repeat("money or your life") == 1
    assert not length_ok("")
    assert length_ok("Money or your life.")
    assert distinct_n(["a b c", "a b c"], 2) == 0.5
    assert distinct_n(["a b", "c d"], 2) == 1.0
    print("ok")



if __name__ == "__main__":
    selfcheck()