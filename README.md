# Skyrim NPC Dialogue Generator

A LoRA fine-tuned GPT-2 model that generates in-character Skyrim NPC dialogue on demand, deployed as a Discord bot. Built end-to-end over a weekend: data scraping, cleaning, fine-tuning, evaluation, and deployment.

**Live in Discord:**

> **/npc** `character: Belethor` `situation: Greeting a customer entering his shop`
> **Belethor**: *"I'm here to make some tea."*

## Why this project

Fine-tuning is often demoed on toy datasets with pre-cleaned inputs. This project instead starts from raw wiki markup on a public game wiki, builds a real scraping + cleaning pipeline, and ships the result as something interactive — a Discord bot anyone in a server can talk to.

## Pipeline

```
discover_pages.py   ─▶  finds all NPC page titles via the wiki category API
        │
        ▼
scrape.py            ─▶  pulls raw wikitext, extracts dialogue with two regex patterns
        │
        ▼
format_data.py        ─▶  filters, dedupes, formats into a prompt template, splits train/val/test
        │
        ▼
train.py (Colab, T4)  ─▶  LoRA fine-tune of GPT-2 via trl's SFTTrainer
        │
        ▼
generate.py / generate_base.py  ─▶  inference + before/after comparison
        │
        ▼
bot.py                 ─▶  Discord slash-command bot serving the fine-tuned model
```

## Data

- **Source**: [Elder Scrolls Fandom wiki](https://elderscrolls.fandom.com), via the MediaWiki API (`action=parse&prop=wikitext`) rather than rendered HTML — dialogue on this wiki lives inside JS-collapsible sections that don't appear in plain HTML scrapes.
- **Two dialogue patterns** required distinct extraction logic:
  - `{{AudioQuote|dialogue|speaker|...}}` — single attributed lines
  - `'''Speaker:''' ''"dialogue text"''` — multi-turn conversation blocks
- **Discovery**: walked `Category:Skyrim: Characters` to auto-collect 1,168 NPC page titles rather than hand-picking characters.
- **Result**: 8,304 raw extracted lines → 4,357 examples after filtering (player-voiced lines removed, deduped, length-filtered to 3–60 words) → split 90/5/5 into train/val/test.

## Model & training

| Choice | Value | Why |
|---|---|---|
| Base model | GPT-2 small (124M) | Fits fully in a T4's memory at fp16 — no need for 4-bit quantization |
| Method | LoRA (not QLoRA) | QLoRA's quantization overhead buys memory savings this model doesn't need; LoRA alone already gives a small, portable adapter |
| Target modules | `c_attn` | GPT-2's combined QKV projection |
| Rank / alpha | r=8, alpha=16 | Standard small-adapter defaults (alpha = 2×r) |
| Trainer | `SFTTrainer` (trl) | Handles the causal-LM SFT loop, masking, and eval out of the box |
| Epochs | 3 | ~5.5 minutes on a T4 |

**Prompt format** (also what the Discord bot constructs at inference time):
```
### Character: Belethor
### Situation: Greeting a customer entering his shop
### Dialogue: <the model completes here><|endoftext|>
```

## Results

Training loss dropped from **4.71 → 2.94**; eval loss from **2.86 → 2.63**, improving at every checkpoint with no sign of overfitting in this 3-epoch run. Token-level accuracy rose from **0.28 → 0.53**.

### Before vs. after fine-tuning

Same prompt, same decoding settings — the only difference is the LoRA adapter.

**Prompt:** `Character: Belethor / Situation: Greeting a customer entering his shop`

| Base GPT-2 | Fine-tuned |
|---|---|
| *"The protagonist asks the girl if she can buy him some of her clothes... they talk about what happened at"* | *"I thought I was going to give you a good deal on something."* |

**Prompt:** `Character: A guard / Situation: Warning a stranger to watch their step`

| Base GPT-2 | Fine-tuned |
|---|---|
| *"None. A man with the long black hair and glasses is sitting next, wearing an outfit he calls 'The Black Man'..."* | *"I'll give you the dagger."* |

Base GPT-2 doesn't recognize the prompt template at all — it treats `### Character:` as literal text, invents unrelated fields, and drifts into generic forum/fanfiction-style rambling. The fine-tuned model reliably stays in the structured format, stays on-topic for the given situation, and produces a distinct tone per character (shopkeeper pitches from Belethor, menace from bandits, warmth from innkeepers).

One early issue — repetition loops (e.g. *"I'm not going to lie. I'm not going to lie."*) — was fixed via decoding parameters (`repetition_penalty=1.3`, `no_repeat_ngram_size=3`) rather than retraining, since it was a decoding-time failure mode, not a training one.

## Deployment

Packaged as a Discord bot (`bot.py`) using `discord.py`'s slash command interface:

```
/npc character:<name> situation:<short description>
```

The model loads once at startup and stays resident in memory, so each `/npc` call only pays the generation cost, not a reload. Runs on CPU locally (no GPU required for inference at this model size).

## Stack

Python · Hugging Face `transformers`, `peft`, `trl`, `datasets` · PyTorch · Google Colab (T4, training) · `discord.py` · MediaWiki API

## Limitations & future work

- Single-game dataset (Skyrim only) — kept deliberately scoped rather than risking style bleed from mixing franchises; a `### Game:` field in the prompt template would be the clean way to extend to multiple games later.
- Small base model (124M) — dialogue is coherent but generic in places; a larger base model or more training data would likely improve character distinctiveness further.
- CPU inference is functional but not fast; a hosted GPU endpoint would be the next step for a public-facing deployment.
