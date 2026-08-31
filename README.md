# SkyrimNPCBot

A Discord bot that generates in-character Skyrim NPC dialogue. Type `/npc character:Belethor situation:Greeting a customer` and get a line back that (usually) sounds like Belethor.

**At a glance**
- Qwen2.5-1.5B-Instruct + LoRA adapter (r=8), fine-tuned on 4,357 scraped NPC lines
- Served via vLLM on RunPod serverless, warm response in 2 to 3 seconds
- Evaluated head to head against the base model on a 40-case harness: refusal-register dropped 0.8% to 0%, repetition and diversity statistically flat, one isolated failure mode found and documented
- Built after a year of self-studying AI/ML, coming from five years in Unreal Engine and VR game development

## What it does

You give it a character name and a short situation. It returns one or two lines of dialogue in that character's voice, styled like Skyrim's in-game barks.

```
/npc character:Bandit situation:Threatening the player on the road
> "Money or your life, traveler. I won't ask twice."
```

![Belethor and Kodlak responding in character](assets/demo-1.png)
![Thalmor Justiciar and Bandit Chief responding in character](assets/demo-2.png)

## Why I built this

The real goal underneath it: figure out whether small, cheap fine-tunes can hold a character's voice well enough to be useful, since that's the foundation for a longer-term idea I have about AI-driven NPCs in location-based VR arcades. This project is the "can I even get the basics right" step before that.

## How it's built

**Data.** Scraped from the Elder Scrolls Fandom wiki via the MediaWiki API's raw wikitext endpoint, not rendered HTML, since dialogue lives inside templates that don't come through cleanly as plain text. 4,357 examples across 1,168 NPC pages after filtering and dedup.

**Model.** Started with GPT-2 (124M). Didn't work: more data per character didn't help, eval loss stayed high with no overfitting signal, which meant it was a capacity problem, not a data problem. Switched to Qwen2.5-1.5B-Instruct, trained a LoRA adapter on top. Three epochs on a Colab T4, about 38 minutes. Perplexity dropped 61.4 to 2.44, though that number alone doesn't mean much since it's scored on the training distribution. See Evaluation below for the numbers that actually mattered.

**The wikitext bug.** About three months in, roughly 8% of training examples still had raw `[[link|display]]` markup sitting in them unescaped. Two separate scraper bugs, a regex that missed a malformed link pattern and a section-header function that skipped cleaning entirely. Fixed both, rescraped, retrained. The model still trained and produced plausible output the whole time; it just quietly got worse in a way you only catch by going looking.

**Deployment, round one.** EC2 t3.micro, model quantized to GGUF via llama.cpp, no GPU on the free tier. Worked. Four to five minutes per response. Technically functional, not usable.

**Deployment, round two.** Moved inference to a serverless GPU endpoint on RunPod, vLLM, pay-per-second. The Discord bot stayed on the free EC2 box and just makes an HTTP call now. Cold start on an idle worker is about ninety seconds; warm, two to three seconds.

**A bug I didn't expect.** After the GPU move, generations occasionally repeated across unrelated requests, one line bleeding into the next, or spiraling on a single word. My new request payload wasn't passing `repetition_penalty`, `top_p`, or `max_tokens` the way the old CPU script had, so the endpoint silently fell back to more permissive defaults. Added the parameters back explicitly, problem gone. Swapping infrastructure can quietly drop behavior you already built and tested for, even with the model unchanged.

## Evaluation

Until recently this was me reading generations and forming opinions. That's how I caught the RLHF drift and the repetition bug, but "this feels off" isn't a number.

I built a harness (`eval/`) around 40 hand-written cases, merchants, guards, bandits, followers, mages, plus a few deliberately awkward prompts. Each case runs three times against both the base model and the fine-tune, same sampling settings the bot actually uses. Five deterministic and statistical checks, no LLM judge, each tied to a failure I'd actually seen: markup leaking into output, RLHF-style refusal language, degenerate repetition, broken length, lexical diversity per character.

**Base model versus fine-tune, 120 generations each:**

| metric | base | lora-v1 | verdict |
|---|---|---|---|
| refusal register | 0.8% | 0.0% | fine-tune wins, base occasionally drifted apologetic, lora never does |
| refusal on morally-complicated | 0.0% | 0.0% | tie, both clean |
| markup leak | 0% | 0% | tie, clean |
| length ok | 98.3% | 98.3% | tie |
| mean 3-gram repeat | 1.000 | 1.483 (1.008 excl. outlier) | essentially tied once the one outlier is isolated |
| diversity, distinct-2 | 0.929 | 0.898 | fine-tune slightly less varied overall |
| Bandit diversity specifically | 0.989 | 0.496 | driven almost entirely by that same single 58-token loop |

That one outlier: a bandit, asked to write a friendly kitten poem, output the word "Ha" fifty eight times and stopped. One row out of 120. Pull it out and repetition goes 1.01 versus 1.00, diversity goes 0.926 versus 0.929, both flat.

I used to think Brynjolf specifically came out flat. Turns out that was mostly vibes, his diversity score actually sits near the top of the range, not the bottom. I'd based the original read on maybe a dozen of his lines and just not liking them. Worth saying plainly, since catching a gut read that turns out wrong is exactly what a real harness is for.

What this doesn't measure: whether a character's *personality* stays consistent across generations, since lexical diversity is about word choice, not whether the model's read on who Brynjolf is holds together response to response. That would need a rubric an LLM judge scores against, or a lot more manual reading. Neither happened yet.

```bash
python eval/run_eval.py --label base --model Qwen/Qwen2.5-1.5B-Instruct --out eval/results_base.json
python eval/run_eval.py --label lora-v1 --model <your-endpoint> --out eval/results_lora.json
python eval/run_eval.py --compare eval/results_base.json eval/results_lora.json
```

## What works and what doesn't

Most characters land well, guards sound like guards, merchants haggle like merchants, a Legate barking orders to crush a rebellion reads like something that could be in the game.

Not perfect. Bandits and other "morally complicated" characters occasionally swerve into apologetic RLHF-flavored register, though less often in the fine-tune than the base model per the eval above. Likely Qwen's instruction-tuning baked in enough "be nice and cooperative" that a LoRA this size, rank 8, about 4,000 examples, doesn't fully override it every time. There's also the one reproducible failure the eval caught directly: a threatening character asked to do something gentle can trigger a token repetition loop. Happened once in 120 generations. I know exactly which prompt triggers it and roughly why, which is more than I could say before building the harness.

## Cost and infrastructure

EC2 stays free tier regardless of deployment path. RunPod only charges while a worker runs and scales to zero idle, so monthly cost for a low-traffic bot is close to nothing. If traffic grew enough that cold starts became a real problem, the next lever is keeping a worker warm intentionally, a small ongoing cost for consistently fast responses.

## Stack

- **Data**: MediaWiki API, Python, regex-based wikitext cleaning
- **Training**: HuggingFace `transformers`/`peft`/`trl`, PyTorch, Google Colab (T4)
- **Model**: Qwen2.5-1.5B-Instruct + LoRA adapter (r=8)
- **Serving**: vLLM on RunPod serverless (GPU), llama.cpp/GGUF as a CPU fallback path
- **Bot**: `discord.py`, systemd service on AWS EC2 (t3.micro, free tier)
- **Eval**: custom harness (`eval/`), deterministic and statistical checks, no LLM judge

## What I'd do differently next time

Add the sampling parameters and evaluation harness before the first deployment, not after chasing a bug in production. Budget time upfront for the model-capacity question instead of discovering it after weeks of trying to fix GPT-2 with more data, the underfitting signal was there early: high loss, no overfitting gap, no improvement from more examples. I just didn't recognize it yet. And build the eval harness before forming opinions about individual characters, at least one of those opinions turned out to be wrong.