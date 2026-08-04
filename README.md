# SkyrimNPCBot

A Discord bot that generates in-character Skyrim NPC dialogue. Type `/npc character:Belethor situation:Greeting a customer` and get a line back that (usually) sounds like Belethor.

I built this as a portfolio project after a year of self-studying AI/ML, coming from five years in Unreal Engine and VR game development. The real goal underneath it: figure out whether small, cheap fine-tunes can hold a character's voice well enough to be useful, since that's the foundation for a longer-term idea I have about AI-driven NPCs in location-based VR arcades. This project is the "can I even get the basics right" step before that.

## What it does

You give it a character name and a short situation. It returns one or two lines of dialogue in that character's voice, styled like Skyrim's in-game barks.

```
/npc character:Bandit situation:Threatening the player on the road
> "Money or your life, traveler. I won't ask twice."
```

## How it's built

**Data.** I scraped NPC dialogue from the Elder Scrolls Fandom wiki using the MediaWiki API's raw wikitext endpoint, not the rendered HTML, because dialogue lives inside templates that don't come through cleanly as plain text. Ended up with about 4,357 examples across 1,168 NPC pages after filtering and deduplication.

**Model.** I started with GPT-2 (124M parameters) because it was cheap to iterate on. It didn't work. More training data per character didn't help, and the eval loss stayed stubbornly high with no overfitting signal, which told me this wasn't a data problem. It was a capacity problem: the model was too small to hold anything beyond a shallow imitation. I switched to Qwen2.5-1.5B-Instruct (Apache 2.0 license, decent small-model benchmarks) and trained a LoRA adapter on top of it instead of the full model. Three epochs on a Colab T4, about 38 minutes. Perplexity dropped from 61.4 on the base model to 2.44 after fine-tuning.

**The wikitext bug.** About three months into the project I noticed roughly 8% of my training examples still had raw `[[link|display]]` markup sitting in them, unescaped. Traced it to two separate bugs in the scraper: one regex that missed a malformed link pattern, and a section-header function that never ran the cleaning step at all. Fixed both, reran the whole scrape, retrained. This is the kind of bug that's easy to miss because the model still trains and still produces plausible-looking output; it just quietly degrades quality in a way you won't catch unless you go looking.

**Deployment, round one.** I put the bot on an EC2 t3.micro (free tier) running the model quantized to GGUF via llama.cpp, since the free-tier box has no GPU and barely enough RAM. It worked, but a single Discord response took four to five minutes. Technically functional. Not something you'd actually want to use.

**Deployment, round two.** I moved inference to a serverless GPU endpoint on RunPod (vLLM, a 24GB card, pay-per-second). The Discord bot itself stayed on the free EC2 box, but now it just makes an HTTP call instead of loading the model locally. Cold start on an idle worker is around a minute and a half; once it's warm, generation takes two to three seconds. That's the difference between "cute prototype" and "something you'd put in an actual Discord server."

**A bug I didn't expect.** After switching to the GPU endpoint, I noticed the model would occasionally repeat itself across unrelated requests — one NPC's line would bleed into another's, or a response would spiral into repeating the same word over and over. Turned out my new request payload wasn't passing `repetition_penalty`, `top_p`, or `max_tokens` the way my original CPU-based generation script had. The GPU endpoint was silently falling back to more permissive defaults. Once I added those parameters back explicitly, the repetition problem went away. It's a good reminder that swapping infrastructure can silently drop behavior you'd built and tested for, even when the model itself hasn't changed at all.

## What works and what doesn't

Most characters land well. Guards sound like guards, merchants haggle like merchants, and a Legate barking orders to crush a rebellion reads like something that could genuinely be in the game.

It's not perfect. Brynjolf, specifically, tends to come out flat and a little incoherent regardless of the situation I give him — I suspect his training examples skewed toward a narrower emotional range than other characters. And every so often, especially on characters I'd call "morally complicated" like bandits, the model swerves into an apologetic, RLHF-flavored register that has nothing to do with Skyrim. My guess is that Qwen's instruction-tuning baked in a lot of "be nice and cooperative" behavior, and a LoRA adapter this size (rank 8, about 4,000 training examples) doesn't have enough weight to fully override that on every single generation. I looked into fixing this properly and decided it wasn't worth the effort for what this project is; I'm noting it here instead of hiding it.

## Cost and infrastructure notes

The EC2 box is free tier and stays that way regardless of how the model is deployed. The RunPod GPU endpoint only charges while a worker is actually running, and scales to zero when idle, so for a low-traffic Discord bot the actual monthly cost is close to nothing. If traffic ever grew enough that cold starts became a real problem, the next lever would be keeping a worker warm intentionally, which trades a small ongoing cost for consistently fast responses.

## Stack

- **Data**: MediaWiki API, Python, regex-based wikitext cleaning
- **Training**: HuggingFace `transformers`/`peft`/`trl`, PyTorch, Google Colab (T4)
- **Model**: Qwen2.5-1.5B-Instruct + LoRA adapter (r=8)
- **Serving**: vLLM on RunPod serverless (GPU), llama.cpp/GGUF as a CPU fallback path
- **Bot**: `discord.py`, deployed as a systemd service on AWS EC2 (t3.micro, free tier)

## What I'd do differently next time

I'd add the sampling parameters and evaluation harness before the first deployment, not after chasing a bug in production. And I'd budget time upfront for the model-capacity question instead of discovering it after weeks of trying to fix GPT-2 with more data. In hindsight, the underfitting signal was there early: high loss, no overfitting gap, no improvement from adding examples. I just didn't recognize the pattern yet.