"""
generate.py — Load the fine-tuned GPT-2 + LoRA adapter and generate sample
NPC dialogue, so you can eyeball whether the fine-tune actually worked.

Run this in the same Colab notebook (after training), or locally if you've
downloaded the adapter folder.

Usage (in Colab):
    !python generate.py \
        --adapter_dir /content/drive/MyDrive/Finetuning/checkpoints/gpt2-skyrim-lora/final \
        --base_model gpt2
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# A handful of test prompts, using the SAME format the model was trained on.
# Situation is deliberately varied to see how well the model adapts tone.
TEST_PROMPTS = [
    ("Belethor", "Greeting a customer entering his shop"),
    ("Nazeem", "Boasting about the Cloud District"),
    ("A guard", "Warning a stranger to watch their step"),
    ("An innkeeper", "Welcoming a weary traveler"),
    ("A bandit", "Threatening the player on the road"),
]


# def build_prompt(character, situation):
#     # Matches format_data.py's template, but cut off after "### Dialogue:"
#     # so the model has to complete it.
#     return f"### Character: {character}\n### Situation: {situation}\n### Dialogue:"

#qwen prompt
def build_prompt(tokenizer, character, situation):
    messages = [
        {"role": "system", "content": f"You are {character}, an NPC in The Elder Scrolls V: Skyrim. Stay fully in character. Respond the way {character} would actually speak — in tone, vocabulary, and attitude — in one or two short lines."},
        {"role": "user", "content": f"[Situation: {situation}]"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser(description="Generate sample NPC dialogue from the fine-tuned model.")
    parser.add_argument("--adapter_dir", required=True, help="Path to the saved LoRA adapter (the 'final' folder).")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")#"gpt2")
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--num_samples", type=int, default=2, help="Generations per prompt, to see variety.")
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)

    print(f"Loading LoRA adapter from: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"\nGenerating on device: {device}\n")
    print("=" * 70)

    for character, situation in TEST_PROMPTS:
        #prompt = build_prompt(character, situation)
        prompt = build_prompt(tokenizer, character, situation)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        print(f"\n[{character}] — {situation}")
        print("-" * 70)

        for i in range(args.num_samples):
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.9,
                    repetition_penalty=1.3,   # penalizes tokens already used, kills "I'm not going to lie. I'm not going to lie." loops
                    no_repeat_ngram_size=3,   # hard-blocks any 3-word phrase from repeating verbatim
                    pad_token_id=tokenizer.eos_token_id,
                )
            # generated = tokenizer.decode(output[0], skip_special_tokens=True)
            # # Only print what came after the prompt, for readability
            # completion = generated[len(prompt):].strip()

            input_length = inputs["input_ids"].shape[1]
            completion_ids = output[0][input_length:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            print(f"  ({i+1}) {completion}")

    print("\n" + "=" * 70)
    print("Done. Eyeball these for: does dialogue match the character/situation,")
    print("does it sound Skyrim-ish, and does it stay coherent (not repeat/ramble)?")


if __name__ == "__main__":
    main()