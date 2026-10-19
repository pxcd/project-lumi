"""Autoregressive text generation for Project Lumi."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

from lumi.train import LumiTrainer


def generate(model, tokenizer, prompt, max_tokens, temperature, seed):
    """Generate text using temperature sampling."""
    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    mx.random.seed(seed)
    model.eval()

    token_ids = tokenizer.encode(prompt).ids

    if not token_ids:
        raise ValueError("Prompt produced no tokens")

    generated = list(token_ids)

    for _ in range(max_tokens):
        context = generated[-model.config.context_length:]

        inputs = mx.array([context], dtype=mx.int32)

        logits = model(inputs)[0, -1, :]
        logits = logits / temperature

        next_token = mx.random.categorical(logits)
        mx.eval(next_token)

        generated.append(int(next_token.item()))

    continuation = tokenizer.decode(
        generated[len(token_ids):],
        skip_special_tokens=False,
    )

    return continuation


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/lumi-pretrain",
    )
    parser.add_argument(
        "--tokenizer",
        default="data/prepared/tinystories/tokenizer.json",
    )
    parser.add_argument(
        "--prompt",
        default="Once upon a time, there was a little girl named Lily.",
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="data/first-words.txt",
    )

    args = parser.parse_args()

    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Never overwrite Lumi's historical first response.
    if output.exists():
        parser.error(f"Output already exists: {output}")

    trainer = LumiTrainer.load_checkpoint(args.checkpoint)
    tokenizer = Tokenizer.from_file(args.tokenizer)

    if tokenizer.get_vocab_size() != trainer.model.config.vocab_size:
        raise ValueError("Tokenizer and model vocabulary sizes differ")

    print("Generating Lumi's first words...", flush=True)

    continuation = generate(
        trainer.model,
        tokenizer,
        args.prompt,
        args.max_tokens,
        args.temperature,
        args.seed,
    )

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": "Project Lumi v1",
        "parameters": 29398016,
        "checkpoint": args.checkpoint,
        "training_step": trainer.step,
        "prompt": args.prompt,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_new_tokens": args.max_tokens,
    }

    text = (
        "PROJECT LUMI — FIRST WORDS\n"
        "==========================\n\n"
        + json.dumps(record, indent=2)
        + "\n\nGENERATED TEXT\n"
        "==============\n\n"
        + args.prompt
        + continuation
        + "\n"
    )

    # Exclusive creation prevents accidental overwrites.
    with output.open("x", encoding="utf-8") as file:
        file.write(text)

    print("\n" + "=" * 50)
    print("LUMI'S FIRST WORDS")
    print("=" * 50)
    print(args.prompt + continuation)
    print("\n" + "=" * 50)
    print(f"Preserved in: {output}")


if __name__ == "__main__":
    main()
