#!/usr/bin/env python3
"""Reproducible generation comparison: Lumi v1.0.0 vs v1.1.

Generates 72 continuations (2 models × 6 prompts × 6 decoding configs)
and writes them to a JSONL file.

Decoding configs per model+prompt:
  - greedy (argmax, no seed)
  - temperature 1.0 sampling with seeds 42, 43, 44, 45, 46

Termination behavior:
  - v1.0: no EOS token; always exhausts max_tokens (finish_reason "length")
  - v1.1: stops on <|endofstory|> (id 0) or exhausts max_tokens

Decode convention (matches each model's existing code):
  - v1.0: skip_special_tokens=False (no special tokens exist)
  - v1.1: skip_special_tokens=True (strips <|endofstory|>)

Usage:
    python experiments/v1-vs-v1.1/evaluate.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]

V1_RELEASE_DIR = REPO_ROOT / "releases" / "lumi-v1-mlx"
V11_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "lumi-v1.1" / "best"
V11_TOKENIZER_PATH = REPO_ROOT / "data" / "prepared" / "tinystories-v1.1" / "tokenizer.json"
V11_DATASET_PATH = REPO_ROOT / "data" / "prepared" / "tinystories-v1.1" / "dataset.json"

OUTPUT_PATH = Path(__file__).resolve().parent / "results" / "generations.jsonl"

MAX_TOKENS = 200
SEEDS = [42, 43, 44, 45, 46]

PROMPTS = [
    "Once upon a time, there was a little girl named Lily.",
    "Once upon a time, there was a little boy named Tom.",
    "Lily found a mysterious box in the garden.",
    "The little dragon was afraid of fire.",
    "One day, a girl named Anna lost her favorite toy.",
    "The boy looked at his mother and said, \"",
]


@dataclass
class GenerationRecord:
    model_version: str
    checkpoint: str
    training_step: int
    prompt: str
    decoding: str
    seed: int | None
    temperature: float | None
    max_tokens: int
    generated_tokens: int
    finish_reason: str
    text: str
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def load_v1_model():
    """Load v1.0 model from the standalone release directory."""
    model_py = V1_RELEASE_DIR / "model.py"
    spec = importlib.util.spec_from_file_location("lumi_v1_model", model_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {model_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config_data = json.loads((V1_RELEASE_DIR / "config.json").read_text())
    config = module.LumiConfig(
        **{
            key: value
            for key, value in config_data.items()
            if key in module.LumiConfig.__dataclass_fields__
        }
    )
    model = module.LumiModel(config)
    model.load_weights(str(V1_RELEASE_DIR / "model.safetensors"))
    model.eval()
    mx.eval(model.parameters())

    tokenizer = Tokenizer.from_file(str(V1_RELEASE_DIR / "tokenizer.json"))
    return model, config, tokenizer


def load_v11_model():
    """Load v1.1 model from the version-2 checkpoint."""
    metadata = json.loads((V11_CHECKPOINT_DIR / "metadata.json").read_text())
    if metadata.get("version") != 2:
        raise ValueError("Expected a version 2 checkpoint")

    # Import the shared model module from the repo
    sys.path.insert(0, str(REPO_ROOT))
    from lumi.model import LumiConfig, LumiModel

    config = LumiConfig(**metadata["config"])
    model = LumiModel(config)
    model.load_weights(str(V11_CHECKPOINT_DIR / "model.safetensors"))
    model.eval()
    mx.eval(model.parameters())

    tokenizer = Tokenizer.from_file(str(V11_TOKENIZER_PATH))

    # Verify vocabulary size matches
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size {tokenizer.get_vocab_size()} != "
            f"model vocab size {config.vocab_size}"
        )

    # Get EOS token ID from dataset metadata
    dataset_meta = json.loads(V11_DATASET_PATH.read_text())
    eos_token_id = dataset_meta["eos_token_id"]

    # Verify the tokenizer actually has this token
    if tokenizer.token_to_id("<|endofstory|>") != eos_token_id:
        raise ValueError(
            f"Tokenizer <|endofstory|> id {tokenizer.token_to_id('<|endofstory|>')} "
            f"!= dataset eos_token_id {eos_token_id}"
        )

    training_step = metadata["step"]
    return model, config, tokenizer, eos_token_id, training_step


def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float | None,
    seed: int | None,
    eos_token_id: int | None,
    context_length: int,
    skip_special: bool,
) -> tuple[str, int, str]:
    """Generate text. Returns (continuation, generated_token_count, finish_reason)."""
    if temperature is not None:
        mx.random.seed(seed)
    model.eval()

    token_ids = tokenizer.encode(prompt).ids
    if not token_ids:
        raise ValueError("Prompt produced no tokens")

    generated = list(token_ids)
    finish_reason = "length"

    for _ in range(max_tokens):
        context = generated[-context_length:]
        inputs = mx.array([context], dtype=mx.int32)
        logits = model(inputs)[0, -1, :]

        if temperature is None:
            # Greedy: argmax
            next_token = mx.argmax(logits)
            mx.eval(next_token)
            next_token_id = int(next_token.item())
        else:
            # Temperature sampling
            logits = logits / temperature
            next_token = mx.random.categorical(logits)
            mx.eval(next_token)
            next_token_id = int(next_token.item())

        generated.append(next_token_id)

        if eos_token_id is not None and next_token_id == eos_token_id:
            finish_reason = "eos"
            break

    continuation = tokenizer.decode(
        generated[len(token_ids):],
        skip_special_tokens=skip_special,
    )

    return continuation, len(generated) - len(token_ids), finish_reason


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output JSONL path (created exclusively; will not overwrite)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help="Maximum tokens to generate",
    )
    args = parser.parse_args()

    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")

    # Never overwrite existing results
    if args.output.exists():
        print(f"ERROR: Output already exists: {args.output}", file=sys.stderr)
        print("Remove it or use --output to specify a different path.", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading v1.0 model...", flush=True)
    v1_model, v1_config, v1_tokenizer = load_v1_model()
    print(f"  v1.0: vocab={v1_config.vocab_size}, ctx={v1_config.context_length}", flush=True)

    print("Loading v1.1 model...", flush=True)
    v11_model, v11_config, v11_tokenizer, v11_eos_id, v11_step = load_v11_model()
    print(f"  v1.1: vocab={v11_config.vocab_size}, ctx={v11_config.context_length}, "
          f"EOS id={v11_eos_id}, step={v11_step}", flush=True)

    # Build the list of (model, config, tokenizer, eos_id, skip_special, version, checkpoint, step)
    models = [
        {
            "model": v1_model,
            "config": v1_config,
            "tokenizer": v1_tokenizer,
            "eos_token_id": None,  # v1.0 has no EOS
            "skip_special": False,  # v1.0 inference.py uses skip_special_tokens=False
            "version": "v1.0.0",
            "checkpoint": str(V1_RELEASE_DIR.relative_to(REPO_ROOT)),
            "step": 11000,  # from README: "Training steps: 11,000"
        },
        {
            "model": v11_model,
            "config": v11_config,
            "tokenizer": v11_tokenizer,
            "eos_token_id": v11_eos_id,
            "skip_special": True,  # v1.1 generate.py uses skip_special_tokens=True
            "version": "v1.1",
            "checkpoint": str(V11_CHECKPOINT_DIR.relative_to(REPO_ROOT)),
            "step": v11_step,
        },
    ]

    # Build all 72 generation tasks
    tasks = []
    for m in models:
        for prompt in PROMPTS:
            # Greedy
            tasks.append((m, prompt, None, None))  # temperature=None means greedy
            # Temperature 1.0 with each seed
            for seed in SEEDS:
                tasks.append((m, prompt, 1.0, seed))

    print(f"\nTotal generations: {len(tasks)}", flush=True)
    print(f"Output: {args.output}", flush=True)
    print("=" * 60, flush=True)

    records: list[GenerationRecord] = []
    start_time = time.time()

    for i, (m, prompt, temperature, seed) in enumerate(tasks):
        decoding = "greedy" if temperature is None else f"temp={temperature}"
        label = f"[{i+1}/{len(tasks)}] {m['version']} {decoding}" + (
            f" seed={seed}" if seed is not None else ""
        )

        t0 = time.time()
        text, n_tokens, finish_reason = generate(
            model=m["model"],
            tokenizer=m["tokenizer"],
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=temperature,
            seed=seed,
            eos_token_id=m["eos_token_id"],
            context_length=m["config"].context_length,
            skip_special=m["skip_special"],
        )
        elapsed = time.time() - t0

        record = GenerationRecord(
            model_version=m["version"],
            checkpoint=m["checkpoint"],
            training_step=m["step"],
            prompt=prompt,
            decoding=decoding,
            seed=seed,
            temperature=temperature,
            max_tokens=args.max_tokens,
            generated_tokens=n_tokens,
            finish_reason=finish_reason,
            text=text,
        )
        records.append(record)

        print(
            f"{label} → {n_tokens} tokens, {finish_reason}, {elapsed:.1f}s",
            flush=True,
        )

    total_time = time.time() - start_time

    # Write JSONL (exclusive create)
    with args.output.open("x", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")

    print("=" * 60, flush=True)
    print(f"Done. {len(records)} generations in {total_time:.1f}s", flush=True)
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
