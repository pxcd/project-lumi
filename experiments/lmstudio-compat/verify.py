#!/usr/bin/env python3
"""Compare native Lumi and converted StarCoder2 MLX inference."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "releases" / "lumi-v1-mlx"
DEFAULT_CONVERTED = Path(__file__).resolve().parent / "converted" / "lumi-v1"
DEFAULT_PROMPT = "Once upon a time, there was a little girl named Lily."


def load_native(source: Path):
    spec = importlib.util.spec_from_file_location("lumi_release_model", source / "model.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import the Lumi release model")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config_data = json.loads((source / "config.json").read_text())
    config = module.LumiConfig(
        **{
            key: value
            for key, value in config_data.items()
            if key in module.LumiConfig.__dataclass_fields__
        }
    )
    model = module.LumiModel(config)
    model.load_weights(str(source / "model.safetensors"))
    model.eval()
    mx.eval(model.parameters())
    return model, config


def greedy(model, prompt_tokens: list[int], max_tokens: int, context_length: int) -> list[int]:
    generated = prompt_tokens.copy()
    for _ in range(max_tokens):
        inputs = mx.array([generated[-context_length:]], dtype=mx.int32)
        next_token = mx.argmax(model(inputs)[0, -1, :])
        mx.eval(next_token)
        generated.append(int(next_token.item()))
    return generated[len(prompt_tokens):]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--converted", type=Path, default=DEFAULT_CONVERTED)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    source = args.source.resolve()
    converted = args.converted.resolve()
    native, native_config = load_native(source)
    compatible, _ = load_model(converted, strict=True)

    source_tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    converted_tokenizer = Tokenizer.from_file(str(converted / "tokenizer.json"))
    source_tokens = source_tokenizer.encode(args.prompt).ids
    converted_tokens = converted_tokenizer.encode(args.prompt).ids
    if source_tokens != converted_tokens:
        raise AssertionError("Tokenizer IDs differ")

    inputs = mx.array([source_tokens], dtype=mx.int32)
    native_logits = native(inputs)
    compatible_logits = compatible(inputs)
    mx.eval(native_logits, compatible_logits)
    max_abs_logit_error = float(mx.max(mx.abs(native_logits - compatible_logits)).item())

    native_generated = greedy(native, source_tokens, args.max_tokens, native_config.context_length)
    compatible_generated = greedy(
        compatible, source_tokens, args.max_tokens, native_config.context_length
    )
    if native_generated != compatible_generated:
        for index, (native_token, compatible_token) in enumerate(
            zip(native_generated, compatible_generated)
        ):
            if native_token != compatible_token:
                raise AssertionError(
                    f"Greedy output diverged at token {index}: "
                    f"native={native_token}, converted={compatible_token}"
                )
        raise AssertionError("Greedy output lengths differ")

    continuation = converted_tokenizer.decode(
        compatible_generated, skip_special_tokens=False
    )
    print(f"prompt_tokens={len(source_tokens)}")
    print(f"max_abs_logit_error={max_abs_logit_error:.9g}")
    print(f"greedy_tokens_match={len(native_generated)}/{len(native_generated)}")
    print(args.prompt + continuation)


if __name__ == "__main__":
    main()
