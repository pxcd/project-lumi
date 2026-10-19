"""Standalone MLX inference for Lumi v1."""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

from model import LumiConfig, LumiModel


def main():
    parser = argparse.ArgumentParser(description="Lumi v1 MLX inference")
    parser.add_argument(
        "--prompt",
        default="Once upon a time, there was a little girl named Lily.",
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    directory = Path(__file__).resolve().parent

    config_data = json.loads(
        (directory / "config.json").read_text()
    )
    generation_config = json.loads(
        (directory / "generation_config.json").read_text()
    )

    config = LumiConfig(
        **{
            key: value
            for key, value in config_data.items()
            if key in LumiConfig.__dataclass_fields__
        }
    )

    model = LumiModel(config)
    model.load_weights(str(directory / "model.safetensors"))
    model.eval()
    mx.eval(model.parameters())

    tokenizer = Tokenizer.from_file(
        str(directory / "tokenizer.json")
    )

    max_tokens = (
        args.max_tokens
        if args.max_tokens is not None
        else generation_config["max_new_tokens"]
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else generation_config["temperature"]
    )
    seed = (
        args.seed
        if args.seed is not None
        else generation_config["seed"]
    )

    if max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if temperature <= 0:
        parser.error("--temperature must be positive")

    mx.random.seed(seed)

    tokens = tokenizer.encode(args.prompt).ids
    if not tokens:
        parser.error("Prompt produced no tokens")

    generated = tokens.copy()

    for _ in range(max_tokens):
        context = generated[-config.context_length:]
        inputs = mx.array([context], dtype=mx.int32)

        logits = model(inputs)[0, -1, :] / temperature
        next_token = mx.random.categorical(logits)
        mx.eval(next_token)

        generated.append(int(next_token.item()))

    continuation = tokenizer.decode(
        generated[len(tokens):],
        skip_special_tokens=False,
    )

    print(args.prompt + continuation)


if __name__ == "__main__":
    main()
