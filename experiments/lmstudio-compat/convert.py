#!/usr/bin/env python3
"""Convert Lumi v1 MLX weights to LM Studio's StarCoder2 MLX layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "releases" / "lumi-v1-mlx"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "converted" / "lumi-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_shape(weights: dict[str, mx.array], name: str, shape: tuple[int, ...]) -> None:
    if name not in weights:
        raise ValueError(f"Missing source tensor: {name}")
    actual = tuple(weights[name].shape)
    if actual != shape:
        raise ValueError(f"Unexpected shape for {name}: expected {shape}, got {actual}")


def convert(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must not be the source directory or contain it")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    source_config = json.loads((source / "config.json").read_text())
    expected = {
        "vocab_size": 8192,
        "context_length": 512,
        "n_layers": 8,
        "d_model": 512,
        "n_heads": 8,
        "d_ff": 2048,
    }
    actual = {key: source_config.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"This converter targets Lumi v1; config mismatch: {actual}")

    source_weights_path = source / "model.safetensors"
    source_weights = mx.load(str(source_weights_path))
    converted: dict[str, mx.array] = {}

    d_model = expected["d_model"]
    d_ff = expected["d_ff"]
    vocab_size = expected["vocab_size"]
    require_shape(source_weights, "embedding.weight", (vocab_size, d_model))
    require_shape(source_weights, "final_norm.weight", (d_model,))
    require_shape(source_weights, "final_norm.bias", (d_model,))

    converted["model.embed_tokens.weight"] = source_weights["embedding.weight"]
    converted["model.norm.weight"] = source_weights["final_norm.weight"]
    converted["model.norm.bias"] = source_weights["final_norm.bias"]

    for layer in range(expected["n_layers"]):
        source_prefix = f"blocks.{layer}"
        target_prefix = f"model.layers.{layer}"

        mappings = {
            f"{source_prefix}.attn_norm.weight": f"{target_prefix}.input_layernorm.weight",
            f"{source_prefix}.attn_norm.bias": f"{target_prefix}.input_layernorm.bias",
            f"{source_prefix}.ffn_norm.weight": f"{target_prefix}.post_attention_layernorm.weight",
            f"{source_prefix}.ffn_norm.bias": f"{target_prefix}.post_attention_layernorm.bias",
            f"{source_prefix}.attention.q_proj.weight": f"{target_prefix}.self_attn.q_proj.weight",
            f"{source_prefix}.attention.k_proj.weight": f"{target_prefix}.self_attn.k_proj.weight",
            f"{source_prefix}.attention.v_proj.weight": f"{target_prefix}.self_attn.v_proj.weight",
            f"{source_prefix}.attention.o_proj.weight": f"{target_prefix}.self_attn.o_proj.weight",
            f"{source_prefix}.ffn.up.weight": f"{target_prefix}.mlp.c_fc.weight",
            f"{source_prefix}.ffn.up.bias": f"{target_prefix}.mlp.c_fc.bias",
            f"{source_prefix}.ffn.down.weight": f"{target_prefix}.mlp.c_proj.weight",
            f"{source_prefix}.ffn.down.bias": f"{target_prefix}.mlp.c_proj.bias",
        }

        expected_shapes = {
            f"{source_prefix}.attn_norm.weight": (d_model,),
            f"{source_prefix}.attn_norm.bias": (d_model,),
            f"{source_prefix}.ffn_norm.weight": (d_model,),
            f"{source_prefix}.ffn_norm.bias": (d_model,),
            f"{source_prefix}.attention.q_proj.weight": (d_model, d_model),
            f"{source_prefix}.attention.k_proj.weight": (d_model, d_model),
            f"{source_prefix}.attention.v_proj.weight": (d_model, d_model),
            f"{source_prefix}.attention.o_proj.weight": (d_model, d_model),
            f"{source_prefix}.ffn.up.weight": (d_ff, d_model),
            f"{source_prefix}.ffn.up.bias": (d_ff,),
            f"{source_prefix}.ffn.down.weight": (d_model, d_ff),
            f"{source_prefix}.ffn.down.bias": (d_model,),
        }
        for source_name, target_name in mappings.items():
            require_shape(source_weights, source_name, expected_shapes[source_name])
            converted[target_name] = source_weights[source_name]

        # StarCoder2 allocates attention biases. Lumi has none, so zero is exact.
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            converted[f"{target_prefix}.self_attn.{projection}.bias"] = mx.zeros(
                (d_model,), dtype=source_weights["embedding.weight"].dtype
            )

    expected_source_keys = 3 + expected["n_layers"] * 12
    if len(source_weights) != expected_source_keys:
        unknown = sorted(set(source_weights) - {
            "embedding.weight",
            "final_norm.weight",
            "final_norm.bias",
            *[
                name
                for layer in range(expected["n_layers"])
                for name in (
                    f"blocks.{layer}.attn_norm.weight",
                    f"blocks.{layer}.attn_norm.bias",
                    f"blocks.{layer}.ffn_norm.weight",
                    f"blocks.{layer}.ffn_norm.bias",
                    f"blocks.{layer}.attention.q_proj.weight",
                    f"blocks.{layer}.attention.k_proj.weight",
                    f"blocks.{layer}.attention.v_proj.weight",
                    f"blocks.{layer}.attention.o_proj.weight",
                    f"blocks.{layer}.ffn.up.weight",
                    f"blocks.{layer}.ffn.up.bias",
                    f"blocks.{layer}.ffn.down.weight",
                    f"blocks.{layer}.ffn.down.bias",
                )
            ],
        })
        raise ValueError(f"Unexpected source tensor set; extra tensors: {unknown}")

    output.mkdir(parents=True)
    mx.save_safetensors(str(output / "model.safetensors"), converted)

    target_config = {
        "architectures": ["Starcoder2ForCausalLM"],
        "model_type": "starcoder2",
        "hidden_size": d_model,
        "num_hidden_layers": expected["n_layers"],
        "intermediate_size": d_ff,
        "num_attention_heads": expected["n_heads"],
        "num_key_value_heads": expected["n_heads"],
        "norm_epsilon": 1e-5,
        "vocab_size": vocab_size,
        "max_position_embeddings": expected["context_length"],
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
    }
    (output / "config.json").write_text(json.dumps(target_config, indent=2) + "\n")

    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    added = tokenizer.add_special_tokens(["<|endoftext|>"])
    if added != 1 or tokenizer.token_to_id("<|endoftext|>") != vocab_size:
        raise ValueError("Could not allocate the required tokenizer-only EOS sentinel")
    tokenizer.save(str(output / "tokenizer.json"))
    tokenizer_config = {
        "add_prefix_space": False,
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|endoftext|>",
        "model_max_length": expected["context_length"],
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    (output / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2) + "\n"
    )
    generation_config = json.loads((source / "generation_config.json").read_text())
    generation_config["eos_token_id"] = vocab_size
    (output / "generation_config.json").write_text(
        json.dumps(generation_config, indent=2) + "\n"
    )

    manifest = {
        "source": str(source.relative_to(REPO_ROOT)),
        "source_model_sha256": sha256(source_weights_path),
        "target_architecture": "starcoder2",
        "conversion": "Tensor rename plus zero attention biases; no trained values changed.",
        "eos_sentinel_id": vocab_size,
        "eos_sentinel_note": "Tokenizer-only and outside the model output vocabulary.",
    }
    (output / "conversion-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        convert(args.source, args.output)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
