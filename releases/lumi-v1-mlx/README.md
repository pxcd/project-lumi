# Lumi v1 — MLX Release

Lumi is a decoder-only Transformer language model built and trained
from scratch using Apple's MLX framework.

## Model

- Architecture: Decoder-only Transformer
- Parameters: 29,398,016
- Layers: 8
- Hidden dimension: 512
- Attention heads: 8
- Feed-forward dimension: 2048
- Context length: 512 tokens
- Vocabulary: 8,192 tokens
- Positional embeddings: RoPE
- Attention: Causal self-attention
- Output embeddings: Tied
- Training steps: 11,000
- Training tokens processed: 22,528,000
- Training dataset: TinyStories
- Hardware: Apple Silicon M5 Pro
- Framework: MLX 0.32.4

## Requirements

- macOS on Apple Silicon
- Python 3.12
- Dependencies listed in requirements.txt

## Installation

Create and activate a Python 3.12 virtual environment.

Install dependencies:

    pip install -r requirements.txt

## Inference

Run:

    python inference.py

Custom prompt:

    python inference.py --prompt "Once upon a time, there was a dragon."

Generation options:

    python inference.py --max-tokens 200 --temperature 1.0 --seed 42

## Files

- model.safetensors — Trained model weights
- model.py — MLX Transformer implementation
- config.json — Model architecture configuration
- tokenizer.json — Trained BPE tokenizer
- generation_config.json — Default generation settings
- inference.py — Standalone generation script
- requirements.txt — Python dependencies

## Limitations

Lumi v1 is an experimental research model.

It was trained on children's stories and is not instruction-tuned.

Known limitations include:

- Poor long-range narrative coherence
- Character and object inconsistencies
- Repetition
- Malformed words
- No explicit end-of-story token
- No KV cache
- Maximum context length of 512 tokens

This package uses native MLX inference. It is not automatically
compatible with mlx-lm, Ollama, or llama.cpp.

## Historical milestone

First successful pretraining and generation: October 19, 2026.

The original first-generation output is preserved separately
in the Project Lumi repository.
