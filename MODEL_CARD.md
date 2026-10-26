---
license: mit
language:
  - en
datasets:
  - roneneldan/TinyStories
tags:
  - mlx
  - text-generation
  - trained-from-scratch
---

# Lumi v1.0.0 Model Card

## Model details

Lumi is an experimental English story-continuation model created by [pxcd](https://github.com/pxcd). The decoder-only Transformer was initialized and trained from scratch with Apple's MLX framework on an Apple Silicon M5 Pro.

- **Parameters:** 29,398,016, counted from the released safetensors header.
- **Architecture:** 8 layers, hidden width 512, 8 attention heads, feed-forward width 2,048, RoPE, tied input/output embeddings.
- **Vocabulary:** 8,192 tokens, using the release's byte-level BPE tokenizer.
- **Context window:** 512 tokens.
- **Weights:** native MLX safetensors; float32.
- **License:** MIT for Lumi's code and released trained weights, under the accompanying `LICENSE` file. Dataset and dependency licenses remain separate.
- **Source:** https://github.com/pxcd/project-lumi
- **Download:** https://github.com/pxcd/project-lumi/releases/tag/v1.0.0

## Intended use

Local experimentation, education, inspecting a small language model's implementation, and generating short story continuations. The release is not instruction-tuned and has no chat template.

## Training data and procedure

The project uses a sampled subset of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), the synthetic short-story dataset by Ronen Eldan and Yuanzhi Li. The upstream dataset card identifies its license as CDLA-Sharing-1.0. The training corpus is not included in the release.

The original v1.0 release record reports 11,000 training steps and 22,528,000 processed training tokens. These are historical run records; the public packaging check does not rerun training. Lumi was not initialized from pretrained Qwen, GPT, or other model weights.

The later v1.1 development run adds an end-of-story token and uses different token IDs. Its tokenizer and checkpoint are not interchangeable with v1.0 and are not part of this downloadable model.

## Evaluation and evidence

The repository contains 72 saved generations: 36 each for v1.0 and v1.1, covering six prompts and six decoding configurations. v1.0 reaches the 200-token limit in all 36 records. v1.1 ends on EOS in 30 records and reaches the limit in six. Ending on EOS measures termination behavior, not story quality.

See [the repository evidence page](https://github.com/pxcd/project-lumi/blob/main/docs/EVIDENCE.md) for raw outputs, verification commands, provenance, and limitations. The six-prompt comparison is small, is not a general capability benchmark, and does not establish reasoning or factual reliability.

## Known limitations

- Repetition, story restarts, character inconsistencies, and nonsensical text.
- A short context window and weak long-range narrative coherence.
- No EOS token in v1.0; generation stops at the configured token limit.
- No KV cache in the native implementation.
- No validated instruction-following, factuality, or safety evaluation.
- Only macOS on Apple Silicon with the stated Python/MLX dependencies is supported by this release.

## Run locally

After extracting the release archive, create a Python 3.12 virtual environment, install its `requirements.txt`, and run:

```sh
python inference.py --prompt "Once upon a time, there was a little girl named Lily." --max-tokens 200 --temperature 1.0 --seed 42
```

Keep `model.safetensors`, `tokenizer.json`, and the configuration files from the same archive together. The native package needs conversion before use with LM Studio's MLX runtime; the conversion experiment is documented in the source repository.
