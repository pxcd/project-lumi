# Project Lumi

A small experimental language model built and pretrained from scratch on Apple Silicon using [MLX](https://github.com/ml-explore/mlx).

Lumi learns to continue short English stories using a subset of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). This repository contains the Transformer implementation, tokenizer preparation, training and generation code, and saved experiments. It is an educational research project with real limitations: stories can repeat, contradict themselves, or turn into nonsense.

**[Download Lumi v1.0.0](https://github.com/pxcd/project-lumi/releases/tag/v1.0.0)** · **[Model card](MODEL_CARD.md)** · **[Evidence and sample outputs](docs/EVIDENCE.md)** · **[Wishlist](WISHLIST.md)**

## Model

| Property | Value |
| --- | --- |
| Parameters | 29,398,016 |
| Architecture | Decoder-only Transformer, initialized from scratch |
| Layers / hidden width / attention heads | 8 / 512 / 8 |
| Feed-forward width | 2,048 |
| Vocabulary / context window | 8,192 / 512 tokens |
| Position encoding | RoPE |
| Input and output embeddings | Tied |
| Framework / training hardware | MLX / Apple Silicon M5 Pro |
| Training steps | 11,000 |

The parameter count was checked against the released weight file. See the [evidence page](docs/EVIDENCE.md) for the distinction between saved training records, fresh checks, and model-quality claims.

## Run the released model

Requires **macOS on Apple Silicon and Python 3.12**. The standalone release runs locally and does not need an inference service or API key.

Download the two assets from the [v1.0.0 release](https://github.com/pxcd/project-lumi/releases/tag/v1.0.0) into an empty directory, or use:

```sh
mkdir lumi-download
cd lumi-download
curl -fL -O https://github.com/pxcd/project-lumi/releases/download/v1.0.0/lumi-v1.0.0-mlx.tar.gz
curl -fL -O https://github.com/pxcd/project-lumi/releases/download/v1.0.0/lumi-v1.0.0-mlx.tar.gz.sha256
shasum -a 256 -c lumi-v1.0.0-mlx.tar.gz.sha256
tar -xzf lumi-v1.0.0-mlx.tar.gz
cd lumi-v1-mlx
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python inference.py --prompt "Once upon a time, there was a little girl named Lily." --max-tokens 200 --temperature 1.0 --seed 42
```

The archive includes the weights, matching tokenizer, configuration, inference code, model card, and license. GitHub's automatically generated source archives and a normal Git clone exclude the weights.

v1.0 has no end-of-story token and generates until the requested token limit. Sampling can vary across runtime versions; the seed identifies the configuration, not a promise of identical text on every system.

For LM Studio, see the [v1.0 compatibility experiment](experiments/lmstudio-compat/README.md). The downloadable package uses native MLX inference; conversion is a separate step.

## Versions

| Version | Status | Difference |
| --- | --- | --- |
| v1.0.0 | Downloadable standalone MLX release | Original tokenizer; no end-of-story token |
| v1.1 | Development code and saved evaluation outputs | EOS-aware tokenizer, story-boundary preparation, reproducible training and resumable checkpoints |

**Keep each version's tokenizer and weights together.** v1.1 changes token IDs; its tokenizer cannot be substituted into the v1.0 release. The v1.1 checkpoint and prepared training corpus are not included in this public release.

## Explore the source

```sh
git clone https://github.com/pxcd/project-lumi.git
cd project-lumi
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=. python tests/smoke_test.py
python -m unittest discover -s tests -p 'v1_1_test.py' -v
```

| Path | Purpose |
| --- | --- |
| `lumi/model.py` | Transformer architecture |
| `lumi/dataset.py` | Tokenizer and dataset preparation |
| `lumi/train.py`, `lumi/pretrain.py` | Training, checkpointing, and validation |
| `lumi/generate.py` | Generation with EOS support |
| `releases/lumi-v1-mlx/` | v1.0 standalone inference source and tokenizer |
| `experiments/` | Conversion code, comparison script, and saved generations |
| `tests/` | Smoke, reproducibility, and overfit checks |

Training requires separately supplied data. The v1.1 preparation command expects both the original sampled TinyStories text and its historical reference corpus groups; a fresh clone does not contain those files. Training a new model is possible from the source, but exact reproduction of the existing run requires those original inputs. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Limitations

- Story continuation only; Lumi has not been instruction-tuned as a general chat assistant.
- A 512-token context window and limited long-range coherence.
- Repetition, character drift, fabricated facts, and malformed language occur.
- Evaluation uses six fixed prompts and does not establish broad language or reasoning capability.
- Current implementation targets MLX on Apple Silicon; other platforms are not validated.

## License and credits

Lumi's code and released model weights are available under the [MIT License](LICENSE). The training dataset and dependencies retain their own licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).

TinyStories was created by Ronen Eldan and Yuanzhi Li. MLX is developed by Apple and contributors. Lumi's first saved generation is dated October 19, 2026.
