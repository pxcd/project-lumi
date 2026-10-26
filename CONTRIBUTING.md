# Contributing

Use an [issue](https://github.com/pxcd/project-lumi/issues/new/choose) for a reproducible bug or a concrete feature proposal. Keep changes focused, include the affected Lumi version, and preserve the matching tokenizer/weight pair. Future ideas are tracked in [WISHLIST.md](WISHLIST.md); an entry does not mean implementation has been approved.

## Development setup

Use macOS on Apple Silicon and a project-local Python 3.12 environment:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=. python tests/smoke_test.py
python -m unittest discover -s tests -p 'v1_1_test.py' -v
```

These checks use small synthetic fixtures and do not need the training corpus. For a change, run the narrowest relevant checks and report what was actually tested. Do not commit environments, secrets, training corpora, checkpoints, or model binaries. Do not overwrite saved experiment results when running a new comparison; use a new output path.

## Training and reproduction

`python -m lumi.dataset --help`, `python -m lumi.pretrain --help`, and `python -m lumi.generate --help` describe the source entry points. Data preparation expects the sampled raw TinyStories file plus its historical reference corpus groups, which are excluded from Git. The original sample is not currently distributed, so a fresh clone alone cannot reproduce the original v1.1 training run exactly.

The training code can be studied and adapted to separately obtained data. Identify the source, preprocessing, seed, runtime versions, and artifact hashes for any new run. Keep v1.0 and v1.1 outputs separate: their token IDs have different meanings. Do not compare token-level losses across the two tokenizers as if the tokenization were identical.

## License

Contributions to Lumi's code and documentation are submitted under the repository's MIT license. Third-party materials retain their own terms and must be attributed.
