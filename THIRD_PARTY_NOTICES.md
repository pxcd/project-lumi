# Third-party Notices

## TinyStories training data

Lumi was trained on a sampled subset of **TinyStories**, created by **Ronen Eldan and Yuanzhi Li**.

- Dataset and attribution: https://huggingface.co/datasets/roneneldan/TinyStories
- Paper: *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* https://arxiv.org/abs/2305.07759
- Dataset license: **Community Data License Agreement – Sharing, Version 1.0 (CDLA-Sharing-1.0)**, as identified by the upstream dataset card.
- License text: https://cdla.dev/sharing-1-0/

The raw and prepared training corpora are not distributed in this repository or its model archive. The MIT license for Lumi's code and trained weights does not relicense TinyStories. Under section 3.5 of CDLA-Sharing-1.0, computational Results are not subject to the agreement's publication restrictions; the definition of Results excludes substantial portions of the original data. Users redistributing any training data must follow that data's own license.

## Runtime dependencies

Lumi uses MLX and its Metal runtime, NumPy, and Hugging Face Tokenizers. These packages are installed separately and retain their own licenses and notices:

- MLX: https://github.com/ml-explore/mlx
- NumPy: https://github.com/numpy/numpy
- Tokenizers: https://github.com/huggingface/tokenizers

The optional LM Studio compatibility experiment uses its installed MLX runtime and `mlx-lm`: https://github.com/ml-explore/mlx-lm. LM Studio is separate software and is not bundled with Lumi.
