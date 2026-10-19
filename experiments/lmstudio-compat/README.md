# Lumi v1 in LM Studio

This experiment converts the immutable Lumi v1 MLX release into the tensor and
metadata layout used by LM Studio's bundled `mlx-lm` StarCoder2 implementation.
It does not alter the release checkpoint.

## Why StarCoder2 is compatible

Both implementations use sequential pre-LayerNorm decoder blocks, separate
Q/K/V projections, causal attention, full-head RoPE with base 10,000, exact
GELU, a two-linear-layer 4x MLP with biases, a final LayerNorm, and tied input
and output embeddings. StarCoder2's attention linears include biases; the
converter supplies zeros, which is mathematically equivalent to Lumi's
bias-free attention linears.

LM Studio requires at least one EOS token. Lumi has none, so the converted
tokenizer adds `<|endoftext|>` as ID 8192. That ID is outside Lumi's 8192-entry
model vocabulary and therefore cannot be generated; generation remains
token-count-limited.

## Convert and verify

From the repository root:

```sh
.venv/bin/python experiments/lmstudio-compat/convert.py
```

The generated model is written to `converted/lumi-v1`, which is ignored by Git.
The verifier must run with an environment containing `mlx-lm`:

```sh
python experiments/lmstudio-compat/verify.py
```

It performs a strict weight load, compares prompt logits, and checks 100 greedy
tokens against the native Lumi implementation.

## LM Studio

LM Studio derives the publisher and model identifiers from the two directory
components below its model root. Install the generated directory at
`~/.lmstudio/models/lumi.ai/lumi-v1` (a symlink is sufficient) to display
publisher `lumi.ai` and model `lumi-v1`. These names are packaging identity;
the required internal architecture remains `starcoder2`.

Select the MLX runtime, load with a 512-token context, disable prompt/chat
templating, and use the raw prompt:

> Once upon a time, there was a little girl named Lily.

Lumi has no EOS token, so generation must be limited by token count. For the
reference sampling test use temperature 1.0, seed 42, and 100 output tokens.
Sampling output need not be byte-identical across runtimes; the verifier's
logit and greedy-token comparisons are the conversion correctness test.

LM Studio 0.4.26+4 currently crashes when its `lms` CLI tries to wake the app
on this machine. That launcher issue is independent of model compatibility;
use the already-running GUI for the final import/load/generation check.

## Verified result

Verified on October 19, 2026 with LM Studio 0.4.26+4 and its bundled MLX
runtime 1.13.1 (`mlx-lm` 0.31.4.dev132):

- LM Studio derived publisher `lumi.ai` and model `lumi-v1` from the catalog
  path, discovered it as MLX/fp32 with architecture `starcoder2`, and loaded it
  with a 512-token context.
- Strict `mlx-lm` weight loading succeeded.
- Native and converted prompt logits had a maximum absolute error of `0`.
- Native and converted greedy generation matched for 100/100 tokens.
- LM Studio generated text from the reference prompt at 331.10 tokens/second
  and stopped at the configured response-length limit (99 tokens shown by the
  UI).

The LM Studio seed-42 sample did not match native MLX's seed-42 sample. This is
a sampler/RNG difference, not a conversion discrepancy; exact logit and greedy
parity isolate the model path from sampling behavior.
