# Lumi Evidence and Verification

Checked for public release on **October 27, 2026**. Historical generations and the original release are from **October 19, 2026**.

## What is available to inspect

| Claim | Evidence | What it establishes |
| --- | --- | --- |
| Lumi implements a decoder-only Transformer | [Model source](../lumi/model.py) and [standalone model](../releases/lumi-v1-mlx/model.py) | The actual architecture and forward pass |
| The v1.0 model has 29,398,016 parameters | Released `model.safetensors`; [artifact manifest](evidence/artifact-manifest.json) | Tensor-shape count from the saved weight file, excluding safetensors metadata |
| The original model produced story continuations | [Preserved v1.0 first-generation record](evidence/first-words-v1.txt) | Saved prompt, settings, timestamp, and complete output, including its flaws |
| v1.1 added EOS-aware generation | [Generation source](../lumi/generate.py), [focused tests](../tests/v1_1_test.py), and [preserved v1.1 output](evidence/first-words-v1.1.txt) | Implemented stopping behavior and a saved example |
| A version comparison was recorded | [Raw generations](../experiments/v1-vs-v1.1/results/generations.jsonl) and [evaluation source](../experiments/v1-vs-v1.1/evaluate.py) | 72 unique saved model/prompt/decoding/seed combinations |

Hashes identify the exact artifact bytes; they are not independent proof of training provenance or a substitute for a reproducible training run.

## Checked raw comparison results

The saved JSONL contains **72 records**, covering two models, six fixed prompts, and six configurations per prompt: greedy generation plus temperature 1.0 sampling with seeds **42, 43, 44, 45, 46**. Output is capped at 200 generated tokens.

| Model | Saved records | EOS stop | Token-limit stop |
| --- | --- | --- | --- |
| v1.0.0 | 36 | 0 | 36 |
| v1.1 | 36 | 30 | 6 |

These numbers describe termination behavior. An EOS stop can still follow a poor or nonsensical story. They do not measure general intelligence, factual accuracy, or instruction-following.

The [historical comparison report](../experiments/v1-vs-v1.1/report.md) is preserved unchanged. It contains an inconsistent seed list and conflicting repetition summaries, so use the raw records and the counts above for quantitative claims. Its story-quality judgments are subjective. Its statement that there were no existing tests does not describe this checkout, which includes smoke and v1.1 tests.

To check the raw counts without loading a model, run from the repository root:

```sh
python3 - <<'PY'
import json
from collections import Counter
from pathlib import Path

path = Path('experiments/v1-vs-v1.1/results/generations.jsonl')
records = [json.loads(line) for line in path.read_text().splitlines()]
keys = {(r['model_version'], r['prompt'], r['decoding'], r['seed']) for r in records}
assert len(records) == len(keys) == 72
print(Counter((r['model_version'], r['finish_reason']) for r in records))
print(sorted({r['seed'] for r in records if r['seed'] is not None}))
PY
```

## Public-release checks

- Counted **29,398,016** parameters directly from the v1.0 safetensors header; all stored tensors are float32.
- Rechecked all **72 unique saved records**, their sampling seeds, and the finish-reason totals above.
- Ran the existing **5 smoke checks** on host Metal: initialization, forward pass, causal attention, training step, and checkpoint restoration. All passed.
- Ran the existing **6 v1.1 tests** for story boundaries, EOS vocabulary and stopping, fixed validation windows, deterministic initialization, and checkpoint state/integrity. All passed.
- Extracted the new public archive into a temporary directory, verified all original release files were byte-identical, and ran a 200-token native inference check. Its seed-42 output matched the preserved v1.0 example exactly. See the [fresh release smoke record](evidence/release-smoke-2026-10-27.json). This used the existing Python environment; a clean dependency installation was not tested.

The historical generation files are copied byte-for-byte from the local project records. Training was not rerun during public-release preparation. The original v1.0 weight file and historical release archive are preserved; the public archive adds license and model documentation around copies of the original files.

## Scope and reproduction limits

The downloadable release is **v1.0.0**, including its matching weights and tokenizer. v1.1's checkpoint, raw data sample, and prepared corpus are excluded from the public release. The saved v1.1 generations can be inspected, but reproducing that run requires those original artifacts. The upstream TinyStories dataset is linked in [third-party notices](../THIRD_PARTY_NOTICES.md).

The [LM Studio experiment](../experiments/lmstudio-compat/README.md) records historical conversion parity and a machine-specific runtime result. It was not rerun during public-release preparation, and its throughput figure should not be presented as a general benchmark.
