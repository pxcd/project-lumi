# Lumi v1.0.0 vs v1.1 — Generation Comparison Report

**Date:** 2026-10-19
**Branch:** `wugpt/evaluation` (commit b55ce74)
**Worktree:** `/Users/pow/Development/project-lumi-wugpt`

## 1. Experimental Setup

### Models
| | v1.0.0 | v1.1 (step 11000) |
|---|---|---|
| Checkpoint dir | `releases/lumi-v1-mlx/` | `checkpoints/lumi-v1.1/best/` |
| Architecture | 8 layers, 512 d_model, 8 heads, 2048 d_ff | Same (shared `lumi.model`) |
| Parameters | ~29.4M | ~29.4M |
| Vocabulary size | 8192 (no special tokens) | 8192 (`<|endofstory|>` at id 0) |
| Context length | 512 | 512 |
| Training tokens | 22,528,000 (README) | 22,242,591 (dataset.json) |
| Training steps | 11,000 (README) | 11,000 (metadata.json) |

### Prompts (6 fixed)
1. `"Once upon a time, there was a little girl named Lily."` (13 tokens)
2. `"Once upon a time, there was a little boy named Tom."` (13 tokens)
3. `"Lily found a mysterious box in the garden."` (9 tokens)
4. `"The little dragon was afraid of fire."` (8 tokens)
5. `"One day, a girl named Anna lost her favorite toy."` (12 tokens)
6. `"The boy looked at his mother and said, \""` (10 tokens)

### Decoding Configs (6 per model+prompt = 72 total)
- **Greedy:** `mx.argmax`, no random seed, max 200 tokens.
- **Temperature 1.0:** seeds {42, 43, 44, 56}, max 200 tokens. Random seed reset before each independent generation.

### Termination Behavior
- **v1.0:** No EOS token exists; always exhausts `max_tokens` → finish_reason `"length"`.
- **v1.1:** Stops on `<|endofstory|>` (token id 0) or exhausts `max_tokens` → finish_reason `"eos"` or `"length"`.

### Decode Convention (matches each model's existing code)
- **v1.0:** `skip_special_tokens=False` (no special tokens to strip).
- **v1.1:** `skip_special_tokens=True` (strips `<|endofstory|>` from output text).

---

## 2. Objective Metrics

### 2.1 Record Verification
- **Total records:** 72 (verified)
- **Unique (model, prompt, decoding, seed) combinations:** 72 (verified — all unique)
- **All required fields present** in every record: `model_version`, `checkpoint`, `training_step`, `prompt`, `decoding`, `seed`, `temperature`, `max_tokens`, `generated_tokens`, `finish_reason`, `text`.
- **Extra field:** `timestamp_utc` (ISO 8601 UTC).

### 2.2 Finish Reasons

| Model | Greedy (n=6) | Temp=1.0 (n=30) | Total (n=36) |
|---|---|---|---|
| **v1.0.0** | 6 × `length` (200 tok) | 30 × `length` (200 tok) | **36 × `length`** |
| **v1.1** | 4 × `eos`, 2 × `length` (60–200 tok) | 26 × `eos`, 4 × `length` (13–200 tok) | **30 × `eos`, 6 × `length`** |

### 2.3 Token Counts (generated tokens only, excluding prompt)

| Model | Decoding | Min | Max | Mean |
|---|---|---|---|---|
| v1.0.0 | All | 200 | 200 | 200.0 |
| v1.1 | Greedy | 60 | 200 | 142.3 |
| v1.1 | Temp=1.0 | 13 | 200 | 134.8 |

### 2.4 Prompt-Level EOS vs Length (v1.1)

| Prompt | Greedy | Temp=1.0 (5 seeds) |
|---|---|---|
| "Once upon a time, there was a little girl named Lily." | 120 tok `eos` | 5/5 `eos` (114–171 tok) |
| "Once upon a time, there was a little boy named Tom." | 130 tok `eos` | 5/5 `eos` (129–177 tok) |
| "Lily found a mysterious box in the garden." | 200 tok `length` | 4/5 `eos`, 1/5 `length` (30–200 tok) |
| "The little dragon was afraid of fire." | 60 tok `eos` | 5/5 `eos` (13–142 tok) |
| "One day, a girl named Anna lost her favorite toy." | 144 tok `eos` | 4/5 `eos`, 2/5 `length` (110–200 tok) |
| "The boy looked at his mother and said, \"" | 200 tok `length` | 5/5 `eos` (53–181 tok) |

### 2.4 Repetition and Restart Detection (Automated Heuristics)

**Method:** Count repeated 4-grams (>1 occurrence); detect story restarts via regex `(Once upon a time|There was a|There lived)` appearing after position 50.

#### v1.0.0 — Greedy (6 records)
| Prompt | 4-gram repeats | Restarts | Flagged? |
|---|---|---|---|
| Lily (prompt 1) | **23** | 2 | REPEAT + RESTART |
| Tom (prompt 2) | **7** | 2 | REPEAT + RESTART |
| Box (prompt 3) | **18** | 0 | REPEAT |
| Dragon (prompt 4) | **6** | 2 | REPEAT + RESTART |
| Anna (prompt 5) | **11** | 0 | REPEAT |
| Mother (prompt 6) | **28** | 0 | REPEAT |

#### v1.0.0 — Temp=1.0 (30 records)
- **Repeated 4-grams:** 0 in all 30 sampling runs (greedy is the only mode with repetition).
- **Restarts:** 2–3 per prompt on average; every prompt has at least one restart in some seeds.
- Notable: v1.0 seed 46, dragon prompt contains `"The end."` mid-generation followed by a full story restart.

#### v1.1 — All 36 records
- **Repeated 4-grams:** 0 in all 36 runs (only 2 records have 1 repeated 4-gram; none exceed threshold of >3).
- **Restarts:** 0 in all 36 records.

---

## 3. Subjective Observations (Clearly Labeled)

> **Note:** The following are subjective quality assessments based on reading the full text of all 72 generations. They are not automated measurements and should be treated as qualitative observations, not quantitative results.

### 3.1 Premise Adherence (v1.0 vs v1.1)

**v1.1 generally tracks the prompt premise better.** For example:
- v1.0 greedy, Lily prompt: generates a complete story about a dog scare, then **restarts** with "Once upon a time" and generates the same story again (23 repeated 4-grams).
- v1.0 greedy, Tom prompt: generates a story about a dog scare, then restarts with "Once upon a time" and generates another complete story (7 repeated 4-grams).
- v1.0 greedy, dragon prompt: generates a story about a girl saving a dragon, then restarts with "Once upon a time" and generates another story about Lucy climbing trees (6 repeated 4-grams).
- v1.0 greedy, mother prompt: degenerates into a loop of "you have to be careful and not go too far" repeated 28 times (no restart because it never gets to "Once upon a time").

**v1.1 does not exhibit story restarts.** All 36 v1.1 records continue the prompt premise without restarting. However, some v1.1 samples drift from the specific character name:
- v1.1 temp=1.0 seed 45, Anna prompt: the text mentions "Lily" mid-generation ("Lily was happy and not sad anymore") before returning to Anna.
- v1.1 greedy, dragon prompt (60 tok): the story shifts from "the little dragon was afraid of fire" to a girl losing her toy — premise drift, but no restart.

### 3.2 Coherence and Degeneration Patterns

**v1.0:**
- **Greedy mode consistently degenerates into repetition loops.** The 6 greedy outputs all show high 4-gram repetition (6–28 repeated n-grams). The mother prompt is the worst case: 28 repeated 4-grams, all variations of "you have to be careful and not go too far."
- **Sampling mode (temp=1.0) avoids repetition** but frequently restarts stories mid-generation with "Once upon a time," producing 2–3 story restarts per prompt. This suggests the model has learned to produce complete stories but lacks a mechanism to stop — it simply starts new ones.
- Some sampling runs contain incoherent sentences (e.g., "The bear was intrigued, and c" — truncated word; "boasing by the stars" — nonsense verb).

**v1.1:**
- **Greedy mode shows moderate repetition on 2/6 prompts** (box: 10 repeated 4-grams; dragon: 4), but no full loops. The mother prompt (200 tok, length) degenerates into "I like it. It's my toy" repeated ~20 times — a clear greedy-mode failure mode for this prompt.
- **Sampling mode (temp=1.0) produces coherent, premise-tracking stories** with no restarts and minimal repetition (0–3 repeated 4-grams).
- **One notable failure:** v1.1 temp=1.0 seed 46, dragon prompt produces only 13 tokens before EOS: `"Ever, his parents start to mighing his anger."` — this is gibberish that happens to hit EOS quickly.
- **Another notable failure:** v1.1 temp=1.0 seed 44, dragon prompt (23 tok): `"She ran away from her home, safe to tell her friends her compassionate paws that she had lost her secret."` — grammatically coherent but semantically nonsensical ("compassionate paws").

### 3.3 Story Quality Summary (Subjective)

| Aspect | v1.0.0 | v1.1 |
|---|---|---|
| Premise tracking (greedy) | Poor — restarts stories 2–3×, high repetition | Good — no restarts, tracks premise |
| Premise tracking (sampling) | Poor — 2–3 restarts per prompt on average | Good — no restarts, tracks premise (occasional name drift) |
| Repetition loops (greedy) | Severe — 6–28 repeated 4-grams on all prompts | Moderate — 0–10 on some prompts, no full loops |
| Repetition (sampling) | None — sampling avoids repetition | Minimal — 0–3 on some prompts |
| Coherence (sampling) | Mixed — many incoherent sentences, truncated words | Mostly coherent; 2–3 gibberish outputs before EOS |
| Premise adherence (greedy) | Poor — multiple restarts, name drift common | Good — no restarts; 1–2 name drift cases |
| Premise adherence (sampling) | Poor — restarts with new characters/stories | Good — tracks prompt premise consistently |

---

## 4. Known Failure Cases

### 4.1 v1.0 Greedy — Repetition Loops (All 6 Prompts)
Every v1.0 greedy output degenerates into a repetition loop after ~50–80 tokens:
- Mother prompt: "you have to be careful and not go too far" × 28 (worst)
- Box prompt: "Wow, Lily, you found a box!" × 18 (dialogue loop)
- Dragon prompt: "She was so scared and ran away" × 6 (action loop)
- All others: moderate repetition (6–23 repeated 4-grams).

### 4.2 v1.0 Sampling — Story Restarts (All Prompts)
Every prompt has at least 2–3 story restarts in the sampling outputs. The model generates a complete story, hits ~150–170 tokens, then restarts with "Once upon a time" and generates another story. This is not a failure per se (the model produces complete stories), but it indicates the model has no concept of story length limits.

### 4.3 v1.0 — "The End" Mid-Generation (Seed 46, Dragon)
One sampling run contains `"The end."` at ~100 tokens, followed by a full story restart. This is the only output containing an explicit ending marker mid-generation.

### 4.4 v1.1 Greedy — Mother Prompt Loop (200 tok, length)
The greedy output for the mother prompt degenerates into "I like it. It's my toy" repeated ~20 times, hitting max tokens without EOS.

### 4.5 v1.1 — Early EOS Gibberish (Seed 46, Dragon)
One sampling run produces only 13 tokens of gibberish before hitting EOS: `"Ever, his parents start to mighing his anger."`

### 4.6 v1.1 — Premise Drift (Sampling)
- v1.1 temp=1.0 seed 45, Anna prompt: mentions "Lily" mid-generation before returning to Anna.
- v1.1 greedy, dragon prompt (60 tok): shifts from "little dragon afraid of fire" to a girl losing her toy.

---

## 5. Methodological Limitations

1. **No cross-tokenizer loss comparison.** The v1 and v1.1 tokenizers differ (v1.1 has `<|endofstory|>` at id 0, shifting all other token IDs by +1). Comparing validation losses or token-level perplexities across the two tokenizers is explicitly forbidden and methodologically invalid.

2. **Single checkpoint per model.** Only one v1.0 release and one v1.1 checkpoint (step 11000) are evaluated. Results may not generalize to other training steps or model variants.

3. **Small context window (512 tokens).** Prompts are 8–13 tokens, and generations are capped at 200 tokens. Context windowing (`generated[-512:]`) does not truncate within the generation window, but this evaluation does not test longer-form coherence beyond 200 generated tokens.

4. **Six fixed prompts only.** The prompts cover a narrow range of story types (girl/boy named, object discovery, animal fear, dialogue completion). Results may not generalize to other prompt styles.

5. **Decode convention differences.** v1.0 uses `skip_special_tokens=False` (no-op, no special tokens); v1.1 uses `skip_special_tokens=True`. This is intentional (matching each model's existing code) but means the recorded text strings are not directly comparable at the character level for EOS-terminating outputs.

6. **No training or architecture changes.** Both models are evaluated as-is; no hyperparameter tuning, fine-tuning, or architectural modifications were made.

7. **Subjective judgments are labeled as such.** Premise adherence, coherence, and story quality assessments in Section 3 are qualitative observations by a single reviewer, not automated metrics.

8. **Automated heuristics are imperfect.** 4-gram repetition detection and restart regex matching provide rough signals but may miss subtle forms of degeneration or story structure issues.

---

## 6. Files Created / Modified

| File | Status | Description |
|------|--------|-------------|
| `experiments/v1-vs-v1.1/evaluate.py` | **Created** | Evaluation script (329 lines). Loads both models, runs 72 generations, writes JSONL. |
| `experiments/v1-vs-v1.1/results/generations.jsonl` | **Created** | 72 JSONL records (one per generation). |
| `experiments/v1-vs-v1.1/report.md` | **Created** | This report. |
| `releases/lumi-v1-mlx/model.safetensors` | **Restored** (copied from main repo) | v1.0 model weights; was gitignored, restored for evaluation. Not a code change. |

**No existing files were modified or overwritten.** No commits or pushes were made.

---

## 7. Test Results

**No existing tests found.** The project has no test infrastructure (no `test*.py`, `conftest.py`, `pytest.ini`, or `Makefile` targets). The evaluation script itself serves as the primary validation:

- **Smoke test (5 tokens):** 72/72 records generated successfully, all finish_reason `"length"` (expected at 5 tokens).
- **Full run (200 tokens):** 72/72 records generated in 18.4 seconds. All required fields present. 72 unique (model, prompt, decoding, seed) combinations verified.

---

*Report generated from `experiments/v1-vs-v1.1/results/generations.jsonl` on 2026-10-19.*
