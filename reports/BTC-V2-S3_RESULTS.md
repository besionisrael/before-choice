# BTC-V2-S3 — Gemma-4-31B-it screening: **SCREEN PASS** (S1 ✓, S2 ✓, S3 ✓, S4 ✓)

**Date:** 2026-11-13 · **Authorisation:** owner, BTC-V2-S3, single GPU execution, separate transformers 5.23.0 / torch 2.6.0 environment for this run only · **Status:** executed once, scored, **STOPPED**. C1, D3 and later stages are NOT RUN and not authorised.

> A pre-registered **behavioural screening** result. It establishes that Gemma-4-31B-it chooses the optimal action on this SCREEN set far above chance with first-token answers. It is **not** the competence checkpoint (C1–C5, separate COMPETENCE pool) and **not** evidence for H1, H2 or H3, which remain unchanged and untested. No probes were trained and no activations were captured.

## 1. Pre-inference verification (owner conditions)

Run on CPU in `.venv-gemma4` (torch 2.6.0+cpu, transformers 5.23.0), then repeated as assertions inside the container:

| Check | Result |
|---|---|
| Revision | `842da3794eaa0b77d5f08bae87a17459d91ff475` |
| `chat_template.jinja` sha256 | `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4` (archived `outputs/v2_audit/gemma4_31b_chat_template.jinja`) |
| Rendered prompt | `<bos><\|turn>user\n…Respond with A or B only.<turn\|>\n<\|turn>model\n<\|channel>thought\n<channel\|>`, with a single `<bos>` |
| **Thought channel** | **empty**: nothing between `<\|channel>thought\n` and `<channel\|>`. The template renders the same empty channel with or without `enable_thinking=False`; we pass `False` as frozen. |
| Pre-answer position | last token `<channel\|>` |
| `A` / `B` ids | single tokens 236776 / 236799 |
| Frozen tail / token counts vs CPU-check reference | 240/240 / 240/240 |
| Pool hashes and prompt digest | `cd3499d6…` / `42cc0517…` / `6e52d9dc…` ✓ |

No memory, loading, tokenisation or software error occurred. No assertion fired in the container.

## 2. Run

| Item | Value |
|---|---|
| Command | `modal run src/v2_run_gemma.py --run-id BTC-V2-S3` (`5b25f5ee…`; `v2_remote` `fef0dcda…`; `v2_common` `8f6d6a53…`) |
| Modal app | `ap-yjwZVM5mErrWE8YuBbn29W`, exit 0, stopped. Single attempt, `retries=0`, `max_containers=1`; new Gemma image built (83 s) |
| Data | SCREEN only: 240 worlds + 20 replays = 260 forwards |
| Hardware / env | 1 × A100-SXM4-80GB, cpu=(4,4), memory=(16, 96) GiB, timeout 780 s / guard 720 s; torch 2.6.0+cu124, transformers 5.23.0, CUDA 12.4 |
| **GPU memory peak** | **63.1 GB** |

**Timing:**

| Phase | Seconds |
|---|---|
| Wait for GPU allocation (not billed) | 472.0 |
| Tokenizer | 8.8 |
| Weights download + load (62.5 GB) | **179.5** |
| 260 forwards | **32.7** (≈ 126 ms each) |
| Remote total | 221.0 |
| Local wall time | 692.7 |

The container logged an unauthenticated-HF-Hub warning (rate limits). It had no effect on the run.

## 3. Pre-registered criteria (applied without adaptation)

| ID | Criterion | Observed | Result |
|---|---|---|---|
| S1 | depth-2 base+cf primary accuracy ≥ 0.70 (n = 40, 20 fam.) | **1.000** (40/40; 0 ties; strict = lenient = 1.000); CI [1.00, 1.00] | **PASS** |
| S2 | depth-2 base/cf both correct ≥ 0.50 | **20/20 = 1.00** | **PASS** |
| S3 | A+B mass ≥ 0.5 in ≥ 95 % | 240/240 (min 0.541) | **PASS** |
| S4 | top-1 over the full vocabulary ∈ {A, B} in ≥ 95 % | **239/240 = 0.996** (1 world: top-1 "To") | **PASS** |

**SCREEN_PASS = True.**

## 4. Performance by depth and counterfactual pairs

| Depth | {base, cf} n | Ties | Primary | 95 % CI | base/cf both correct | base/cf choice changed |
|---|---|---|---|---|---|---|
| 2 | 40 | 0 | **1.000** | [1.00, 1.00] | **20/20** | 20/20 |
| 3 | 40 | 0 | **0.950** | [0.88, 1.00] | 18/20 | 18/20 |
| 4 | 40 | 0 | **0.750** | [0.65, 0.85] | **10/20** | 10/20 |

**By variant** (n = 60 each): base 0.900 · cf 0.900 · swap_base 0.950 · swap_cf 0.908 (1 tie).

**Choice changes where the optimum changes:**

| Pair | Choice changed |
|---|---|
| base → cf | 48/60 |
| swap_base → swap_cf | 52/60 |
| base → swap_base (letters exchanged) | 53/60 |

This is the profile the previous models lacked: the choice **tracks the decisive transition** and follows the path, not the letter. Sensitivity weakens at depth 4.

## 5. A/B choices, ties, margins, errors, reproducibility

**Choices:** A 138 · B 101 · tie 1 (`F10105-swap_cf`).

| Depth | A / B / tie | Accuracy when optimal = A | Accuracy when optimal = B | Median margin log p(A) − log p(B) | Median \|margin\| |
|---|---|---|---|---|---|
| 2 | 41 / 39 / 0 | 1.00 | 0.97 | +10.81 | 22.08 |
| 3 | 43 / 37 / 0 | 1.00 | 0.93 | +7.56 | 20.47 |
| 4 | 54 / 25 / 1 | 0.97 | **0.60** | +7.38 | 13.59 |

**Errors:** 21/240 in total.
- 17 at depth 4, 3 at depth 3, 1 at depth 2.
- **19 of the 21 errors are "A" choices.** A residual A default appears when the model fails, concentrated at depth 4 (accuracy 0.60 when B is optimal).
- Errors spread evenly across variants: base 6, cf 6, swap_cf 6, swap_base 3.
- **14 confident errors** (|margin| ≥ 3), 11 of them at depth 4. Examples: `F10136-base` (+19.8), `F10123-base` (+18.5), `F10137-swap_base` (+17.6).

**Non-A/B top-1:** `F10146-cf` (depth 4). Top-3 tokens are "To" (−0.80), "A" (−0.93), "B" (−1.93), with A+B mass 0.541; the forced choice is A, which is wrong.

**Reproducibility:** the 20 replays have identical choices and max |Δ log p| = 0.0.

**POST HOC / EXPLORATORY** (not pre-registered): at depth 4, accuracy is 20/28 for reward gaps ≤ 2 against 43/52 for gaps > 2. At depth 3 it is 34/36 against 43/44. Small gaps may contribute to depth-4 errors; this is descriptive only.

## 6. Cost (actual, `modal billing report --show-resources`)

| Resource | $ |
|---|---|
| A100-80GB | 0.1333 |
| CPU | 0.0136 |
| Memory | 0.0069 |
| Network egress | 0.0152 |
| **Run total** | **$0.169** |
| Gemma image build (billed separately) | ≈ $0.01–0.02 |

- Below the $0.85 worst case and the $1.00 objective. **No overrun.** $0.00 charged (credits).
- The 472 s allocation wait was not billed.
- **V2 total so far:** S1 $0.158 + S2 $0.245 + S3 $0.169 + 2 image builds ≈ **$0.60**.

## 7. Pre-registered selection rule (applied locally; launches nothing)

`python src/score_v2.py --select` → `outputs/v2/selection.json` (`f2028313…`):

| Run | Model | SCREEN_PASS | depth-4 accuracy |
|---|---|---|---|
| BTC-V2-S1 | Qwen3-32B | ✗ | 0.500 |
| BTC-V2-S2 | Olmo-3.1-32B-Instruct | ✗ | 0.550 |
| BTC-V2-S3 | **Gemma-4-31B-it** | **✓** | **0.750** |

**Decision: SELECTED = `gemma4_31b`** (the only passer). STOP V2 does **not** apply.

This selection **does not authorise C1**. The COMPETENCE pool (1,320 worlds) remains unopened by any model.

## 8. Reading and caveats (no overclaim)

**Justified:** with first-token A/B answering and no textual reasoning, Gemma-4-31B-it solves this SCREEN set's depth-2 and depth-3 problems almost perfectly. Its choices follow the counterfactual edge change, unlike Qwen3-4B, Qwen3-32B and Olmo-3.1-32B.

**Not justified:**
- That competence is established. That is C1–C5 on the fresh COMPETENCE pool, run once.
- That D-1′ is feasible. **Depth-4 accuracy on SCREEN is 0.75, close to the C4 threshold of 0.70**, and only 10/20 depth-4 base/cf pairs are both correct (C5 concerns `upath` admissibility, not measured here).
- Anything about prospective path representations, forward vs backward computation (P0 shows backward-from-reward is the cheapest route), or forward sweep. H1–H3 are untested.

**Other limits:**
- SCREEN has 20 families per depth.
- Gemma 4 runs on a different software stack than M1/M2.
- The model differs from M1/M2 in more than size (template, architecture, lab).

## 9. Artifacts

| Artifact | Path / hash |
|---|---|
| Raw outputs | `outputs/v2/BTC-V2-S3/behavior_raw.jsonl` (`1c095226…`) |
| Manifest (template check, timing, env, cost, decision) | `outputs/v2/BTC-V2-S3/manifest.json` |
| Scored metrics + per-world | `outputs/v2/BTC-V2-S3/scored.json` (`f0debab2…`) |
| Selection | `outputs/v2/selection.json` (`f2028313…`) |
| Logs | `outputs/logs/v2s3_run.log`, `v2s3_score.log`, `v2s3_billing_resources.csv`, `v2_selection.log`, `v2s3_{start,end}_utc.txt` |

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (unchanged, untested)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently (pinned 842da379…)
[YES] Preserved oracle isolation and train/val/test independence (SCREEN only; COMPETENCE unopened)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (BTC-V2-S3; Gemma stack for this run only)
[YES] Preserved negative results and raw logs
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none. Selection rule applied locally as pre-registered (no GPU).
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decision on BTC-V2-C1 (Gemma-4-31B-it, COMPETENCE pool, once).
```
