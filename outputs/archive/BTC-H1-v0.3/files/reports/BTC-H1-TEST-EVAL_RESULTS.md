# BTC-H1-TEST-EVAL — confirmatory H1 result on LOCKED-TEST: **H1 NOT MET**

**Date:** 2026-11-14 · **Authorisation:** owner, "AUTORISATION SCIENTIFIQUE — BTC-H1-TEST-EVAL" (single analytical access, single evaluation) · **Status:** completed once, **STOPPED.**

> **This is the pre-registered confirmatory result** (H1 v0.3 plus owner decisions C-1, C-2, O-2 and T-1). It was computed exactly once and with no adjustment.
> **Verdict:** H1 is **NOT MET** and L2 was not evaluated. Under v0.3 §7 the strongest permitted statement is: **H1 not supported beyond the baselines.** The frozen probe does not decode the unchosen depth-2 label better than the prompt-text baseline, the permuted-target control or the anchor-and-eliminate baseline.

## 1. Preconditions (before any analytical read): all PASS

The checks are in `outputs/h1_test_preconditions.json`. The evaluator repeated its own prechecks.

| Check | Result |
|---|---|
| 16 frozen predictors; `STANDARDISATION.npz` `9cd6b229…` | hashes equal FINAL.json, the freeze manifest and `prepared.BTC-H1-TEST-EVAL` |
| Evaluator `a967c42e20e3…` | equals `prepared.BTC-H1-TEST-EVAL`; all prepared code hashes equal |
| Seals LOCKED-TEST-D5 and LOCKED-TEST-D3-ACT | intact (`hash_only`); activation `SEAL_MANIFEST` `fe7c5756…` |
| Capture traces | pool access log, attempt-1 and R1 logs, capture unlock and its attempt-1 archive: all hashes unchanged |
| No earlier evaluation | `outputs/h1_test/` absent; no `EVAL_STARTED.json`; activations never read; oracle never read |

## 2. Execution

| Item | Value |
|---|---|
| Unlocks | `outputs/approvals/unlock/H1-TEST-EVAL/UNLOCK_LOCKED-TEST-D5.json` (`worlds.jsonl`, `model_inputs.jsonl`) and `UNLOCK_LOCKED-TEST-D3-ACT.json` (`*`), scope H1-TEST-EVAL, with your approval id |
| Command | `.\.venv-cpucheck\Scripts\python src\sealed_stage\h1_eval_test.py --run`, CPU, 4 threads, no GPU |
| UTC | 14:06:58 → 14:07:28 (30 s); EXIT 0; `EVAL_STARTED.json` written before the first sealed read |
| Outputs | `outputs/h1_test/EVAL_RESULT.json` (sha `e0219378…`), `outputs/h1_test/per_world.jsonl` (`5f3aa6de…`, 3,432 evaluable worlds) |

## 3. Admissibility and exclusions

| Item | Count |
|---|---|
| TEST worlds | 4,800 |
| Exact A/B ties (not admissible) | **9** |
| Forced choice ≠ oracle optimum (not admissible) | 1,000 (= 4,800 − 3,791 − 9) |
| **Admissible** | **3,791** |
| Admissible decoys excluded from every decoder (10 candidates at depth 4, 7 at depth 3, vs the frozen 6-way decoders) | 224 + 135 = 359 |
| Evaluable worlds | 3,432 (depth 4: 2,154; depth 3: 1,278) |
| **Primary population:** admissible depth-4 `upath` pairs | **293 pairs, 200 families**, so the INSUFFICIENT rule (≥ 180 pairs, ≥ 100 families) is **not** triggered |

## 4. Primary result (paired accuracy on the 293 pairs; chance 1/36 = 0.028)

| Decoder | Paired accuracy |
|---|---|
| **H1 probe** (ℓ\* = 32) | **0.020** (95 % CI 0.007–0.038) |
| B1 (`mean_emb`) | 0.000 |
| B2_0 / B2_1 / B2_2 (permuted-trained, scored vs true u2; C-1) | 0.020 / 0.007 / 0.000 → max **0.020** |
| **B3** (prompt text) | **0.195** |
| B6 (anchor-and-eliminate, expectation) | 0.020 |
| B5 (candidate only), reported | 0.000 |
| B4 | 0 by construction (not computed) |

**Intersection-union test:** family bootstrap, 10,000 resamples, seed 0, the same resamples for every comparison.

| Δ = probe − B_k | Point | 95 % CI | Lower bound > 0 |
|---|---|---|---|
| vs B1 | +0.020 | [+0.007, +0.038] | yes |
| vs B2 (max_k inside each replicate) | 0.000 | [−0.021, +0.020] | **no** |
| vs B3 | **−0.174** | [−0.220, −0.129] | **no** |
| vs B6 | +0.001 | [−0.016, +0.020] | **no** |

**H1 verdict: NOT MET.** Three of the four required comparisons fail. The probe is significantly **below** the prompt-only baseline B3.

**L2:** not evaluated (v0.3 §6.3: only if H1 is MET). No L1 or L2 attribution.

## 5. Secondary analyses (O-2: every pre-registered item with its status)

| ID | Analysis | Status | Result |
|---|---|---|---|
| S1 | depth 3 + 4 pooled `upath` | INFERENTIAL (Holm) | 467 pairs / 300 families. Probe 0.015 vs B3 0.319, Δ −0.304; one-sided bootstrap p = 1.0; **Holm p = 1.0** |
| S2 | `xswap`, depth 4 | INFERENTIAL (Holm) | 286 pairs / 197 families. Probe 0.007 vs B3 0.234, Δ −0.227; p = 1.0; **Holm p = 1.0** |
| S3 | per-depth reconstruction (admissible depth-4 base/cf, chance 1/6 = 0.167) | DESCRIPTIVE | order u1 0.191, probe u2 0.198, order u3 0.205; anchors c1 0.188, **c3 0.363**, u1 0.238, u3 0.274 |
| S4 | decoy control | **NOT COMPUTABLE** | 10 or 7 candidates vs 6-way frozen decoders; restricting the candidates would require the oracle |
| H2 | descriptive layer profiles | **NOT RUN** | each frozen head exists at one vector index; a profile needs new DEV training, which is not authorised |
| B7 | `cpath` (chosen branch) reference | DESCRIPTIVE | probe paired accuracy 0.119 on 286 pairs (not a threshold) |
| B4 | answer-only | NOT COMPUTED | 0 by construction |

The Holm family {S1, S2} is a **subset** of the pre-registered secondary plan, not the whole plan.

**Descriptive observations (POST HOC, not confirmatory, no inference drawn):**
- The anchor heads are modestly above chance, highest for c3, the chosen branch at depth 3.
- B7 (0.119) exceeds the primary unchosen-branch accuracy (0.020).
- Both observations agree with DEV. They suggest more decodable information about the **chosen** branch than the unchosen one. But they are not controlled against prompt baselines here and **do not support any claim.**

## 6. Access logs and hashes

**LOCKED-TEST-D5 access log** (4 entries, sha `66922721…`):
1. 06:00:20, `model_inputs.jsonl`, D3-TEST-CAPTURE (attempt 1)
2. 13:35:47, `model_inputs.jsonl`, D3-TEST-CAPTURE (R1)
3. 14:07:12, `model_inputs.jsonl`, H1-TEST-EVAL
4. 14:07:12, `worlds.jsonl`, H1-TEST-EVAL

**LOCKED-TEST-D3-ACT access log** (sha `79f0d261…`): 39 entries, all H1-TEST-EVAL. They cover the manifest, the labels, the embeddings and 12 × (index, acts, mean_emb).

**After the evaluation:** both seals are still intact and their files read-only.

**Expected test failure:** `tests/test_d5_dev.py::test_locked_test_sealed_read_only` now **fails, as designed**. Amendment D3-A4 allowed only capture reads until the H1-TEST-EVAL unlock, and the log now holds the two authorised analytical reads. I have not modified the test. Updating it to record the authorised H1-TEST-EVAL reads would need your decision.

**Manifest:** `BTC-H1_FREEZE_MANIFEST.json → artifacts.H1-TEST-EVAL` (hashes of results, marker, preconditions, log, unlocks, both access logs).

## 7. Interpretation and limits

**Justified:**
- Under the frozen protocol, pre-answer activations of Gemma-4-31B-it at the `<channel|>` position, decoded by the frozen bilinear probe at ℓ\* = 32, do **not** carry unchosen-branch depth-2 information beyond strong prompt and capacity baselines.
- This replicates the DEV observation.

**Not justified:**
- That the model has no representation of the unchosen path. The result is limited to:
  - this probe family (low-rank bilinear, about 2,600 training worlds, strong overfitting);
  - this position;
  - this grid of 17 vectors;
  - this task.
- Any conclusion about H2 or H3 (not tested). Any conclusion from the descriptive chosen-branch signals.

**Limits:**
- The decoy control (S4) could not be computed, and the H2 profiles were not run.
- B3 exploits the anchor-and-eliminate prompt shortcut identified in P0, so it is a strong baseline by design.
- About 21 % of worlds were not admissible: the forced choice differed from the oracle optimum.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (no H2/H3 experiment)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently (no retraining)
[YES] Preserved oracle isolation and train/val/test independence (single logged analytical access after freeze)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (BTC-H1-TEST-EVAL)
[YES] Preserved negative results and raw logs (NOT MET reported in full; per-world file kept)
[YES] Labeled unrun stages NOT RUN and limitations honestly (S4 not computable, H2 not run, L2 not evaluated)
Any deviation: none. Expected post-evaluation failure of the D3-A4 seal test reported, not modified.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decision on how to close or continue the study (no H2/H3 authorised).
```
