# Before the Choice — BTC-H1 v0.3 final scientific report

**Date:** 2026-11-14 · **Status:** CLOSED. Official owner verdict: **H1 NOT MET** ("DÉCISION SCIENTIFIQUE — CLÔTURE BTC-H1 v0.3").

**Scope of this report:**
- H1 only. H2 and H3 were not tested.
- LOCKED-TEST is now **consumed**. No further computation on it is permitted, except integrity checks of seals and logs.

> **One-sentence result.** At the pre-answer position of Gemma-4-31B-it, a pre-registered low-rank bilinear probe did **not** decode the unchosen branch's depth-2 state beyond prompt-only, permuted-capacity and anchor-and-eliminate baselines: paired accuracy 0.020 vs B3 0.195, on 293 admissible pairs from 200 held-out families.

---

## 1. Question and hypothesis

- **Research target:**
  - The study asks whether a language model's activations, **before** it emits its choice, encode the future path of the action it will **not** choose.
  - This would be a candidate functional analogue of hippocampal forward sweep.
  - It is not a claim about agency, consciousness or welfare.
- **H1 (primary):**
  - The route-depth-2 state of the **unchosen** branch is decodable from pre-answer activations.
  - The decoding must beat strong lexical/prompt, capacity and answer-based baselines.
- **Evidential ladder:**
  - L1: decodable beyond baselines.
  - L2: ordered, and label- and order-general (same-layer order test, alias and shuffle generalisation).
  - L3 and L4 were out of reach by design.

## 2. Methods (frozen before any TEST access; `reports/BTC-H1_CONFIRMATORY_FREEZE_PROPOSAL.md` v0.3)

| Component | Specification |
|---|---|
| Task | Synthetic deterministic two-action maze, depths 3–4. Local unordered transition rules, terminal rewards, private oracle. 16 controlled variants per family, including `upath` (unchosen-path counterfactual that preserves the choice), `cpath`, `xswap`, `alias`, `shuffle`, `decoy` |
| Model | `google/gemma-4-31B-it` @ `842da3794eaa0b77d5f08bae87a17459d91ff475`, bf16, first-token A/B forced choice (ids 236776 / 236799). Selected after a pre-registered competence check (V2-C1 PASS); the first candidates (Qwen3-4B, Qwen3-32B, Olmo-3.1-32B) failed the screens |
| Data | PROBE-DEV (seed 2026111401, 4,800 worlds, 300 families) for training and selection. **LOCKED-TEST** (seed 2026111402, 4,800 worlds, 300 independent families) generated and sealed at freeze |
| Representation | 62 vectors at the `<channel\|>` pre-answer token (61 HF hidden states + raw block-60 pre-norm), `mean_emb`, frozen candidate embeddings e(c). Capture fidelity: max\|Δ\| ≤ 0.25 over the full vocabulary, A/B decision preserved (D3-A2/A3) |
| Primary endpoint | **Paired accuracy** on admissible depth-4 `upath` pairs (base→upath_base, cf→upath_cf). Target: the unchosen route-depth-2 label among 6 prompt-derived candidates. Chance 1/36. Admissible = forced choice equals the oracle optimum, and no exact tie |
| Probe | s(c) = zᵀUVᵀê(c). Grid: 17 vectors × r {8, 32} × λ {1e-4…1e-1} × 3 seeds. Selection on DEV-val; final probe = average of 3 seeds refit on all DEV. Candidates in canonical (label-sorted) order; no oracle input |
| Baselines (thresholds) | **B1** `mean_emb` probe. **B2** same probe trained on paired permuted targets, scored vs true u2, max over 3 permutations inside each bootstrap replicate (C-1). **B3** prompt-text conditional logit (7 features, model's chosen letter). **B6** four anchor heads with joint distinct assignment, expectation p₁·p₂ |
| Decision | INSUFFICIENT if < 180 pairs or < 100 families. **MET** iff the 95 % family-cluster bootstrap lower bound (10,000 resamples, seed 0) of probe − B_k is > 0 for every k (intersection-union). L2 only if MET |
| Integrity | DEV-only selection; 16 predictors hashed before TEST; frozen DEV standardisation; stage-scoped sealed access (capture = prompts only; analysis = separate owner unlock); single evaluation guarded by a marker; all accesses logged |

## 3. DEV results (selection stage; exploratory)

- **Selected probe:**
  - ℓ\* = 32, r = 8, λ = 0.1, E = 5.
  - DEV-val paired accuracy **0.030** (chance 0.028). No vector exceeded 0.030 on average.
- **B3:** 0.156.
- **Anchor heads:** 0.24–0.31 on DEV-val vs 0.80–0.93 on DEV-train. This is strong overfitting; no fit hit the epoch cap.
- **Compute:** 1.475 CPU-h of 48.
- **Report:** `reports/BTC-H1-DEV-TRAIN_RESULTS.md`.

## 4. Confirmatory results on LOCKED-TEST (single evaluation, 2026-11-14 14:06:58–14:07:28Z)

**4.1 Admissibility**

| | n |
|---|---|
| Worlds | 4,800 |
| Exact A/B ties (excluded) | 9 |
| Forced choice ≠ oracle optimum (excluded) | 1,000 |
| Admissible | 3,791 |
| Admissible decoys not evaluable (10 or 7 candidates vs 6-way decoders) | 359 |
| **Primary pairs / families** | **293 / 200** (sufficient) |

**4.2 Primary endpoint** (paired accuracy; chance 0.028)

| Decoder | Accuracy | Δ = probe − B [95 % CI] | Criterion |
|---|---|---|---|
| **H1 probe** | **0.020** [0.007, 0.038] | — | — |
| B1 | 0.000 | +0.020 [+0.007, +0.038] | pass |
| B2 (max of 3) | 0.020 (0.020 / 0.007 / 0.000) | 0.000 [−0.021, +0.020] | **fail** |
| B3 | 0.195 | **−0.174** [−0.220, −0.129] | **fail** |
| B6 | 0.020 | +0.001 [−0.016, +0.020] | **fail** |
| B5 (reported) | 0.000 | — | — |
| B4 | 0 by construction | — | — |

**Verdict: H1 NOT MET. L2 not evaluated. No L1 or L2 attribution.**

**4.3 Secondary analyses** (all pre-registered items, each with its status)

| ID | Status | Result |
|---|---|---|
| S1 pooled depth 3 + 4 | inferential, Holm | 467 pairs; probe 0.015 vs B3 0.319; Holm p = 1.0 |
| S2 `xswap` | inferential, Holm | 286 pairs; probe 0.007 vs B3 0.234; Holm p = 1.0 |
| S3 per-depth heads | descriptive | u1 0.191, u2 0.198, u3 0.205; anchors c1 0.188, c3 0.363, u1 0.238, u3 0.274 (chance 0.167; single-world, depth-4 base/cf) |
| S4 decoy | **not computable** | candidate-set size incompatible with the frozen decoders |
| H2 profiles | **not run** | would need new training |
| B7 `cpath` | descriptive reference | 0.119 |

The Holm family {S1, S2} is a subset of the pre-registered secondary plan, not the whole plan.

## 5. Interpretation

**What the result establishes:**
- Under a pre-registered, DEV-selected decoder and on independent held-out families, there is **no evidence** that pre-answer activations carry unchosen-branch depth-2 information beyond what the prompt text alone gives.
- In fact the prompt-only baseline is **far stronger**.
- The DEV observation replicated on TEST.

**What it does not establish:**
- **Absence** of such a representation in the model.
- The negative result is conditional on all of the following:
  - the decoder family (low-rank bilinear with static label embeddings);
  - the single position (`<channel|>`);
  - the 17 sampled vectors;
  - the training-set size (about 2,600 worlds against d = 5,376);
  - this task, in which the anchor-and-eliminate and backward-from-reward shortcuts exist (P0).
- Nothing about H2 (dynamics) or H3 (causal use), which were not tested.

**Tempting but unjustified readings:**
- "Gemma does not plan ahead."
- "The model only represents the chosen path." The descriptive c3 and B7 signals are not baseline-controlled.
- "Layer 32 is where planning happens."

**Sensitivity:** with 200 depth-4 families, the design had power ≥ 0.93 for an advantage of +0.15 over the strongest baseline, and 0.62–0.76 for +0.10 (D5 §R1). The observed effect is negative relative to B3, so the verdict does not depend on power.

## 6. Limitations

1. **Single decoder family**, chosen on cost grounds (v0.3 grid). DEV overfitting suggests the probe was data-limited rather than unable to fit.
2. **Static candidate embeddings.**
   - Labels are random 4-character strings split over 1–5 tokens.
   - The model's in-context representation of a label may bear little relation to its mean token embedding.
3. **One token position.** Information may be carried at rule-line or label-token positions instead.
4. **Task shortcuts** (P0). The choice can be solved without computing the unchosen depth-2 state, so the model may not compute it.
5. **Admissibility:** 21 % of worlds were excluded because the forced choice was not oracle-optimal. Behavioural competence is moderate at depth 4.
6. **Uncomputed analyses:** decoy control (S4) not computable; H2 profiles not run.
7. **Operational incidents** (preserved, not affecting the result):
   - D3-DEV strict top-1 abort, fixed by amendment D3-A2;
   - D3-TEST attempt-1 startup crash, $0, fixed in R1.

## 7. Post-evaluation status of seals and logs

| Object | Status |
|---|---|
| LOCKED-TEST-D5 pool seal | intact, read-only. Access log: 4 entries (2 capture-scope prompt reads, 2 H1-TEST-EVAL reads: prompts and oracle). **Consumed** |
| LOCKED-TEST-D3-ACT activation seal | intact, read-only, 40 files. Access log: 39 H1-TEST-EVAL entries |
| Unlock files | D3-TEST-CAPTURE (R1, with attempt 1 embedded and archived read-only); H1-TEST-EVAL (pool and activations) |
| Historical test `tests/test_d5_dev.py::test_locked_test_sealed_read_only` (D3-A4) | **Kept unchanged in its historical state.** It now fails by design: it allowed only capture reads before the analytical unlock |
| Proposed replacement check | `tests_proposed/test_post_eval_seal.py`. Not in any suite until approved. It accepts only properly authorised reads (approval ids from the unlock files; analytical reads after the `EVAL_STARTED` time), requires intact seals and a single evaluation, and checks that both logs are **append-only** against the closure snapshot. It deletes nothing and changes no result |

## 8. Archive

Everything is archived in `outputs/archive/BTC-H1-v0.3/ARCHIVE_MANIFEST.json`:
- the protocol, the 16 predictors, the standardisation statistics, all hashes, logs, access-log snapshots and unlock files;
- the results and reports.

The small artifacts are copied and set read-only. The two 3.4 GB activation sets are referenced by seal or manifest hash.

## 9. Spend

**Cloud total:** about **$2.43** (Modal billing reports).
- V2 screens/competence: about $0.86.
- D2: about $0.09.
- D3-DEV: $0.17 (failed) + $0.579 (R1).
- D3-TEST: $0.00 (attempt 1) + $0.746 (R1).

**CPU:** 1.5 CPU-h for DEV training, plus about 1 minute for the TEST evaluation.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (H2/H3 not tested)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (LOCKED-TEST used once; now consumed)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change
[YES] Preserved negative results and raw logs (all comparisons reported; no selection)
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none. No computation on LOCKED-TEST beyond the single evaluation.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decision on the BTC-V2 study design note.
```
