# BTC-H1b — Understanding the Negative Result: exploratory block X1 + X2 + X7

**Date:** 2026-11-14 · **Authorisation:** owner, "CLÔTURE BTC-H1 v0.3 ET OUVERTURE BTC-H1b" (CPU only; X1, X2, X7; 4 threads; 5 CPU-h) · **Status:** completed once, **STOPPED.**

> **Status of every number here: EXPLORATORY (POST HOC).**
> - PROBE-DEV was already used to select the H1 v0.3 probe. Nothing here is confirmatory, and nothing changes the official **H1 NOT MET**.
> - **No access to LOCKED-TEST.** No GPU, no new pool, no H2/H3 work.
> - The H1 v0.3 archive was only read: archived copies and originals are unchanged, verified against `ARCHIVE_MANIFEST.json`.

## 1. Set-up

**Script:** `src/h1b_explore.py`, sha `2fa1f3e4…`, committed before the run (git `0dc4409`).
- Plan: `outputs/h1b/PLAN.json`, written before any fit. The X7 thresholds were fixed there.
- Outputs: every configuration in `outputs/h1b/results.jsonl` (645 fits); summary `outputs/h1b/SUMMARY.json`.

**Run:**
- 2026-11-14 14:41:40 → 14:46:49 UTC.
- **0.200 CPU-h of 5** (`time.process_time()`), 4 threads.
- No fit reached the 60-epoch cap.

**Splits, all family-disjoint:**

| Split | Families | Training-rule worlds |
|---|---|---|
| inner-train | 181 | 2,112 |
| inner-val (early stopping and **every** choice) | 45 | 513 |
| **DEV-val** (reported only, never used for a choice) | 74 | 855 |

- DEV-val contains 77 depth-4 `upath` pairs.
- Training rows follow the H1 v0.3 rule: non-decoy worlds whose forced choice equals the oracle optimum, with no tie.

**Chance levels:**
- Single-world candidate decoding: pooled 0.196 for slots that exist at depths 3 and 4; 0.167 for c3/u3 (depth 4 only).
- u2 paired accuracy: 0.028.

## 2. X1 — positive controls (sensitivity of the decoding chain)

### 2a. Scalar task variables (linear logistic probes; DEV-val accuracy; λ chosen on inner-val)

| Vector | Model's chosen letter (majority 0.579) | Depth 3 vs 4 (majority 0.643) | Reward gap > median (majority 0.579) |
|---|---|---|---|
| 0 (embeddings) | 0.579 | 0.643 | 0.579 |
| 16 | 0.546 | **1.000** | 0.646 |
| 32 (ℓ\* of H1) | 0.587 | 1.000 | 0.804 |
| 36 | 0.767 | 1.000 | 0.875 |
| 40 | **0.967** | 1.000 | **0.932** |
| 48 | 0.994 | 1.000 | 0.936 |
| 60 / 61 | **1.000** / 1.000 | 0.999 | 0.871 / 0.870 |
| `mean_emb` | 0.532 | 1.000 | 0.487 |

All 17 vectors are in `SUMMARY.json`.

**Prompt-only reference:**
- Depth and reward gap can be read exactly from the prompt, so the prompt alone gives 1.0 on both.
- The choice requires following paths, so no trivial prompt rule exists for it.

**Reading:**
- The activations are informative, and the standardise-and-linear-probe pipeline is sensitive.
- The chosen letter becomes linearly decodable from vector 36–40 onwards (~0.97–1.0).
- The reward gap peaks at 40–48 (~0.93).
- These are descriptive layer profiles of a single forward pass, **not** time.

### 2b. Candidate-label decoding (the H1 bilinear probe, r = 8; DEV-val single-world accuracy; train = inner-train)

| Slot (relation to prompt) | Prompt baseline B3+ (val) | Probe, best vector on DEV-val (shown, not selected) | Probe train (inner-train) |
|---|---|---|---|
| **A1** (label printed in "At S, action A leads to X"; **purely lexical**) | **1.000** | 0.213 (vector 16) | 0.75–0.82 |
| **c1** (printed in the chosen letter's action line) | **1.000** | 0.221 (61) | 0.62–0.74 |
| u1 (printed in the other action line) | 1.000 | 0.257 (16) | 0.68–0.82 |
| c2 (1 hop) | 0.352 | 0.249 (48) | 0.60–0.69 |
| **u2** (1 hop; the H1 target) | 0.327 (paired 0.156) | 0.261 (48) (paired 0.000) | 0.56–0.68 |
| c3 (2 hops) | 0.247 | 0.244 (48) | 0.74–0.84 |
| u3 (2 hops) | 0.222 | 0.216 (48) | 0.73–0.80 |

- The val values at all 5 vectors (0, 16, 32, 48, 61) lie between 0.13 and 0.26 for every slot (chance 0.17–0.20).
- At vector 0 the probe stays at chance (train 0.13–0.22).

**Key observation:**
- The same decoder that gave the H1 result **cannot read even a label that is printed verbatim next to the chosen letter** (c1, A1): chance on DEV-val, against 1.0 for the prompt baseline.
- Meanwhile, linear probes on the **same activations** read scalar task variables almost perfectly (2a).
- The candidate-decoding chain therefore has **no demonstrated sensitivity to label identity at this position.** Its failure on u2 is not informative about whether u2 is represented.

## 3. X2 — capacity, regularisation, learning curves (targets u2 and c1; vectors 16, 32, 48)

All 78 configurations (each a 3-seed average) are in `results.jsonl`: bilinear r ∈ {2, 8, 32} × λ ∈ {0.1, 1, 10}, PCA-reduced inputs k ∈ {16, 64, 256}, and an MLP head.

| Target | DEV-val range over the 39 configurations | Best on inner-val → its DEV-val | Train range | Prompt baseline (val) |
|---|---|---|---|---|
| u2 | 0.158–0.296 (paired 0.000–0.039) | vector 48, r = 32, λ = 10: **0.296** (paired 0.039) | 0.29–0.95 | 0.327 (paired 0.156) |
| c1 | 0.147–0.256 | vector 48, r = 8, λ = 0.1 (inner-val 0.250): **0.172** | 0.37–0.96 | 1.000 |

**Pattern:**
- **Stronger regularisation, smaller rank or PCA reduction lower train accuracy but do not raise val accuracy above about 0.30.**
- The overfitting gap closes from above (train drops) rather than from below.
- Vector 48 is somewhat better than 16/32 for u2 (val 0.25–0.30 vs 0.19–0.23). It stays below B3+ on single worlds, and far below on pairs.

**Learning curves** (inner-val-best configuration, fractions of inner-train families):

| Target | 25 % (512 worlds) | 50 % (1,029) | 100 % (2,112) |
|---|---|---|---|
| u2, DEV-val (paired) | 0.232 (0.013) | 0.228 (0.000) | 0.296 (0.039) |
| c1, DEV-val | 0.198 | 0.212 | 0.172 |

- The u2 curve rises only at 100 %, with large noise.
- There is no clear trend that more DEV data would close the gap to the prompt baseline, still less for the lexical control c1.

**MLP head: degenerate (limitation).**
- All 18 MLP fits early-stopped at epoch 1–3, with train 0.29–0.41 and val 0.16–0.20.
- Validation loss rose immediately, so this head was effectively untrained.
- It **does not test** non-linear decodability. This is a negative implementation result, recorded as such and not tuned after the fact.

## 4. X7 — confidence and depth strata

- **Confidence** = |logp_A − logp_B|. The tertile thresholds, 12.81 and 19.12, come from DEV-train training-rule worlds and were fixed in PLAN.json before any accuracy was computed.
- **Results are on DEV-val:** n per cell, accuracy, and Wilson CIs in SUMMARY.json.

| Stratum | n | Admissibility, all non-decoy val worlds (n) | B3+ u2 | Probe u2 (vector 32) | B3+ c1 | Probe c1 (vector 32) |
|---|---|---|---|---|---|---|
| low confidence, depth 3 | 46 | 0.61 (75) | 0.435 | 0.304 | 1.000 | 0.174 |
| low, depth 4 | 222 | 0.65 (340) | 0.225 | 0.149 | 1.000 | 0.171 |
| mid, depth 3 | 98 | 0.98 (100) | 0.520 | 0.306 | 1.000 | 0.224 |
| mid, depth 4 | 194 | 0.86 (226) | 0.165 | 0.206 | 1.000 | 0.191 |
| high, depth 3 | 161 | 1.00 (161) | 0.571 | 0.236 | 1.000 | 0.224 |
| high, depth 4 | 134 | 1.00 (134) | 0.261 | 0.172 | 1.000 | 0.157 |

**Reading:**
- Model confidence predicts **behavioural** correctness strongly: admissibility 0.61–0.65 in the low tertile vs 1.00 in the high tertile.
- Probe accuracy shows **no** increase with confidence for either u2 or c1.
- Depth-3 worlds are easier for the prompt baseline (4 candidates, chance 0.25), and the probe follows only weakly.
- **E7 (competence-limited representation) gains no support:** even the most confident worlds give probe accuracies near chance.

## 5. Explanations E1–E7 in the light of the DEV observations (none established)

| ID | Explanation | DEV evidence from this block | Current status |
|---|---|---|---|
| E1 | Unchosen state not computed (shortcuts) | Cannot be judged: the decoder fails equally on information that is certainly in the prompt (c1, A1). The u2 failure no longer counts against E1 **or** for it | **Undetermined** (not testable with this decoder) |
| E2 | Decoder capacity / data | Rank, λ, PCA and data fraction move train accuracy, not val accuracy; the lexical control also fails | **Weakened** as the main explanation (within the families tested; MLP untested) |
| E3 | Static label embeddings are a poor key | Consistent with the strongest observation: scalars are decodable, labels are not, even lexical ones | **Most compatible**, not established |
| E4 | Wrong position (labels not carried at `<channel\|>`) | Also compatible: the final token may carry the decision (choice 1.0) without carrying label identities | **Compatible; indistinguishable from E3 on CPU** |
| E5 | Unsampled layers | Profiles over 17 vectors are smooth (choice, gap); no sign of a narrow missed peak | Unlikely, not excluded |
| E6 | Only the chosen branch is represented | c1 is no better than u1/u2 (all near chance); B7 / c3 hints from TEST cannot be checked with this decoder | **Not supported by DEV here** |
| E7 | Competence-limited | Confident worlds do not decode better | **Not supported** |

## 6. Recommendation: next discriminating experiment

**Priority: a combined X3 + X4 DEV capture, which needs a GPU and a new decision.**
- Capture hidden states at the **label-occurrence positions** in the prompt (every rule-line token span) as well as at `<channel|>`, over a denser set of layers.
- This one capture supports two analyses:
  - **(X3) pointer decoder:** score each candidate by the similarity between z (at `<channel|>` or at the action-line positions) and that label's **own contextual state**. This replaces the static key that failed (E3).
  - **(X4) position analysis:** where the label of each slot is decodable along the prompt (E4).
- **The decisive internal positive control is c1/A1.**
  - A decoder that cannot recover c1, which is printed next to the chosen letter, is not a valid instrument for u2.
  - If the pointer decoder recovers c1/A1 well above chance, its u2 result becomes informative about E1.
  - If it still fails on c1 at `<channel|>` but succeeds at the action-line positions, label identity is simply not carried to the answer position (E4).
- **Why this rather than X6** (a task without shortcuts): changing the task before having a validated instrument would add a second unknown. A valid instrument comes first, then the task.

**Indicative cost for a decision** (not requested here): one PROBE-DEV re-capture with extra positions, of the same order as D3-DEV-R1 (≈ $0.6–1.5), larger storage, and a GPU request in the usual format.

Any resulting hypothesis would need a new frozen protocol and a new, independent sealed TEST pool.

## 7. Limitations

- **Re-use of PROBE-DEV:** it was already used for selection. Inner-val choices limit, but do not remove, that dependence.
- **Single DEV-val split:** 74 families and 77 u2 pairs, so single-world differences of a few points are noise. The confidence intervals are in SUMMARY.json.
- **MLP head degenerate (§3):** non-linear decoding is untested.
- **Positions:** all decoding is at the single `<channel|>` position, the only one captured.
- **B3+:** it includes the action-line letter feature, which makes it perfect for c1/u1/A1 by construction. That is the intended reference for "information present in the prompt".

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (no H2/H3 work)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (PROBE-DEV only; LOCKED-TEST not accessed; archive unchanged)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (CPU block authorised; 0.20 / 5 CPU-h)
[YES] Preserved negative results and raw logs (all 645 fits, MLP failure reported)
[YES] Labeled unrun stages NOT RUN and limitations honestly (X3–X6 not run)
Any deviation: none.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decision on a combined X3 + X4 DEV capture (GPU).
```
