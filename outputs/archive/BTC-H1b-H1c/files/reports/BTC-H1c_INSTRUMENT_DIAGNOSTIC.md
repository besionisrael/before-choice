# BTC-H1c — instrument diagnostic on the existing DEV activations (POST HOC / EXPLORATORY)

**Date:** 2026-11-14 · **Owner decision:** "DÉCISION SCIENTIFIQUE — BTC-H1c, DIAGNOSTIC DEV" (CPU only, 1 CPU-h cap, 4 threads) · **Status:** completed, **STOPPED.**

> **Status of these results:** every number here is **POST HOC**. They do **not** change the frozen protocol, amendment A1, or the historical DEV result.
> **What was used:** the existing DEV activations only (`outputs/h1c_dev/capture/`). No GPU, no new pool, no TEST.

**Run:**
- Script: `src/h1c_instrument_diagnostic.py`, committed before the run (git `49c1fe9`).
- Output: `outputs/h1c_dev/diagnostic/DIAGNOSTIC.json` (sha `1ef5cc60…`).
- Compute: **0.027 CPU-h of 1.**

**Settings:**
- Splits: as in the DEV analysis. Inner-train 852 / inner-val 211 admissible worlds (DEV-train families); DEV-val 383 worlds (100 distinct families).
- Decoder: the same pointer (r = 8, λ = 0.1, 3 seeds, early stopping on inner-val). **No new hyper-parameter search.**

## 1. PC1 by layer (item 1) and overfitting (item 2)

**12-way c1 pointer** (ANS query × 12 arrival keys; chance 1/12):

| Layer | train | inner-val | **DEV-val** | best epochs |
|---|---|---|---|---|
| 24 | 0.852 | 0.275 | 0.342 | 4, 2, 2 |
| 32 | 0.937 | 0.464 | 0.499 | 4, 2, 3 |
| **40** | 0.993 | **0.711** | **0.752** | 4, 5, 6 |
| 48 | 1.000 | 0.659 | 0.595 | 6, 6, 9 |
| 56 | 0.975 | 0.417 | 0.399 | 2, 4, 3 |

**Binary continuation decoders** (chance 0.5):

| Layer | **unchosen (P)** train / inner-val / DEV-val | **chosen (PC2)** train / inner-val / DEV-val |
|---|---|---|
| 24 | 0.642 / 0.521 / 0.475 | 0.845 / 0.488 / 0.507 |
| 32 | 0.809 / 0.464 / 0.499 | 0.907 / 0.517 / 0.512 |
| **40** | 0.765 / 0.498 / **0.504** | 0.942 / 0.697 / **0.666** |
| 48 | 0.891 / 0.536 / 0.514 | 0.847 / 0.621 / 0.595 |
| 56 | 0.752 / 0.555 / 0.473 | 0.873 / 0.588 / 0.543 |

**Strong overfitting everywhere:**
- train accuracy is 0.64–1.00 against validation 0.34–0.75;
- early stopping triggers after 1–10 epochs.

## 2. Does the layer-56 selection explain the PC1 failure? (item 5)

- **Largely, yes.**
  - The DEV analysis selected layer 56 from the **binary unchosen** grid, where every cell was at chance; the selection was effectively noise.
  - PC1 was pre-declared at that layer: 0.399.
  - At the layer that is best on inner-val **for PC1 itself** (layer 40), PC1 reaches **0.752** on DEV-val.
- **But it is still below the 0.80 requirement.** The layer choice explains most, not all, of the failure.

## 3. Effective decoding vs reconstruction (item 3)

c1 at layer 40, on DEV-val:

| Method | Accuracy |
|---|---|
| **Text baseline** (the candidate printed in the action line of the chosen letter) | **1.000** |
| **Choice × arrival keys** (linear, no query activation) | **0.995** |
| Pointer PC1 (query × keys) | 0.752 |

**Reading:**
- c1 is **trivially reconstructible** from the choice plus the keys, or from the text.
- The pointer stays **below** these controls.
- PC1 therefore measures whether the **query–key pointer can learn the choice binding** with the available data, not whether the information exists.
- Passing PC1 is necessary but not sufficient evidence of instrument sensitivity (as stated in the protocol).

## 4. Learning curve and confusions (item 4)

**PC1 at layer 40**, by fraction of inner-train families:

| Training worlds | train | inner-val | DEV-val |
|---|---|---|---|
| 218 (25 %) | 0.945 | 0.389 | 0.394 |
| 421 (50 %) | 0.990 | 0.668 | 0.621 |
| **852 (100 %)** | 0.993 | 0.711 | **0.752** |

- The curve is **still rising steeply** at the full DEV size.
- **For comparison (different task and template):** in H1b X3/X4 the same pointer design reached c1 0.96 with 2,112 training worlds.
- The instrument is **data-limited**.

**Errors of PC1** on DEV-val (95 errors), by the role of the predicted candidate:

| Role | Count |
|---|---|
| **u1** (the other action-line label) | **64** |
| c2 | 9 |
| ue1 | 5 |
| ce1 | 4 |
| others | ≤ 3 each |

- About **two thirds** of the errors confuse c1 with u1. The pointer finds "a depth-1 label" but often fails to bind it to the model's choice.
- This is the same choice-binding weakness seen under low data.

## 5. Observations relevant to H1c (POST HOC; no inference)

- **Chosen continuation (PC2) is decodable at layer 40** (DEV-val 0.666, inner-val 0.697).
- **Unchosen continuation (P) stays at chance at every layer** (0.47–0.51), including layer 40.
- The asymmetry is suggestive, but it is a post hoc observation on one DEV split, with an instrument that does not yet pass its sensitivity requirement. **It must not be read as evidence that the unchosen continuation is absent.**

## 6. Recommendation

| Option | Assessment |
|---|---|
| **Improve the instrument, no new data** | **Limited.** The bottleneck is data (§4). Fixing the PC1 / selection layer independently (e.g. layer 40 chosen from PC1 on inner-val) helps (0.40 → 0.75), but does not reach 0.80. A wider hyper-parameter search would be optimistic and was not authorised |
| **Increase DEV** | **The most informative scientifically.** The learning curve is still steep, and the H1b precedent (about 2,100 worlds → 0.96) suggests that about 2,000–2,500 training worlds could pass PC1. At that point the unchosen-continuation result would become interpretable. **Indicative cost:** about +600 families (3,000 worlds) captured at ANS + 12 arrival keys, layers {40, 48} only, ≈ **$0.70**. BTC cumulative ≈ $3.76 → ≈ $4.46 |
| **Close H1c now** | Defensible. The DEV result is null, the instrument is insufficient, and the remaining budget (≈ $1.24) cannot pay for **both** a larger DEV and a TEST capture under $5 |

**Recommendation (reasoned):**
1. **Increase DEV once, with the instrument fixes pre-declared**, only if the owner is prepared either to raise the $5 limit for an eventual TEST, or to accept that H1c ends as a DEV study. The fixes would be:
   - PC1 layer chosen from PC1 on inner-val;
   - the unchosen-continuation decoder evaluated at that same layer;
   - the robust sign-flip rule unchanged.
2. **If the limit stays fixed at $5, close H1c now**, with these documented conclusions:
   - the task and controls work as designed: bitwise key identity, controls at 0.5 by construction, calibrated rule, competence 0.72;
   - the H1-style pointer is **data-limited** on this task;
   - post hoc, **the chosen continuation is decodable** at layer 40 while the unchosen one is not, which is an open question for a future, better-powered study.

**My preference:** close H1c under the current $5 limit, because a larger DEV without a confirmatory TEST would remain exploratory. Choose "increase DEV" if a budget increase is foreseen.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (existing DEV activations only; no TEST)
[YES] Did not equate token availability with prospective path representation (c1 reconstruction controls reported)
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (CPU only; 0.027 / 1 CPU-h)
[YES] Preserved negative results and raw logs (historical result unchanged; all diagnostics post hoc)
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decision: increase DEV (and budget question) or close H1c.
```
