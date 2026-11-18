# BTC-H1c — final diagnostic on existing DEV data (POST HOC / EXPLORATORY)

**Date:** 2026-11-14 · **Owner decision:** "DÉCISION SCIENTIFIQUE — BTC-H1c, DERNIER DIAGNOSTIC" (option c; CPU only, 1 CPU-h cap, 4 threads) · **Status:** completed, **STOPPED.**

> **Status of these results:** all **POST HOC / EXPLORATORY**. Every earlier result is kept unchanged: H1c DEV (inconclusive), the instrument diagnostic and DEVX (INSTRUMENT ADEQUATE; late unchosen 0.521).
> **What was used:** existing activations only. No GPU, no new pool, no TEST, no hyper-parameter search, no change to any historical rule.

**Run:**
- Script: `src/h1c_final_diagnostic.py`, committed before the run (git `5e38894`).
- Output: `outputs/h1c_devx/final_diagnostic/FINAL_DIAGNOSTIC.json` (sha `0e8c24b2…`).
- Compute: **0.016 CPU-h of 1.**

**Settings:**
- Same frozen roles: TRAIN 2,551 / SELECT 364 / EVAL 711 admissible worlds. All numbers are on **EVAL** (never-seen families).
- Same layer: **40**.
- Same decoder family: pointer r = 8, λ = 0.1, seeds 0–2, early stopping on SELECT. Linear controls use L-BFGS with λ = 0.01, as in the DEV analysis.
- Same robust statistic: per family, mean correctness over the worlds of complete pairs, minus 0.5; one-sided sign-flip test; family-bootstrap 95 % CI.

## 0. Key identity within pairs (measured on the captured vectors, layer 40)

| Pairs | Binary arrival keys bitwise identical between members |
|---|---|
| late unchosen (base_L / dswap_L) | **141 / 141** |
| early unchosen (base_E / dswap_E) | **0 / 118**, as expected: the departure lines precede the arrivals |
| cswap chosen (base_L / cswap_L) | **26 / 141**. The generator only enforces late order for the **unchosen** branch, so chosen keys are identical only when the chosen arrivals happen to precede the chosen swapped lines |

## 1. Early pairs: is the 0.703 unchosen signal decodable without the query? (item 1)

| Decoder (unchosen continuation, early pairs; 118 families, 236 worlds) | Accuracy | 95 % CI | Sign-flip p |
|---|---|---|---|
| **Pointer P** (ANS query × arrival keys) | **0.703** | [0.648, 0.758] | < 1e-4 |
| key-only (arrival keys, no query, no choice) | 0.445 | [0.381, 0.508] | 0.97 |
| choice × keys (linear) | 0.487 | [0.432, 0.547] | 0.61 |
| μ_choice query × keys (capacity-matched bilinear) | 0.517 | [0.462, 0.572] | 0.23 |
| **P − best control** (per family) | **+0.186** | [+0.110, +0.263] | < 1e-4 (two-sided) |

**Reading:**
- In early pairs the keys **differ** between members, yet **no query-free control decodes the continuation**.
- The early-pair signal therefore **requires the pre-answer query**, but in a condition where the keys also carry pair-specific content.
- This shows **decodable information arising from the query–key interaction**. It does **not** isolate information carried by the query alone: in the late pairs, where the keys are identical, the same pointer stays at 0.52.

**Alternative explanations, not tested:**
- **(i) Training mix.** The query-free controls were trained on the full mix of variants, where keys are mostly uninformative. They might decode early-pair keys if trained on early worlds only.
- **(ii) Query as a selector.** The query may carry a generic "continuation feature" that becomes usable only when the keys hold the matching structural information.

## 2. cswap pairs: chosen continuation under conditions as comparable as possible to the late pairs (item 2)

**All cswap pairs** (141 families, 282 worlds):

| Decoder | Accuracy | 95 % CI | p |
|---|---|---|---|
| **chosen pointer (PC2)** | **0.770** | [0.720, 0.819] | < 1e-4 |
| key-only | 0.514 | — | 0.34 |
| choice × keys | 0.504 | — | 0.51 |
| μ_choice × keys | 0.496 | — | 0.50 |

**Subset with identical chosen keys** (the late-comparable condition; 26 families, 52 worlds):

| Decoder | Accuracy | 95 % CI | p |
|---|---|---|---|
| **chosen pointer (PC2)** | **0.712** | [0.596, 0.827] | 2e-4 |
| key-only, choice × keys, μ_choice × keys | **0.500 exactly** (identical inputs) | — | 1.0 |

**Reading:** when the chosen candidates' keys are identical across members, so that only the query can tell which chosen state continues, the pointer still decodes the **chosen** continuation (0.71, from a small sample of 26 families).

## 3. Branch comparison (item 3)

| Population (EVAL) | Families | Accuracy [95 % CI] | p |
|---|---|---|---|
| **chosen**, cswap identical-key subset | 26 | 0.712 [0.596, 0.827] | 2e-4 |
| **unchosen**, late pairs (keys identical by construction) | 141 | 0.521 [0.482, 0.557] | 0.18 |
| **chosen − unchosen**, per family, families with both | 25 | **+0.26** [+0.12, +0.40] | 0.002 (two-sided) |

**Why this is not a demonstrated asymmetry:**
1. **Different populations and admissibility.**
   - A cswap pair needs `base_L` and `cswap_L` both admissible; a late pair needs `base_L` and `dswap_L`.
   - The identical-key chosen subset (26 / 141) is selected by a line-order property that the generator did not control, and it may correlate with other features.
2. **Small sample:** 26 families on the chosen side; the CI is wide.
3. **Post hoc:** neither the subset nor the comparison was pre-declared.
4. **Decoder target asymmetry:** the chosen decoder is a different head (PC2), trained with the same recipe on a different target.

## 4. Interpretation (item 4): three kinds of information

| Kind | Chosen continuation | Unchosen continuation |
|---|---|---|
| **Decodable** (some decoder succeeds) | yes (0.77 on all cswap pairs) | yes on early pairs (0.70); **no on late pairs** (0.52) |
| **Available in the keys alone** (query-free controls) | not shown (controls at chance) | not shown (controls at chance, even on early pairs where the keys differ) |
| **Specific to the pre-answer query** (keys identical within the pair) | **observed** on 26 families (0.71) | **not detected** (0.52, 141 families) |

- **None of this is evidence of a forward sweep.** Decodability says nothing about the order or direction of construction: backward inference from the rewards remains possible, nothing here is sequential, and there is no causal test.
- **The most parsimonious hypothesis (not established):** the pre-answer state carries usable information about the **chosen** continuation. For the **unchosen** continuation, usable information appears only when the label representations themselves are informative, and it was **not detected in the query alone**.

## 5. Final recommendation

**Close H1c on its original question,** with the following documented conclusion:
- with an instrument shown to be sensitive (PC1 0.89, chosen continuation 0.76 on never-seen families);
- and with controls that are exact by construction;
- the pre-answer query did **not** show information about the unchosen continuation on late pairs (0.521, p = 0.18, CI up to 0.557).

A confirmatory TEST of that hypothesis is **not recommended**: if the effect is near the DEV estimate (δ ≈ 0.02), its power would be about 0.35.

**Possible future study (not proposed for execution here; would need its own design and decision):**
- Pre-register the **chosen-vs-unchosen asymmetry** under strictly comparable conditions.
- This requires a generator change that enforces late order for **both** branches, so that the chosen keys are identical within cswap pairs by construction.
- Then a new sealed TEST. Budget: BTC spend ≈ $4.24, cap $6.
- This is the most informative follow-up suggested by these diagnostics, but it is a **new hypothesis**, generated post hoc on DEV.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (frozen roles; EVAL families; no TEST)
[YES] Did not equate token availability with prospective path representation (keys / query / decodable distinguished)
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (CPU only; 0.016 / 1 CPU-h)
[YES] Preserved negative results and raw logs (all historical results unchanged)
[YES] Labeled unrun stages NOT RUN and limitations honestly (post hoc status, small chosen subset, population differences)
Any deviation: none.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decision: close H1c (recommended) or design a new asymmetry study.
```
