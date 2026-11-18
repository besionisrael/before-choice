# BTC-V2 study — design note: why did the H1 probe fail, and which experiments can tell the explanations apart?

**Date:** 2026-11-14 · **Status:** DESIGN NOTE ONLY (CPU writing). Nothing here is run, frozen or authorised.

> **Naming.** "BTC-V2" was already used for the model-screening stage (`BTC-V2_DESIGN_REVIEW.md`, `BTC-V2-S1…C1`). To avoid confusion I suggest calling this study **BTC-H1b** or **BTC-V3**. The owner decides.

## 0. Ground rules carried over

**LOCKED-TEST is consumed.**
- It must not be used to choose, tune or compare methods. Not even a "quick look" is allowed.
- Its observations stay as they are (§1) and cannot validate any new hypothesis.

**Any new confirmatory hypothesis must be:**
1. explored on DEV data only (PROBE-DEV, or a new DEV pool);
2. frozen: target, decoder, grid, baselines, thresholds, power;
3. tested once on a **new, independently generated and sealed TEST pool**.

**Every GPU run or new pool** needs its own owner authorisation.

**Out of scope here:** H2 and H3 stay conditional on a positive H1 and are not proposed.

## 1. What is OBSERVED (LOCKED-TEST, confirmatory or descriptive) — fixed, not to be reinterpreted

| Observation | Source | Status |
|---|---|---|
| Probe paired accuracy 0.020 vs chance 0.028; B3 0.195; B2max 0.020; B6 0.020; **H1 NOT MET** | LOCKED-TEST | **confirmatory** |
| S1 pooled depth 3 + 4: probe 0.015 vs B3 0.319. S2 `xswap`: 0.007 vs 0.234 (Holm p = 1.0) | LOCKED-TEST | confirmatory secondary |
| Per-head single-world accuracy (depth-4 base/cf): u1 0.191, u2 0.198, u3 0.205; anchors c1 0.188, **c3 0.363**, u1 0.238, u3 0.274 (chance 0.167) | LOCKED-TEST | descriptive only, not baseline-controlled |
| B7 `cpath` (chosen branch) probe paired accuracy 0.119 | LOCKED-TEST | descriptive reference |
| 21 % of worlds not admissible (forced choice ≠ optimum); 9 exact ties | LOCKED-TEST | descriptive |
| DEV: heavy overfitting (train 0.57–0.93 vs val 0.03–0.31); no fit at the epoch cap | PROBE-DEV | exploratory |

## 2. Candidate explanations (EXPLORATORY HYPOTHESES — none is supported yet)

| ID | Explanation | What it predicts |
|---|---|---|
| **E1 Not computed** | The model solves the choice by shortcuts (backward from the minimum reward, anchor-and-eliminate; P0) and never builds the unchosen depth-2 state | No decoder at any position beats prompt baselines on the unchosen branch. Tasks that *require* the state should change both behaviour and decodability |
| **E2 Decoder limited** | The information is present, but the low-rank bilinear probe on about 2,600 worlds × 5,376 dims cannot learn it (overfitting) | Regularised or reduced probes, or more DEV data, raise DEV-val accuracy above B3. Learning curves still rising |
| **E3 Label-representation mismatch** | The model refers to labels through **contextual** representations (pointer to where the label occurs in the prompt). Static mean-token embeddings of 1–5-token random strings are a poor key | A "pointer" decoder that scores candidates by the hidden state at each label's occurrence in the prompt succeeds where the static one fails |
| **E4 Wrong position** | The state is computed at rule-line or action-line tokens, not at `<channel\|>` | Decoding succeeds at earlier positions and fades at the answer position |
| **E5 Wrong layer sampling** | The signal lies at vectors outside the 17 sampled | Dense layer sweep on DEV finds a peak between sampled indices (weakly plausible, since neighbouring layers are correlated) |
| **E6 Chosen-branch asymmetry** | Only the chosen path is represented, as the c3 / B7 hints suggest | Baseline-controlled chosen-branch decoding succeeds while unchosen-branch decoding stays at baseline |
| **E7 Competence limited** | At depth 4 the model is often not really solving the task, so representations are noisy | Decodability rises with model confidence (\|logp_A − logp_B\|) and at depth 3 / easier worlds |

These explanations are not mutually exclusive.

## 3. Discriminating experiments (all on DEV first; ordered by information gained per cost)

| # | Experiment | Separates | Compute | Needs a new capture? |
|---|---|---|---|---|
| **X1** | **Decoder sensitivity check (positive control) on existing PROBE-DEV activations.** Decode variables that must be present (the model's own chosen letter; a label printed in the action line) with the same probe family, including prompt baselines | Whether the probe pipeline can detect anything beyond B3 at all (E2 / E3 vs E1) | CPU, existing data | no |
| **X2** | **Learning curves and capacity variants** on PROBE-DEV: PCA-reduced z, stronger regularisation, linear-softmax and small-MLP heads, under the same split | E2 | CPU | no |
| **X3** | **Pointer (contextual-label) decoder:** candidate key = hidden state at the label's prompt occurrence | E3 | **GPU** (positions at label tokens) | yes (DEV) |
| **X4** | **Position sweep:** end of each rule line, action lines, `<channel\|>`, at a denser layer grid | E4, E5 | **GPU** | yes (DEV) |
| **X5** | **Baseline-controlled chosen vs unchosen comparison:** same decoders and baselines for c2 and u2 | E6 | CPU (on X3/X4 data) | with X3/X4 |
| **X6** | **Task variant that *requires* the unchosen state:** for example the reward depends on an intermediate state, or the terminal rewards are hidden behind intermediate labels, so that the P0 shortcuts are removed. Check competence first | E1 vs others (behaviour + decodability) | CPU generator + **GPU** competence/capture | yes |
| **X7** | **Confidence-stratified analysis** (logp margin, depth) | E7 | CPU on DEV | no |

**Suggested first step (lowest cost):** X1 + X2 + X7 on existing PROBE-DEV activations, CPU only, with no new data.
- If the pipeline cannot detect even a "must-be-present" variable beyond the prompt baseline, the decoder or label representation is the bottleneck (E2/E3). Then X3 is the priority.
- If it can, E1 or E4 become more likely, and X4 or X6 come next.

**Use of PROBE-DEV:** PROBE-DEV was already used for selection. Re-using it for exploration is legitimate, but any method chosen on it **must** be validated on a new DEV-val split or pool, and then on a new sealed TEST pool.

## 4. Requirements for any future confirmatory test

- **New pool:** new seed; independent families, labels and prompt hashes; same independence tests as D5.
- **Freeze before generation or sealing:** target, decoder, positions, layers, baselines (the B3-type prompt baseline stays mandatory), the IUT rule, INSUFFICIENT thresholds, a power analysis given B3 ≈ 0.2–0.3, and secondaries with statuses.
- **Same integrity machinery:** stage-scoped access, single-evaluation marker, append-only logs.
- **Budget estimates for a decision:**
  - X1/X2/X7: about 2–5 CPU-h.
  - X3/X4 DEV capture: about 1 GPU-h at most (≈ $1–2 per capture of 4,800 worlds with extra positions; storage grows with the number of positions).
  - New TEST capture: ≈ $0.75.

## 5. Decision requested from the owner

1. Name of the follow-up study (BTC-H1b / BTC-V3 / other).
2. Whether to authorise the CPU-only exploratory block **X1 + X2 + X7** on PROBE-DEV, labelled EXPLORATORY, with no TEST access.
3. Whether to plan GPU-dependent work (X3, X4, X6) after X1/X2 results.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (none proposed)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (LOCKED-TEST consumed; not used)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (nothing run)
[YES] Preserved negative results and raw logs
[YES] Labeled unrun stages NOT RUN and limitations honestly (all items are proposals)
Any deviation: none.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decision on the follow-up design.
```
