# BTC-H1c — revised protocol draft (C-2 revised; CPU only)

**Date:** 2026-11-14 · **Owner decision:** "DÉCISION SCIENTIFIQUE — BTC-H1c". O3, C-1, C-3, C-4 and C-5 validated; C-2 to be revised before any capture.

**Status:** DRAFT for scientific validation.

> **Not done:** no GPU, no new TEST, no probe training.
> **Kept as is:** BTC-H1b v1.0 stays an untouched archive; the cumulative limit stays at $5.

## 0. Design change since the proposal (revision 2 of the task)

**The dead-end side chain now has the same length as the route:** d1 → e1 → e2 → e3, where e3 has no move and no reward. This gives 12 candidates.

**Why:**
- Revision 1 (length 2) removed an occurrence-count cue at depth 2.
- But it left one at depth 3: u3 had an outgoing move, while ue2 did not.
- Now every depth-2 and depth-3 label appears exactly twice in the rule lines, and T and e3 once each. The reward-free graph below u1 is **symmetric**.
- **Only the reward lines identify the continuing branch.**

## 1. Verifications (owner items 1, 3, 6, 7)

Sources:
- `src/h1c_token_checks.py` (Gemma tokenizer, 200 late pairs) → `outputs/h1c_design/token_checks.json`;
- `src/h1c_design_checks.py` (400 families, 2,000 worlds) → `outputs/h1c_design/checks.json`.

| Check | Result |
|---|---|
| Choice, rewards, chosen-branch text, c2, the 12 candidates, depth-2 set, branch membership and occurrence counts identical within each pair; u2 changes; 2 lines differ | 400/400 (late and early) |
| **Item 1 — the binary candidates' arrival keys are computed before the first differing token** | **200/200** |
| They occupy the same token index in both members | 200/200 |
| Token ids identical up to the last of them (so identical contextual keys: causal attention, deterministic forward) | 200/200 |
| All 12 candidates, arrival before / after the first difference | 1,310 before / 1,090 after. Chosen-branch and depth-3+ labels may come after; **only the two binary candidates enter the primary rule** |
| Total token count equal within pairs | 200/200 (the swap exchanges two labels) |
| **Item 7 — reward-free symmetry:** the descendant chains of the two binary candidates are isomorphic once the reward lines are ignored | **200/200** |
| **Item 3 — prompt control ladder:** 1-hop view, occurrence counts and reward-free k-hop signatures identical across members, so any deterministic function of them gives paired 0 | 200/200 (and 800/800 in `checks.json`) |
| **Item 6 — full-structure control** (follows the rules to the reward lines, no activations) | **exact: 2,000/2,000 worlds; 200/200 pairs** |
| Tests | `tests/test_h1c_design.py` 4/4; `tests_gemma/test_h1c_tokens.py` 1/1 |

**Bitwise determinism of identical prefixes:**
- Earlier captures showed 20/20 replays bitwise identical, and ANS 3,480/3,480 identical to D3-DEV-R1.
- The H1c capture would verify it directly, as a pre-registered integrity check: the binary candidates' arrival keys must be bitwise identical within each late pair.

## 2. Revised primary analysis (C-2′; owner item 2)

**Population:** admissible `(base_L, dswap_L)` pairs. Both members must have the forced choice equal to the oracle optimum, with no exact tie.

**Binary candidates:** the **two successors of the unchosen depth-1 label**.
- They are found from the prompt: the target of the action line for the letter **not chosen by the model**, then its two successors.
- They are the same two labels in both members.
- Only which of them continues to the reward differs.

**Decoder P (pointer):**
- The score of each binary candidate is s(c) = z_ANSᵀ U Vᵀ k_c, with k_c its **arrival key**, computed before the modified lines.
- A softmax over the 2 gives the prediction.
- The layer and rank are fixed on the H1c DEV pool only.

**Pair success:** both members correct. Because the correct label flips between members, P must **change its answer** while its two keys stay identical.

**Controls inside the same binary space:**

| Control | Inputs | Paired accuracy on late pairs |
|---|---|---|
| **C_max** = best of {choice + keys (bilinear-matched, linear)} | choice, the two arrival keys | **0 by construction** (identical inputs, so identical output) |
| key-only | the two arrival keys | 0 by construction |
| prompt-only ladder (1-hop, counts, reward-free graph) | prompt features | 0 by construction |
| **P_perm** | P with the ANS query of another family (same choice, same variant), K = 50 | expected ≈ 0.25 |
| independent guessing | — | **0.25** |

**Rule:**
- In every family-bootstrap resample: Δ = P − max(0.25, C_max, P_perm), using paired accuracy.
- Family-cluster percentile bootstrap, 10,000 resamples, seed 0.
- **H1c MET iff the 2.5 % percentile > 0.**

**Interpretation (bounded):**
- **MET:** the pre-answer activation carries information identifying the **unchosen** continuation, under conditions where c2, the chosen branch, 1-hop structure, label frequencies, the reward-free graph and the candidates' own arrival keys are identical across the compared worlds.
- **NOT MET:** no demonstration.
- **Neither outcome is a forward sweep:**
  - backward inference from the rewards is possible;
  - integration happens somewhere between the modified lines and the answer, and its direction and sequence are not measured.

## 3. Secondary analyses (owner items 4 and 7)

| ID | Analysis | Purpose |
|---|---|---|
| S1 | **12-way global decoding** of u2 (all arrival keys) on the same pairs; chance 1/12 per world | continuity with H1/H1b; not decisive |
| S2 | **pre-reward vs post-reward query position:** the binary decoder with its query at the end of the last modified line, at the end of the last rule line (**before** the reward lines), at the reward-line ends, at Q_END and at ANS | item 7: whether the advantage appears only once the reward lines are read. By the reward-free symmetry, pre-reward positions are expected at chance. Measured, with no causal interpretation |
| S3 | early pairs (`*_E`): the same contrasts when the departure lines precede the arrivals (keys *can* then carry the continuation) | how much the keys alone can carry when causality allows |
| S4 | `cswap_L` reference: binary decoding of the **chosen** continuation | symmetry between chosen and unchosen |
| S5 | the full-structure control and the "modified lines only" control (chance) | prompt reconstructibility ceiling and floor |

## 4. Power for C-2′ (`src/power_h1c_binary.py` → `outputs/metrics/power_h1c_binary.json`)

**Simulation:** the exact rule (max of 0.25, C_max and P_perm inside each resample); C_max = 0 by construction; P_perm as noisy expectations around 0.25; 2 late pairs per family; ICC 0.15; B = 1,000; 400 datasets per cell.

| Families | a = 0.60 (δ = 0 / 0.05 / 0.10) | a = 0.75 | a = 0.90 |
|---|---|---|---|
| 400 | 0.025 / 0.40 / 0.93 | 0.018 / 0.63 / 0.99 | 0.007 / 0.73 / 1.00 |
| **600** | **0.022 / 0.58 / 0.985** | **0.015 / 0.78 / 1.00** | **0.020 / 0.88 / 1.00** |
| 800 | 0.020 / 0.78 / 0.998 | 0.033 / 0.89 / 1.00 | 0.015 / 0.95 / 1.00 |

- **The type-I error stays ≤ 0.033.**
- 600 TEST families (C-3) give **≥ 0.985 for δ ≥ 0.10**. For δ = 0.05, power is 0.58–0.88 depending on admissibility, which is measured at the DEV competence gate (C-4).

## 5. Competence, sizes and budget (validated decisions, updated slots)

- **Competence gate (C-4):**
  - DEV per-world choice accuracy ≥ 0.70 overall, and ≥ 0.60 on late pairs;
  - **and** on DEV the binary candidates' arrival keys must be bitwise identical within each late pair (integrity).
  - Otherwise stop.
- **Sizes (C-3):** DEV 400 families × 5 variants (2,000 worlds); TEST 600 families × 2 late pairs (2,400 worlds).
- **Capture slots** (located per world from tokenizer offsets): ANS, 12 arrival keys, 14 rule-line ends, 2 reward-line ends, Q_END = 30 slots.
  - DEV: 5 layers, ≈ 3.2 GB.
  - TEST: 3 layers (around the DEV-chosen layer), ≈ 2.3 GB.

| Budget | Expected | Cap (timeout 850 s) |
|---|---|---|
| H1c-DEV capture | ≈ $0.65 | ≈ $0.82 |
| H1c-TEST capture | ≈ $0.70 | ≈ $0.82 |
| **BTC cumulative** | $3.24 → **≈ $4.59** | worst ≈ **$4.88** (≤ $5) |

No second full attempt is possible for either capture.

## 6. Limits

1. **Backward inference** (P0): not excluded by any decodability result.
2. **Structural summary vs prospection:** a positive result could reflect reward-linked structure computed while reading, available at the end of the prompt. S2 describes when it appears; S4 checks symmetry.
3. **Template change** ("you can move to", dead-end instruction): not comparable to H1/H1b.
4. **Competence** with dead ends is unknown. The gate may stop the study.
5. **DEV optimism:** the layer and rank are chosen on H1c DEV. The decisive controls are zero **by construction**, not estimated, which limits the risk; the 0.25 floor and P_perm remain.

## 7. Decisions requested

| ID | Proposal |
|---|---|
| C-2′ | binary primary rule on late pairs (arrival keys before the modified lines), Δ = P − max(0.25, C_max, P_perm) > 0 |
| C-6 | task revision 2 (dead-end chain of length 3, 12 candidates) |
| C-7 | secondaries S1–S5, with S2 (pre- vs post-reward positions) as the item-7 check |
| C-8 | integrity gate: bitwise identity of the binary arrival keys within DEV late pairs |

After validation, the next CPU steps would be:
- the positions module for the new template, and the DEV pool generator (seed, sealing rules);
- the DEV capture script and its tests;
- then a GPU request for H1c-DEV.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task (controlled extension)
[YES] Kept H1 primary and H2/H3 conditional
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently (no training)
[YES] Preserved oracle isolation and train/val/test independence (design seeds only; no pool)
[YES] Did not equate token availability with prospective path representation (reward-free symmetry; prompt ladder)
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (CPU only)
[YES] Preserved negative results and raw logs (revision 1 residual cue documented)
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decisions C-2′, C-6, C-7, C-8.
```

## 8. Amendment A1 (owner decision "BTC-H1c-DEV"): robust primary rule, frozen before the DEV capture

**Problem confirmed by simulation** (`src/power_h1c_robust.py` → `outputs/metrics/power_h1c_robust.json`; 5,000 datasets per null; 600 families):
- A decoder **without** continuation information can still give anti-correlated predictions across the two members of a late pair, for example from a line-order cue.
- Each pair is then either fully right or fully wrong, so the null paired success is 0.5.
- **The old paired rule (P_pair − 0.25 > 0) rejects in 100 % of such null datasets.** The 0.25 threshold is not valid under dependent predictions.

**Robust rule (replaces C-2′'s threshold):**
- Statistic: per family, the mean **single-world** binary accuracy over the worlds of its admissible late pairs, minus 0.5.
- Test: a one-sided **sign-flip randomisation test** on the family means, α = 0.025, with 10,000 flips. A family bootstrap CI is reported.
- **Why it is valid:** the generator randomises orientation, so under H0 flipping a family's orientation maps each member's correctness c to 1 − c, and the family statistic is symmetric about 0.
- Any decoder whose prediction is identical in both members (key-only, choice + keys, 1-hop prompt controls) gives exactly 0.
- **Secondary contrast:** P − P_perm, using a sign-flip on family means of the correctness differences.
- **Paired accuracy:** reported descriptively only.

**Calibration (5,000 datasets each):**

| Null scenario | Robust rule | Old paired rule |
|---|---|---|
| independent preferences | **0.0246** ± 0.0022 | 0.0228 |
| **anti-correlated predictions** (pair fully right or fully wrong) | **0.0224** ± 0.0021 | **1.000** |
| anti-correlated, with a family-level cue | **0.0222** ± 0.0021 | **1.000** |
| identical predictions in a pair | 0.000 | 0.000 |

**Power** of the robust rule (600 families × 2 late pairs, admissibility 0.75, single-world δ):

| Within-pair correlation of correctness | δ = 0.025 | δ = 0.05 | δ = 0.075 | δ = 0.10 |
|---|---|---|---|---|
| 0 | 0.39 | **0.93** | 1.00 | 1.00 |
| 0.5 | 0.33 | **0.83** | 0.99 | 1.00 |

**Orientation audit** (`src/h1c_orientation_checks.py` → `outputs/h1c_design/orientation_checks.json`; 400 DEV families). Features a decoder without path information could exploit:

| Feature | Rate | Binomial p |
|---|---|---|
| continuing label is canonically first | 0.503 | 0.96 |
| its swapped line comes first | 0.495 | 0.88 |
| its arrival line comes first | 0.513 | 0.65 |
| it is the longer label (ties excluded, n = 232) | 0.534 | 0.32 |
| optimal action is A | 0.500 | 1.00 |

- All pairwise independence tests have p ≥ 0.065 (≈ 15 tests).
- The same pattern holds in both splits.
- A first version of the audit counted equal token lengths as "not longer" (rate 0.31). This was an artefact of 168 ties, corrected.

**Implementation:**
- `h1c_dev_analysis.signflip` reports the robust primary statistic for P and every control, early pairs, and P − P_perm.
- Tests: `tests_cpucheck/test_h1c_dev.py` (planted signal: accuracy > 0.6, p < 0.025; identical-prediction controls exactly 0.5).
