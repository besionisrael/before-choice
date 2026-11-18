# BTC-H1c DEV — results (EXPLORATORY): capture passed all gates; unchosen-continuation decoding at chance; positive control PC1 failed

**Date:** 2026-11-14 · **Owner decision:** "DÉCISION SCIENTIFIQUE — BTC-H1c-DEV". Conditional GO: blocking statistical verification first, then one DEV capture (cap $0.90), then a CPU analysis if G1–G3 pass. **Status:** completed, **STOPPED.**

> **Status of these results:** DEV only, exploratory. No TEST was created or captured and nothing confirmatory was run.
> **Unchanged:** BTC-H1b v1.0 untouched; H1 v0.3 NOT MET.

## 1. Blocking statistical verification (done before the GPU)

| Owner item | Result |
|---|---|
| 1. Null with perfectly anti-correlated predictions | Simulated (`power_h1c_robust.json`, 5,000 datasets): **the old paired rule (P_pair − 0.25 > 0) rejects in 100 % of null datasets.** Each pair is then fully right or fully wrong, so the null paired success is 0.5 |
| 2. Calibration of the 0.25 threshold | **Not valid under dependent predictions** (item 1). It was valid only for independent or positively correlated preferences (0.020 at the boundary) |
| 3. Robust rule and power | **Amendment A1, frozen before the capture** (`BTC-H1c_PROTOCOL_DRAFT.md` §8, git `51cc386`). Single-world binary accuracy over late-pair worlds vs 0.5, with a one-sided **sign-flip randomisation test on family means**. Calibrated: 0.0246 (independent), **0.0224 (anti-correlated)**, 0.0222 (anti-correlated at family level), 0 (identical). Power at 600 families: 0.83–0.93 for δ = 0.05, ≥ 0.99 for δ = 0.075 |
| 4. Orientation randomisation of the generator | `orientation_checks.json`: which label continues, the order of the swapped lines, the order of the arrival lines, label token length (ties excluded) and the optimal letter are all ≈ 0.5 and mutually independent (minimum p 0.065 over ≈ 15 tests) |
| 5. Documentation and freeze | Amendment A1, the audit and the tests were committed and hashed (`PREPARED.json → amendment_A1_frozen_before_capture`) **before** the capture |

## 2. Capture (single run)

| Item | Value |
|---|---|
| Pre-launch | frozen files unchanged; inputs `ca2133ca…`; positions `0d29b278…`; model `842da379…`; no earlier capture; BTC spend before the run $3.243 |
| App | `ap-TNhxsdOiueWYQbINPIvxZI`, 18:43 → 18:54 UTC |
| Timing | load 248.8 s, capture 335.8 s (2,000 worlds + 20 replays), remote 584.6 s (guard 800 s); download 69 s |
| Size | 3.1 GB; peak GPU memory 63.5 GB; watchdog not triggered |
| **G1** integrity | hashes and shapes OK, 2,000/2,000, verified before Volume deletion |
| Fidelity | max\|Δ\| 0.195 (pre and post) ≤ 0.25; 0 non-finite; **20/20 replays bitwise identical**; 24 divergence entries = 12 worlds with an exact original A/B tie (excluded as not admissible) |
| **G2** blocking bitwise identity of the binary arrival keys | **400 / 400 late pairs identical**, so PASS |
| **G3** competence | overall **0.723** (threshold 0.70); late-pair worlds **0.728** (threshold 0.60), so PASS. By variant: base_L 0.720, dswap_L 0.735, base_E 0.710, dswap_E 0.710, cswap_L 0.740 |
| **Cost** (billing report) | **$0.519** (A100 0.422, CPU 0.032, memory 0.022, network 0.043); BTC cumulative ≈ **$3.76** (< $5) |

## 3. DEV analysis (single run; 0.015 CPU-h of 3; 4 threads)

**Splits:**
- inner-train 852 worlds and inner-val 211 (both drawn from the 300 DEV-train families);
- DEV-val 383 admissible worlds from 100 distinct families (73 contributing late pairs).

**Selection:** on inner-val, over layer {24, 32, 40, 48, 56} × rank {2, 8}.
- All 10 cells fall between 0.464 and 0.555 inner-val binary accuracy, i.e. at chance.
- Selected: layer 56, rank 8 (0.555).

**Binary unchosen continuation, DEV-val:**

| Decoder | Single (all worlds) | **Robust primary: late-pair worlds, accuracy (sign-flip p)** | Paired late (descriptive) |
|---|---|---|---|
| **P** (ANS × arrival keys) | 0.473 | **0.479 (p = 0.75)** | 0.041 |
| P on early pairs | — | 0.459 (p = 0.86) | 0.108 |
| key-only | 0.499 | 0.500 exactly (p = 1) | 0.000 |
| C_lin (choice × keys) | 0.501 | 0.500 exactly | 0.000 |
| C_bil (μ_choice query) | 0.507 | 0.500 exactly | 0.000 |
| P_perm | — | 0.488 (p = 0.95) | 0.221 |
| P − P_perm | — | 0.491 (p = 0.65) | — |

Chance is 0.5 for a single world and 0.25 paired.

**Positive controls (pre-declared):**

| ID | Value | Requirement | Outcome |
|---|---|---|---|
| **PC1**: 12-way c1 pointer at the selected layer (56) | **0.399** | ≥ 0.80 | **FAILED: instrument insufficiently sensitive** |
| PC2: binary chosen continuation | 0.543 | — | near chance |
| PC3: chosen letter, linear on ANS | **0.971** | — | high |

**Reading (exploratory):**

1. **No evidence of unchosen-continuation information** in the pre-answer query on DEV, under the robust rule. The controls behave exactly as designed (0.5 by construction) and the integrity gate holds.
2. **The instrument did not pass its own sensitivity requirement (PC1 0.40 < 0.80).** By the pre-declared rule, the DEV result is therefore **not informative** about the presence of the information, and **no TEST is proposed**.
3. **The chosen continuation (PC2) is also near chance, while the choice itself (PC3) is decodable.** This suggests the binary decoders are limited by data and instrument (852 training worlds for a 5,376-dimensional bilinear decoder, and a selection that was noise among chance-level cells), rather than showing an asymmetry between branches.
   - For comparison: in H1b X3/X4 (PROBE-DEV, 2,112 training worlds, H1 template), c1 reached 0.96 at layers 40–48 with arrival keys.
   - Here PC1 is evaluated at the **selected** layer 56, as pre-declared, where the binary selection landed by chance.

## 4. Limits

- **Sample size:** only 300 DEV-train families (about 1,060 admissible training worlds) for the decoders; the selection grid was entirely at chance.
- **The pre-declared PC1 location** (the binary-selected layer) put the sensitivity check at a layer chosen by noise. PC1 at other layers was not computed in this run, and computing it now would be post hoc.
- **Competence** just above threshold (0.72): about 27 % of worlds are excluded as not admissible.
- **Template and task differ** from H1/H1b (dead ends, "you can move to"); results are not directly comparable.

## 5. Status and options (no action taken)

- **The H1c rule is not met on DEV, and the instrument is insufficient (PC1 failed). By the conditions of this decision: stop; no TEST.**
- **Possible future directions** (each would need a new decision and budget; BTC cumulative ≈ $3.76, about $1.24 left under $5):
  - a larger H1c DEV pool;
  - selection of the PC1 layer independent of the binary selection;
  - fewer layers and more data.
- None of these is proposed for execution here.

## 6. Artifacts

| Artifact | sha (prefix) |
|---|---|
| `outputs/h1c_dev/capture/MANIFEST.json` | `930c8685…` |
| `GATES.json` | `14c2b4ad…` |
| `run_manifest.json` | `e4ac4d18…` |
| `outputs/h1c_dev/analysis/DEV_ANALYSIS.json` | `4e08b736…` |

The 3.1 GB of tensors are not versioned. Logs are in `outputs/logs/h1c_dev_*`.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task (controlled extension)
[YES] Kept H1 primary and H2/H3 conditional
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (DEV only; family-disjoint selection; no TEST)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (BTC-H1c-DEV; $0.519 ≤ $0.90 cap; cumulative ≈ $3.76)
[YES] Preserved negative results and raw logs (DEV null, PC1 failure, old-rule invalidity)
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decision on the future of H1c (no TEST under the current rule).
```
