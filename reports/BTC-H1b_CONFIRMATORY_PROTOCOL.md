# BTC-H1b — confirmatory protocol v1.0 (freeze prepared)

**Date:** 2026-11-14 · **Owner decisions:**
- "BTC-H1b v1.0": D-2′, D-3 and D-8 validated; D-5′ = 1,000 families, INSUFFICIENT below 875 contributing families.
- Earlier: D-1, D-4, D-6, D-7.

**Status:**
- **Frozen artifacts prepared and hashed** (`outputs/h1b_freeze/V1_MANIFEST.json`).
- **LOCKED-TEST-H1b not generated; no GPU capture; no evaluation.**
- The H1 v0.3 verdict NOT MET is unchanged.

> **Mandatory interpretation (owner):**
> - Even if H1b is MET, this will establish **additional exploitable information in the pre-answer query**, not necessarily an independent representation of the unchosen branch. **The information could still come from the chosen branch.**
> - NOT MET means that the advantage is not demonstrated; it is not evidence of absence.
> - Neither outcome establishes nor refutes a forward sweep.

## 1. Question and hypothesis

**Question:** does the pre-answer activation contain u2 information beyond what can be rebuilt from the model's chosen letter and the contextual label representations?

**H1b:** on LOCKED-TEST-H1b, the pointer P has a higher paired u2 accuracy than **both**:
- **(a)** C_max = max(C_bil, C_lin), taken inside every bootstrap resample, and
- **(b)** P_perm, the pointer with its query permuted between different families with the same model choice and variant.

## 2. Optimism risk

- PROBE-DEV has served for H1 v0.3 selection, H1b X1/X2/X7, X3/X4, the control-C revision and this freeze (≈ 3,500 fits).
- The primary configuration was chosen after seeing DEV-val, so the DEV numbers are optimistic.
- **No DEV number enters the decision.**

## 3. Frozen configuration and decoders (`src/h1b_freeze_dev.py`; run once, 0.064 CPU-h)

**Configuration:**
- query: ANS (`<channel|>`);
- keys: contextual **arrival** keys (last token of each candidate's arrival occurrence);
- **layer 40**;
- bilinear r = 8, λ = 0.1, **seeds 0, 1, 2**.

**Training:**
- E is selected by early stopping on DEV-val after fitting on DEV-train.
- Each decoder is then refit on all 3,480 PROBE-DEV training-rule worlds with fixed E (3-seed average).
- L-BFGS decoders are refit with their inner-val λ.
- **Standardisation:** z, keys and μ_choice from DEV-train worlds (`STANDARDISATION.npz`).

| Frozen decoder | Role | Notes |
|---|---|---|
| `P_u2_L40` | tested decoder | |
| `C_bil_u2_L40` | control (a), part 1 | same U and V shapes; query = μ_choice |
| `C_lin_u2_L40` | control (a), part 2 | **competitive variant = L-BFGS, λ = 0.01** (inner-val 0.727 vs AdamW 0.725) |
| `P_c1_L40` | elementary sensitivity check | |
| `P_c2_L40` | elimination control (D-8, secondary) | |
| `K_u2_L40` | key-only (λ = 1), secondary | |
| `B3plus_u2` | prompt + choice (λ = 1e-4), secondary | |
| `P_u2_L32`, `P_u2_L48` | descriptive | |

The AdamW variant `C_lin_adamw_u2_L40` is saved and reported, but not used in the decision.

## 4. DEV report for the controls (EXPLORATORY; H1b inner/outer splits; DEV-val reported only)

| u2, DEV-val | single | paired (77 pairs) |
|---|---|---|
| P (pointer) | 0.815 | 0.688 |
| C_bil (capacity-matched) | 0.502 | 0.403 |
| C_lin AdamW | 0.717 | 0.584 |
| **C_lin L-BFGS** (frozen; λ = 0.01 chosen on inner-val) | 0.723 | 0.571 |
| P_perm (between-family) | 0.298 | 0.075 |
| **Elimination via c2 (D-8)** | **0.883** | **0.818** |

- P_c2 on DEV-val: 0.788 for c2.
- The elimination control uses the prompt-derived depth-2 set (2 candidates) and P_c2's choice between them. Its floor is 0.5 for a single world and 0.25 for a pair.

### 4b. Key DEV finding for interpretation (EXPLORATORY)

**The elimination control beats P on DEV** (paired 0.818 vs 0.688).
- On DEV, everything P decodes about u2 can be explained by **chosen-branch information** (P_c2's c2) combined with the prompt's depth-2 structure.
- This is exactly the caveat the owner required. **A MET outcome on the frozen primary contrasts would most likely reflect chosen-branch information, not an independent unchosen-branch representation.**
- The protocol keeps D-8 as **secondary, non-confirmatory**, as decided.
- I flag one option for the owner before the pool is generated (not applied): add **P − elimination > 0** as a third intersection-union contrast, or make it the primary. This is the contrast that addresses the unchosen-branch question.
- Given the DEV numbers, that contrast is **unlikely to be met**. Running it would still turn the current interpretive caveat into a test.

## 5. Population, estimator, decision rule (frozen; `src/sealed_stage/h1b_eval_test.py`)

**Population:** admissible depth-4 `upath` pairs (base→upath_base, cf→upath_cf).
- Admissible = forced choice (logp_A vs logp_B from the capture) equals the oracle optimum.
- Exact ties are excluded.

**Estimator:** paired success = both worlds' u2 argmax correct. Ties are broken by canonical label order. P_perm uses its expectation over K = 50 draws.

**Bootstrap:** family-cluster percentile, **10,000 resamples, seed 0**, shared by both contrasts.
- In every resample: Δ_C = P − max(C_bil, C_lin), and Δ_perm = P − P_perm.

**Outcomes, checked in order:**

| Order | Outcome | Rule | Meaning |
|---|---|---|---|
| 1 | SENSITIVITY CHECK FAILED | P_c1 single-world accuracy on admissible TEST worlds < 0.80 | descriptive only. Passing it is an elementary check, not evidence of query-specific information (C_bil also reaches 0.96 on DEV) |
| 2 | INSUFFICIENT | < **875** contributing families | descriptive only |
| 3 | **H1b MET** | 2.5 % percentiles of Δ_C **and** Δ_perm both > 0 | see the mandatory interpretation |
| 4 | **H1b NOT MET** | otherwise, with the failing contrast(s) named | — |

**Secondary analyses** (one-sided bootstrap p, Holm over S1–S4):
- S1: P − B3+.
- S2: P − K.
- S3: Δ_C on single worlds.
- **S4: P − elimination.**

**Descriptive:** layers 32 and 48, P by model choice, C_bil and C_lin separately.

**Exclusions:**
- Excluded and counted: non-admissible worlds, exact ties, and pairs with a non-admissible member.
- A capture-integrity, tokenisation or fidelity failure stops the capture.
- No other exclusion is allowed.

## 6. Power with the exact rule (owner condition 3; `src/power_h1b_cmax.py` → `outputs/metrics/power_h1b_cmax.json`)

**Simulation:**
- The exact rule: a family bootstrap with C_max taken inside each resample (B = 1,000 per dataset), 400 datasets per cell.
- Latent-Gaussian correctness with ICC 0.15; PROBE-DEV pairs per family unless stated.
- Δ_C is defined against C_lin (0.55 paired); C_bil is 0.40 in the base scenario.

| Scenario (assumption) | Δ_C | Δ_perm = 0 | Δ_perm = 0.025 | Δ_perm = 0.05 | Δ_perm = 0.10 |
|---|---|---|---|---|---|
| **base** (C_bil 0.40; ρ(P, C_lin) 0.7, ρ(P, Q) 0.5) | 0.05 | 0.007 | 0.45 | **0.92** | 0.98 |
| base | 0.10 | 0.025 | 0.48 | 0.93 | 1.00 |
| **equal controls** (adverse: C_bil = C_lin = 0.55) | 0.05 | 0.030 | 0.38 | **0.90** | 0.96 |
| **low correlations** (adverse: every ρ = 0.3) | 0.05 | 0.020 | 0.33 | **0.82** | 0.87 |
| **low admissibility** (adverse: 0 / 1 / 2 pairs with probability 0.10 / 0.50 / 0.40), 1,000 families | 0.05 | 0.020 | 0.40 | **0.89** | 0.95 |
| low admissibility, 875 families | 0.05 | 0.020 | 0.32 | 0.83 | 0.89 |

**Readings:**
- When both contrasts are ≥ 0.05, the joint power with 1,000 families is **0.82–0.96** across scenarios, adverse ones included.
- **If Δ_perm ≤ 0.025, the power is poor** (0.32–0.48).
- If Δ_perm = 0, rejection stays at or below the nominal level (≤ 0.043). This is correct.
- **INSUFFICIENT risk:** with the DEV admissibility we expect about 980 contributing families. Under the adverse low-admissibility scenario we expect about **900**, close to the 875 threshold. If the new pool's admissibility falls below about 87.5 % of families having at least one pair, the outcome would be INSUFFICIENT.
- Earlier analyses kept for the record: `power_h1b.json` (single contrast) and `power_h1b_joint.json` (two contrasts, normal approximation).

## 7. LOCKED-TEST-H1b and capture (scripts frozen, NOT run)

**Generator** (`src/sealed_stage/build_h1b_pool.py`):
- Refused without an owner approval file.
- Seed 2026111403, families F40000–F40999, 1,000 depth-4 families (500 optimal-A / 500 optimal-B).
- Variants base, cf, upath_base, upath_cf: 4,000 worlds.
- Labels, worlds, families and prompt hashes disjoint from PROBE-DEV and the consumed LOCKED-TEST. These sets are regenerated in memory; no sealed file is read.
- Sealed immediately (LOCKED-TEST-H1B).

**Capture** (`src/sealed_stage/h1b_capture_test.py`):
- Stage H1B-CAPTURE: prompts only.
- 7 slots (ANS plus 6 arrival keys) × layers {32, 40, 48}.
- Frozen fidelity check, 20 replays, tokenisation sha.
- A100-80GB, timeout 1,500 s, guard 1,420 s; ≈ 1,000 s expected.
- Activations sealed immediately (LOCKED-TEST-H1B-ACT).

**Evaluation** (`src/sealed_stage/h1b_eval_test.py`):
- Stage H1B-EVAL, with a single-evaluation marker.
- Prechecks of every frozen hash in `V1_MANIFEST.json`.

## 8. Resources and cost ($5 BTC cumulative limit)

| Item | Value |
|---|---|
| Activations | ≈ 0.9 GB |
| Expected cost | ≈ $0.95 |
| Cap | $1.50 |
| BTC cumulative | $3.24 now → ≤ $4.74 after the capture |
| Second full GPU attempt under $5 | **not possible** |
| CPU for the evaluation | < 0.5 h |

## 9. Remaining steps (each needs its own authorisation)

1. Owner decision on the §4b option (keep D-8 secondary, or add / promote the elimination contrast). If changed, the freeze is regenerated and re-hashed **before** the pool exists.
2. Generate and seal LOCKED-TEST-H1b.
3. GPU capture.
4. Single analytical unlock and evaluation.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (no forward-sweep claim either way)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (PROBE-DEV only; no new TEST generated; LOCKED-TEST consumed, untouched)
[YES] Did not equate token availability with prospective path representation (C_max, P_perm, elimination, K, B3+)
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (CPU freeze preparation only)
[YES] Preserved negative results and raw logs (elimination > P on DEV reported prominently)
[YES] Labeled unrun stages NOT RUN and limitations honestly (INSUFFICIENT risk, low power if Δ_perm ≤ 0.025)
Any deviation: none.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decision on the §4b option, then on generation of LOCKED-TEST-H1b.
```
