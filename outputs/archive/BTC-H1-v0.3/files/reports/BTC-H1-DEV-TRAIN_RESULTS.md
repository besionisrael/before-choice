# BTC-H1-DEV-TRAIN — DEV training and selection results (EXPLORATORY / SELECTION ONLY)

**Date:** 2026-11-14 · **Authorisation:** owner, BTC-H1-DEV-TRAIN, single CPU execution on PROBE-DEV · **Status:** completed, `FINAL.json` verified, **STOPPED**.

> **Everything below is DEV selection data, not confirmatory evidence.** H1 is decided only on LOCKED-TEST under the frozen rules (H1 v0.3 §6). LOCKED-TEST is sealed and unread. No D3-TEST capture, no unlock, no causal intervention. H1–H3 and L1–L4 are unchanged.

## 1. Pre-launch changes (authorisation conditions) and checks

**Changes:**
- **B2 predictors** were not being saved in the prepared runner; they are now saved. B2 has no selection step, so its **9 frozen fits** are trained on all DEV training-rule worlds at (ℓ\*, r\*, λ\*), E\* epochs, with one 3-seed-averaged predictor per permutation.
- **Final predictor set** is now asserted complete: **16 predictors.**
  - **Convention:** the previously announced 13 = primary u2 + B6 anchors c1, c3, u1, u3 + order heads u1, u3 + B1 + B5 + B3 + B3-order u1, u2, u3. **B2 had been omitted**, and now adds 3 (one per permutation).
  - The L2 order test uses `primary_u2` as its u2 head.
- **B2 depth-3 rule as approved:** only the 4 available slots are permuted, with sha256 seeds; no fallback to the true target; no derangement imposed (the target stays in place in 17–21 % of training worlds per permutation); the 6-slot permutation is unchanged at depth 4.
- **Ledger:** every fit now records DEV-train **and** DEV-val metrics and a `hit_epoch_cap` flag.

**Checks:**
- Tests: cpucheck 32/32, main 84/84.
- Built-in prechecks PASS: H1 code and document frozen, activation manifest identical, LOCKED-TEST seal intact, never read.
- Counts exactly as authorised: **train 2,625 / val 855**, **77** DEV-val admissible depth-4 `upath` pairs.

## 2. Run

| Item | Value |
|---|---|
| Command | `.\.venv-cpucheck\Scripts\python src\h1_train_dev.py --run` (runner sha `5c9b10f7…`) |
| UTC | 2026-11-14 04:54:07 → 05:29:36 (35.5 min wall) |
| Threads / budget | 4 / 48 CPU-h (`time.process_time()`) |
| **CPU used** | **1.475 CPU-h** (5,310 s; 3.1 % of budget) |
| Fits | 2,130 bilinear + 20 conditional logit = 2,150 budget ticks (`budget.tsv`); 2,146 ledger records (the 4 logit refits are not ledger entries) |
| Grid | exactly v0.3: 17 vectors × r ∈ {8, 32} × λ ∈ {1e-4, 1e-3, 1e-2, 1e-1} × seeds {0, 1, 2}; no change after observation |
| **Fits reaching the 60-epoch cap** | **0 of 2,094** early-stopped bilinear fits |
| Cloud cost | $0 |

## 3. Selections (DEV-val; frozen tie rules)

| Predictor | Selected config | DEV-val metric (mean of 3 seeds) | DEV-train metric | E |
|---|---|---|---|---|
| **Primary u2 (H1 probe)** | ℓ\* = **32**, r\* = 8, λ\* = 0.1 | **paired accuracy 0.030** (seeds 0.052 / 0.013 / 0.026) | paired 0.071 | **5** |
| B6 anchor c1 | ℓ = 32, r = 32, λ = 0.01 | world accuracy 0.248 | 0.795 | 10 |
| B6 anchor c3 | ℓ = 40, r = 32, λ = 0.01 | 0.313 | 0.929 | 11 |
| B6 anchor u1 | ℓ = 56, r = 32, λ = 1e-4 | 0.244 | 0.913 | 8 |
| B6 anchor u3 | ℓ = 52, r = 32, λ = 0.1 | 0.275 | 0.923 | 14 |
| Order head u1 (at ℓ\*) | — | 0.199 / 0.187 / 0.202 | 0.43–0.64 | 5 |
| Order head u3 (at ℓ\*) | — | 0.156 / 0.162 / 0.180 | 0.52–0.63 | 6 |
| B1 (`mean_emb`) | r = 8, λ = 0.1 | paired 0.000 | 0.000 | 6 |
| B5 (candidate-only) | r = 8, λ = 0.1 | paired 0.000 | 0.000 | 6 |
| **B3 (prompt text)** | λ = 0.1 (all λ equal) | **paired 0.156** | 0.268 | — |
| B3-order u1 / u2 / u3 | λ = 0.1 | world accuracy **1.000** / 0.327 / 0.222 | 1.000 / 0.378 / 0.276 | — |
| B2 perm 0/1/2 | (ℓ\*, r\*, λ\*), E\* = 5 | (no DEV-val metric: trained on all DEV by design) | — | 5 |

**Reference values:**
- Chance for the primary paired metric at depth 4 is **1/36 ≈ 0.028**.
- For single-world accuracy, chance is 1/6 at depth 4 and 1/4 at depth 3 (pooled DEV-val ≈ 0.196 for heads that use both depths; 1/6 for c3/u3, which are depth-4 only).

**Primary probe over the whole grid:** DEV-val paired accuracy has a mean of 0.012 and a maximum of 0.052 over 408 single-seed fits. The best mean per layer is ≤ 0.030 for all 17 vectors.

## 4. Reading (DEV only — exploratory; no confirmatory claim)

1. **The selected H1 probe is at chance on DEV-val** (0.030 vs 0.028). No vector, rank or regularisation produced a transferable signal for the unchosen depth-2 state.
2. **The activation-based heads overfit strongly.**
   - Anchors reach 0.80–0.93 on train against 0.24–0.31 on val; the primary probe at layer 4 reaches 0.57 vs 0.03.
   - Early stopping triggered after 3–14 epochs, and no fit reached the cap. **The anticipated under-training risk did not materialise.** The limit is generalisation, with 5,376-dimensional inputs and about 2,600 training worlds.
   - Anchor heads are modestly above chance (0.24–0.31 against 0.196 or 1/6).
3. **The prompt-only baseline B3 beats the probe** (paired 0.156).
   - B3-order u1 is perfect (1.000), because the depth-1 label is printed in the action line, a 1-hop anchor (P0).
   - B3's u2 accuracy (0.327) is consistent with **eliminating the two depth-1 labels via the action lines** and guessing among the remainder: expected ≈ (550 × 1/4 + 305 × 1/2)/855 ≈ 0.34.
   - This is the anchor-and-eliminate shortcut the protocol anticipated. It is a prompt property, not model computation.
4. **B1 and B5 are exactly 0**: neither lexical presence nor candidate identity carries the target, as designed.

**What this does not allow:**
- Any conclusion about H1. The confirmatory evaluation is defined on LOCKED-TEST.
- Any change to the probe, grid, optimiser, baselines or thresholds. The predictors are **frozen and hashed** (§5).
- Interpreting the DEV result as evidence of absence. It is a selection-stage observation with 77 val pairs.

## 5. Final predictors (16) — `outputs/h1_dev/final/`, hashes verified against `FINAL.json`

| Predictor | sha256 (prefix) | | Predictor | sha256 (prefix) |
|---|---|---|---|---|
| primary_u2 | `170c76569cb9` | | B1 | `6b6adf791ce1` |
| anchor_c1 | `82710a6eb4ac` | | B5 | `01e84e689c14` |
| anchor_c3 | `098b44f6ad49` | | B2_perm0 | `8a8944c7398b` |
| anchor_u1 | `64c39f2595d8` | | B2_perm1 | `0e3c1ad46dea` |
| anchor_u3 | `339464972f60` | | B2_perm2 | `0cd613164e1e` |
| order_u1 | `aa8c64a75b2a` | | B3 | `2216c55c0d69` |
| order_u3 | `2771fdf19e6c` | | B3o_u1 / B3o_u2 / B3o_u3 | `7dedd469374e` / `9fa7d3a0ef71` / `441c0d116eac` |

- Full hashes, `FINAL.json`, `SELECTIONS.json`, `fits.jsonl` and `budget.tsv` hashes are recorded in `reports/BTC-H1_FREEZE_MANIFEST.json → artifacts.H1-DEV-TRAIN`.
- Standardisation: DEV-train statistics for z and for candidate embeddings.
- The predictors are fixed **before** any LOCKED-TEST access.

## 6. Status and next decision

- DEV training is complete and frozen. Under H1 v0.3, the remaining confirmatory steps are D3-TEST capture of LOCKED-TEST, unlock, one evaluation, and the frozen decision rules. Each requires a separate owner authorisation.
- **My recommendation** (for the owner's decision): carry out the pre-registered confirmatory evaluation as planned, even though the DEV signal is at chance.
  - Stopping now because of an unfavourable DEV result would be a form of selective reporting.
  - The frozen procedure exists precisely to report the outcome, whatever it is.
  - Its marginal cost is small (≈ $1 GPU + CPU).

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (H1 v0.3 unchanged)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (DEV only; LOCKED-TEST sealed, unread)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (BTC-H1-DEV-TRAIN)
[YES] Preserved negative results and raw logs (fits.jsonl, budget.tsv, all selections)
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none beyond the authorised pre-launch fixes (B2 predictors saved; DEV-train metrics and cap flags logged).
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decision on the confirmatory path (D3-TEST capture, unlock, single evaluation).
```
