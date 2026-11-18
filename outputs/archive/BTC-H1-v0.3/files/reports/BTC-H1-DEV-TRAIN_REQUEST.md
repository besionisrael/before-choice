# BTC-H1-DEV-TRAIN — request for probe training on PROBE-DEV (prepared; NOT executed)

**Date:** 2026-11-13 · **Status:** preparation only.
- **No probe has been trained on real activations.**
- No GPU. LOCKED-TEST is sealed and unread. H1 v0.3 is unchanged.

**Code prepared** (hashes in `reports/BTC-H1_FREEZE_MANIFEST.json → prepared`):
- `src/h1_train_dev.py` (runner, `a3f1d65c…`);
- `src/h1_dev_precheck.py` (`fb1ad7fd…`);
- `tests_cpucheck/test_h1_train_dev.py`.

## 1. Pre-training checks (owner items 1–5) — `outputs/h1_dev/precheck.json`

**1. Integrity (re-verified):**
- 12 shards: sha256 and shapes = `MANIFEST.json`.
- `MANIFEST.json` sha = freeze manifest (`306573de…`); `candidate_embeddings.npy` sha = freeze manifest.
- 4,800 unique `world_id`s, in the exact order of the PROBE-DEV inputs.
- `mean_emb` 12 × (400, 5,376) float32.

**2. Format violations outside decoys: 75 worlds**, all at **depth 4**, all with top-1 token **"To"**.

| Variant | train | val | Total |
|---|---|---|---|
| base | 6 | 2 | 8 |
| cf | 7 | 0 | 7 |
| alias_base / alias_cf | 3 / 2 | 2 / 1 | 5 / 3 |
| swap_base / swap_cf | 1 / 4 | 1 / 2 | 2 / 6 |
| shuffle_base / shuffle_cf | 3 / 5 | 0 / 0 | 3 / 5 |
| upath_base / upath_cf | 4 / 6 | 2 / 1 | 6 / 7 |
| cpath_base / cpath_cf | 5 / 4 | 0 / 0 | 5 / 4 |
| xswap_base / xswap_cf | 6 / 6 | 2 / 0 | 8 / 6 |
| **Total** | **62** | **13** | **75** |

- In 37 of the 75, the forced A/B choice is correct. Under H1 v0.3 §3 (format never used for inclusion or exclusion) these 37 **are included**; the other 38 are excluded for a wrong choice.
- The full `world_id` list is in `precheck.json`.

**3. Exact training-rule counts** (non-decoy, forced A/B choice correct, tie = excluded):

| Split | Depth 4 | Depth 3 | **Total included** | Excluded: wrong choice | Excluded: tie | Excluded: decoy |
|---|---|---|---|---|---|---|
| train | 1,644 | 981 | **2,625** | 531 | 8 | 452 |
| val | 550 | 305 | **855** | 179 | 2 | 148 |

- **Selection population** (DEV-val admissible depth-4 `upath` pairs): **77** (23 not admissible). Train: 224 admissible.
- u3 and c3 anchor heads and the u3 order head train only on depth-4 worlds (1,644 train / 550 val).

**4. Families:** 226 train / 74 val, **overlap 0**.

**5. Representations** (descriptive only; **not used to change the grid**):
- (62, 5,376) per world; **0 non-finite values** in activations, `mean_emb` and candidate embeddings.
- `mean_emb` L2 norm 46.6–55.1 (mean 50.1). Candidate embeddings (5,800 × 5,376) mean norm 70.7.

| Vector | 0 (emb) | 4 | 12 | 24 | 36 | 48 | 56 | 59 | **60 (post-norm)** | **61 (pre-norm, block 60)** |
|---|---|---|---|---|---|---|---|---|---|---|
| mean L2 norm | 82.6 | 95.5 | 245.8 | 274.3 | 255.8 | 142.5 | 92.7 | 103.6 | **391.7** | **10.6** |
| elem. sd | 1.13 | 1.30 | 3.35 | 3.74 | 3.49 | 1.94 | 1.27 | 1.41 | 5.37 | 0.15 |

**Observation recorded without action:** the raw block-60 output (61) has a much smaller norm than block 59 (103.6) and than the post-norm state (391.7). Its link to the model's logits is nonetheless confirmed by the pre-norm fidelity check (max |Δ| 0.205). Standardisation (H1 v0.3 §4) is per dimension, so scale does not by itself favour any vector.

## 2. Execution plan (owner items 6–8)

**Command** (after authorisation only):

```powershell
.\.venv-cpucheck\Scripts\python src\h1_train_dev.py --run
```

- Environment: py3.11, torch 2.5.1+cpu, 4 threads; local CPU only.
- **Built-in prechecks** (abort on failure): H1 code and document hashes = freeze manifest; activation `MANIFEST.json` = archived hash; LOCKED-TEST seal intact; access log absent.

**Pre-declared order and counts** (one global `BudgetMeter`, 48 CPU-hours = process CPU time, all threads):

| Step | Fits | Selection (DEV-val only) |
|---|---|---|
| P: primary u2, 17 ℓ × 2 r × 4 λ × 3 seeds | 408 | mean over seeds of paired accuracy on the 77 admissible depth-4 upath pairs; ties → smaller r, larger λ, lower ℓ → (ℓ\*, r\*, λ\*), E\* |
| A: B6 anchors c1, c3, u1, u3 (each full grid) | 1,632 | per head, single-world accuracy (c3/u3 on depth-4 worlds) |
| B1 (`mean_emb`), B5 (candidate-only): r × λ × seeds | 24 + 24 | paired accuracy |
| B3 (u2) and B3-order (u1, u2, u3): conditional logit, 4 λ each | 4 + 12 (logit) | paired accuracy / per-head accuracy |
| O: order heads u1, u3 at (ℓ\*, r\*, λ\*), 3 seeds, early stopping | 6 | each head's own E (§11.2) |
| B2: 3 paired permutations × 3 seeds at (ℓ\*, r\*, λ\*), E\* epochs | 9 | none |
| F: final refits on **all** DEV training-rule worlds (train + val), fixed epochs, 3 seeds, softmax-averaged | 27 bilinear + 4 logit | — |
| **Total** | **2,130 bilinear + 20 logit** | |

**Intermediate results without opportunistic selection:**
- Every completed fit is appended to `outputs/h1_dev/fits.jsonl` (key, best epoch, val loss, DEV-val metric, cumulative CPU-s). A budget trace goes to `budget.tsv`.
- `select()` **refuses to run unless every planned key of that grid is present**. On budget overrun the runner writes `BUDGET_EXCEEDED.json` (completed fits, status) and stops. **No selection or refit on a partial grid** (tested).
- **No restart without approval:** if `fits.jsonl` exists, the runner refuses unless `--resume --resume-approval <owner file>`. Resuming skips completed keys and never changes the plan.
- **Outputs:**
  - `SELECTIONS.json` (all selections);
  - `final/*.pt` (13 predictors);
  - `FINAL.json` (sha256 of every predictor, the standardisation provenance, CPU-hours used).
  These are frozen **before** any request to unlock LOCKED-TEST. **No TEST evaluation exists in the runner.**

**Budget:**
- Benchmark (synthetic, real shapes): **≈ 9.9 CPU-h worst case** (60 epochs per fit) at 4 threads, ≈ 3.3 CPU-h typical. Wall time ≈ 1–3.5 h.
- Limit **48 CPU-h** (margin ≈ ×4.8). Layer extraction from the uint16 shards is counted inside the same budget.

## 3. Conformance with the H1 v0.3 freeze (item 7) — `outputs/h1_dev/conformance.json`: 15/15 PASS

Checked:
- `h1_probe.py`, `h1_features.py` and the H1 document hashes equal the freeze.
- Grid: 17 vectors, r ∈ {8, 32}, λ ∈ {1e-4 … 1e-1}, seeds {0, 1, 2}.
- AdamW lr 1e-3, batch 256, at most 60 epochs, patience 5. Threads 4, budget 48 h.
- Counts 408 / 1,632 / 24 / 24 / 16; 2,130 bilinear in total.
- Training rule (decoys and ties excluded; forced choice correct).
- Prompt-only candidates in canonical order.
- Selection tie rule.
- No sealed path and no TEST evaluation.

**One operational clarification (please validate):** B2 on depth-3 worlds.
- The frozen B2 permutes 6 slots, but depth-3 worlds have 4 (c3/u3 do not exist).
- My first runner draft fell back to the **true** u2 target when the permuted slot was missing. That would have leaked the target into B2.
- I caught this before any use. The fix applies the same sha256 seeding to a permutation of the **4 available slots**, never falling back to the true target (tested).
- Depth-4 worlds use the frozen 6-slot permutation unchanged.

## 4. Tests

| Suite | Result |
|---|---|
| `tests_cpucheck` (synthetic only) | **31/31**, including the runner tests: frozen counts; end-to-end on synthetic shards with final refits (13 predictors); restart refused; selection refused on an incomplete grid; budget overrun stops without `SELECTIONS.json`; B2 depth-3 slots paired and never falling back |
| Realistic-scale synthetic self-test (1,600 worlds, d = 64, signal planted at vector 12) | selection correctly chose ℓ = 12 (paired 0.67 vs B1 0.03, B5 0.00, B3 0.15) |
| Main | 84/84 |

## 5. Risks and limits (stated before any real result)

1. **Possible under-training** (bias toward false negatives). In the realistic synthetic test, best epochs often reached the 60-epoch cap. With 2,625 training worlds and batch 256 there are ≈ 11 optimiser steps per epoch, so at most ≈ 660 steps at lr 1e-3. The frozen optimiser is **kept as is**. Any change would need an owner decision **before** training on real data and must not depend on DEV results.
2. **Coarse selection metric:** 77 val pairs, so paired accuracy moves in steps of 1/77 and ties between configurations may be frequent. The pre-registered tie rule resolves them deterministically.
3. **Prompt-only baselines can be strong for anchor-adjacent targets.** In the synthetic test, B3-order reached 1.0 for u1, which is one hop from the action line. This is expected and is why B6 and B3 are thresholds.
4. **Not part of this request:** any LOCKED-TEST evaluation, D3-TEST capture, unlock, or H2/H3 analysis.

## 6. Authorisation request

```text
AUTORISATION DEMANDÉE — BTC-H1-DEV-TRAIN
Objectif scientifique : entraîner et sélectionner sur PROBE-DEV uniquement la sonde H1 (D-1′) et toutes ses baselines selon le gel H1 v0.3, puis figer les 13 prédicteurs finaux avant toute évaluation sur LOCKED-TEST.
Why needed / alternative : étape préalable obligatoire à l'évaluation confirmatoire ; aucune alternative.
Exact change/command : .\.venv-cpucheck\Scripts\python src\h1_train_dev.py --run — 2 130 ajustements bilinéaires + 20 logit conditionnels, ordre pré-déclaré, sélection sur DEV-val (77 paires admissibles), refits finaux sur tout DEV ; sorties outputs/h1_dev/ (fits.jsonl, SELECTIONS.json, final/*.pt, FINAL.json avec empreintes).
Compute and cost : CPU local uniquement, 4 threads ; ≈ 3–10 CPU-heures attendues (pire cas benchmark 9,9 h) ; budget global 48 CPU-heures (temps CPU du processus) ; 0 $ cloud.
Risk and stop condition : arrêt immédiat si prévérification échoue (empreintes H1, activations, sceau LOCKED-TEST) ou si le budget est dépassé (BUDGET_EXCEEDED.json, aucune sélection sur grille incomplète) ; aucune reprise sans approbation ; aucune évaluation TEST ; ARRÊT après FINAL.json pour validation.
I will not execute this step until you explicitly approve it.
```

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (H1 v0.3 unchanged)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (prompt-only inputs; families disjoint; LOCKED-TEST sealed, unread)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (training requested, not executed)
[YES] Preserved negative results and raw logs
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none (B2 depth-3 clarification submitted for validation).
Next step: owner decision on BTC-H1-DEV-TRAIN (and on the B2 depth-3 clarification).
```
