# BTC-D3-TEST-CAPTURE — results: **FAILED at container startup (no activations, no retry)**

**Date:** 2026-11-14 · **Authorisation:** owner, "DÉCISION SCIENTIFIQUE — BTC-D3-TEST" (one conditional capture, no evaluation) · **Status:** attempt 1 FAILED, **STOPPED** under the "no relaunch on failure" condition.

> **What was not done:** no activation was produced, nothing was sealed, the oracle was not read and no score was computed. The only access to the sealed pool was **one** read of the prompt file `model_inputs.jsonl` in the capture scope, which was logged. **What changed:** the predictors, H1 criteria and data are unchanged. The evaluator was amended for C-1 and O-2 as instructed (§5).

## 1. Pre-launch checks (owner conditions): all PASS

All checks are recorded in `outputs/d3/d3_test_prelaunch.json`.

| Condition | Result |
|---|---|
| Hashes of the 16 predictors and of `STANDARDISATION.npz` | **PASS.** Hashes equal `FINAL.json` and the freeze manifest. `STANDARDISATION.npz` = `9cd6b229…` |
| Normalisation comes from DEV only and reproduces the training transform | **PASS.** Recomputed from the 2,625 DEV-train training-rule worlds (split `train` only; activation manifest `306573de…` = freeze), the arrays are identical. The z and T transforms are identical to the runner's (`FINAL_VERIFICATION.json`) |
| Gemma model and pinned revision | **PASS.** `google/gemma-4-31B-it` @ `842da3794eaa0b77d5f08bae87a17459d91ff475`, A/B ids 236776/236799, `v2_common.py` frozen |
| Capture can read only the TEST prompts | **PASS.** In the D3-TEST-CAPTURE scope, `worlds.jsonl` and `manifest.json` are refused by code. Only `model_inputs.jsonl` is allowed. Script and access-module hashes match the prepared versions |
| D3-A3 protocol maintained | **PASS.** `d3_capture_core` frozen (`00a3fa22…`): max\|Δ\| 0.25, τ 0.125, non-finite rejection, 20 replays, shard check before Volume deletion |
| Seal state before launch | **PASS.** Pool intact, access log absent, no activation directory |
| T-1 amendment (D3-A4) applied before launch | `tests/test_d5_dev.py` now accepts only capture-scope reads of `model_inputs.jsonl` and fails on any other access. The log is never deleted. Amendment D3-A4 is recorded in the manifest with the old and new hashes |
| Capture unlock | `outputs/approvals/unlock/D3-TEST-CAPTURE/UNLOCK_LOCKED-TEST-D5.json`: scope D3-TEST-CAPTURE, files [`model_inputs.jsonl`]. Written on your authorisation of the capture |

## 2. What happened

| Time (UTC) | Event |
|---|---|
| 06:00:18 | `modal run src/sealed_stage/d3_run_gemma_test.py`. Local preflight passes. One sealed read of `model_inputs.jsonl` (scope D3-TEST-CAPTURE, logged 06:00:20) |
| 06:00:2x | Modal app `ap-GmiXKvUh7mZPZ2ZiqLwRCS` created. The container **fails at startup** with `IndexError: 2` |
| 06:00–06:02 | Modal restarts the container startup automatically (startup failures are retried by the platform, independently of `retries=0`): 12 "failed during startup" messages in the log |
| 06:02:33 | I **stopped the app manually** (`modal app stop`). State: `stopped`, 0 tasks |
| 06:02:43 | Local process ends with `RemoteError`, EXIT=1. The ephemeral Volume is deleted with its context |

**Cause.**
- The capture script computed `ROOT = Path(__file__).resolve().parents[2]` at module level.
- Inside the container the script is mounted as `/root/d3_run_gemma_test.py`, which has only 2 parents, so the import raised `IndexError`.
- The model was never loaded and no forward pass ran.
- The approved DEV script does not have this line, which is why D3-DEV-R1 succeeded.
- **My static tests imported the script only from its project location, so they could not catch this.**

## 3. Integrity, hashes, divergences, duration, cost

| Item | Value |
|---|---|
| Activations | **none** (`outputs/d3/locked_test/` exists but is **empty and unsealed**; it was created before the remote call and is kept as a trace) |
| Fidelity / divergences | not applicable (no forward pass) |
| Pool seal | **intact** (`hash_only` all True); files read-only |
| Pool access log | **1 entry**: `model_inputs.jsonl`, purpose `D3-TEST-CAPTURE: BTC-D3-TEST forward passes (prompts only)`. No access to `worlds.jsonl` |
| Amended seal test (D3-A4) | **PASS** with this log |
| Activation seal | **not created** |
| Duration | 2 min 25 s wall (06:00:18 → 06:02:43) |
| Cost (`modal billing report --for today --show-resources`) | **$0.00** for A100-80GB, CPU and Memory on `ap-GmiXKvUh…`. The report may still be provisional for the current day; I will check it again before any new run |
| Logs | `outputs/logs/d3test_run.log`; start/end UTC files; manifest `prepared.BTC-D3-TEST.attempt_1` (hashes of the log, the prelaunch file and the access log) |

## 4. Fix prepared on CPU (not run)

`src/sealed_stage/d3_run_gemma_test.py` (sha `d7936624…`):
1. `ROOT` is resolved only when the script runs from the checkout. In the container it is `None`, and the local entrypoint asserts that it is set.
2. Preflight accepts an activation directory that is absent **or empty and unsealed**, which is the trace left by attempt 1. Any non-empty directory still blocks the run.
3. Nothing else changed: same capture core, image, limits and transfer order. Tests check this.

**New regression test:** a copy of the script placed outside the project must import cleanly, with `ROOT = None`.

**Tests:** cpucheck **44/44**, main **89/89**. Logs: `outputs/logs/d3test_attempt1_post_tests_*.log`. Hashes are in the manifest under `prepared.BTC-D3-TEST-R1`.

## 5. Evaluator amendments for C-1 and O-2 (CPU, prepared, not run)

`src/sealed_stage/h1_eval_test.py` (sha `1b781430…`):
- **C-1 (as decided):**
  - B2 predictors, trained on permuted targets, are **scored against the true u2 target** on the same admissible pairs as H1;
  - max(B2₀, B2₁, B2₂) is taken inside every bootstrap replicate;
  - unit test: a B2 head that hits the true u2 scores 1, one that hits the permuted slot scores 0.
- **O-2 (as decided):** `plan_status` lists **every** pre-registered secondary analysis with its status:
  - S1 pooled depth 3 + 4, and S2 `xswap`: INFERENTIAL (Holm);
  - S3 per-depth reconstruction: DESCRIPTIVE;
  - S4 decoy: NOT COMPUTABLE, with reason;
  - H2 layer profiles: NOT RUN, with reason;
  - B7: descriptive reference;
  - B4: not computed, 0 by construction, with reason.
  - The Holm family is labelled **"a subset of the pre-registered plan, not the whole plan"**.
- **C-2:** the L2(a) population is unchanged (admissible depth-4 base/cf worlds).

The **separate BTC-H1-TEST-EVAL request is not submitted yet.** It requires TEST activations, which do not exist. When it is prepared, the evaluator's self-hash check will point to its own manifest entry.

## 6. New authorisation requested (relaunch after a fix)

```text
AUTORISATION DEMANDÉE — BTC-D3-TEST-R1
Objectif scientifique : capturer les activations pré-réponse de Gemma-4-31B-it sur les 4 800 mondes de LOCKED-TEST (préalable à l'évaluation confirmatoire H1 ; aucune analyse).
Why needed / alternative : la tentative 1 a échoué au démarrage du conteneur (IndexError sur un calcul de chemin au chargement du module), sans passe avant ni activation ; correctif minimal testé ; aucune alternative CPU.
Exact change/command : `modal run src/sealed_stage/d3_run_gemma_test.py` (sha d7936624…) ; seul changement vs la version approuvée : résolution de ROOT locale uniquement, et prévol acceptant le répertoire vide non scellé laissé par la tentative 1 ; même cœur gelé (00a3fa22…), même image, même révision 842da379… ; entrée = model_inputs.jsonl seul via la portée D3-TEST-CAPTURE (une seconde lecture journalisée) ; activations scellées aussitôt (LOCKED-TEST-D3-ACT).
Compute and cost : 1 × A100-80GB, cpu (4,4), mémoire 16–96 GiB, max_containers=1, retries=0, timeout 1 500 s, garde 1 420 s ; coût attendu ≈ 0,60 $, objectif ≤ 2 $ ; cumul actuel ≈ 1,75 $ (tentative 1 : 0,00 $ selon le rapport, à reconfirmer), limite 5 $.
Risk and stop condition : arrêt sans relance sur tout échec ; nouveauté : si le conteneur échoue au démarrage, j'arrête l'application immédiatement (`modal app stop`) pour couper les redémarrages automatiques de la plateforme. Aucun accès à worlds.jsonl, aucune évaluation.
I will not execute this step until you explicitly approve it.
```

**For the record:** the capture unlock file of attempt 1 is still in place, and its `owner_approval_id` refers to attempt 1. On approval of R1 I would add the R1 approval id to it. The access log already holds the attempt-1 read and is kept intact.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (H1 v0.3 unchanged)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (one logged prompt read; oracle never read)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (single capture authorised; failed; no relaunch)
[YES] Preserved negative results and raw logs (failure log, access log, empty directory kept)
[YES] Labeled unrun stages NOT RUN and limitations honestly (my test gap that missed the bug is stated)
Any deviation: attempt 1 failed at startup; app stopped manually to halt platform startup restarts. Amendment D3-A4 (T-1) applied as approved.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decision on BTC-D3-TEST-R1.
```
