# BTC-D3-TEST — CPU preparation and GPU request (capture of LOCKED-TEST)

**Date:** 2026-11-14 · **Authorisation:** owner, "DÉCISION SCIENTIFIQUE — POST BTC-H1-DEV-TRAIN" (CPU preparation only) · **Status:** PREPARED, **NOT RUN.**

> No GPU was used, no unlock file exists, and no evaluation was run. LOCKED-TEST is still sealed and unread: its access log is absent and its seal verifies with `hash_only`. Nothing was changed in the predictors, grid, thresholds, data or H1 v0.3.

## 1. Item 1: the 16 final predictors are verified

Script: `src/h1_verify_final.py` → `outputs/h1_dev/FINAL_VERIFICATION.json` (sha `1a87724f…`). It ran on CPU and used DEV data only.

| Check | Result |
|---|---|
| sha256 of the 16 predictors = `FINAL.json` = freeze manifest (`artifacts.H1-DEV-TRAIN`) | **PASS** |
| Hashes of `FINAL.json`, `SELECTIONS.json`, runner, `h1_probe.py`, `h1_features.py` | **PASS** |
| Structure vs H1 v0.3: per predictor, ℓ, r, λ, E equal the frozen selections; U/V shapes; 3 seed fits; B2 permutation ids 0–2; B3 heads use the 7 frozen features | **PASS (16/16)** |
| **Bitwise reproduction** (read-only): the primary u2 refit (3 seeds) and the B3 refit are retrained from the frozen configuration and compared with the saved weights | **identical** (max\|Δ\| = 0) |
| Frozen DEV-train standardisation (z per layer, `mean_emb`, candidate table) written once to `final/STANDARDISATION.npz` (sha `9cd6b229…`) | **identical** to the runner's transform |
| Pipeline sanity (primary u2, **in-sample** on DEV, not a result) | 0.565 world accuracy over 3,480 worlds |

**Why the standardisation file is needed.**
- The final refits standardised z and e(c) with DEV-train statistics, but `FINAL.json` did not store those statistics.
- They are now recomputed deterministically from the frozen DEV activations and hashed. The TEST evaluation therefore applies exactly the DEV transform.
- TEST labels are new, so their e(c) come from the TEST capture, computed by the same frozen function. They are standardised with the **DEV** statistics.

## 2. Item 2: the capture script reuses the D3-DEV fidelity controls unchanged

Script: `src/sealed_stage/d3_run_gemma_test.py` (sha `3318895a…`).

**Identical to the approved D3-DEV-R1 run**, and checked by tests:
- the same frozen `d3_capture_core` (sha `00a3fa22…`, checked against the freeze manifest before any read), so the fidelity rules are exactly those of D3-DEV-R1:
  - max\|Δ\| ≤ 0.25 over the full vocabulary;
  - identical A/B decision, with the original-tie rule and the 2d + τ tolerance (τ = 0.125);
  - rejection of non-finite values;
  - 20 bitwise replays;
  - divergence log;
- the same pinned image and model revision;
- the same `run_capture` call;
- the same limits: A100-80GB, cpu (4, 4), memory 16–96 GiB, retries = 0, max_containers = 1, timeout 1,500 s, guard 1,420 s;
- the same transfer path: ephemeral Volume, commit wait 300 s, `scaledown_window = 2`, and `verify_shards` **before** the Volume is deleted.

**Differences, all intended:**
- **Input:** the script reads only `model_inputs.jsonl`, through stage-scoped access (§3).
  - Its hash must equal the frozen `8fbc920d…`.
  - Its schema must match PROBE-DEV exactly. Only world_id, prompt and prompt hash are sent to Modal.
- **Preflight, before anything is read:**
  - frozen code hashes;
  - pool seal intact;
  - no earlier capture;
  - no earlier read outside the capture scope.
- **Completeness:** the captured world ids must equal the input world ids, in addition to the 4,800-world count.
- **Sealing:**
  - The activation directory is **sealed immediately** after verification (seal `LOCKED-TEST-D3-ACT`, read-only).
  - Outside the seal, the only file written is an integrity summary (`outputs/d3/d3_test_capture_summary.json`): counts, fidelity summary, environment, timing.

## 3. Item 3: prompt access for capture is kept separate from the analytical unlock

Module: `src/sealed_stage/sealed_access.py` (sha `87b1e3a6…`). It is built on the frozen `sealed.sealed_read`, which is unchanged.

| Stage | Readable | Not readable |
|---|---|---|
| **D3-TEST-CAPTURE** | `model_inputs.jsonl` of LOCKED-TEST (model-visible prompts, no oracle) | `worlds.jsonl` (oracle), the activations |
| **H1-TEST-EVAL** (analytical unlock) | `worlds.jsonl`, `model_inputs.jsonl`, the sealed TEST activations | — |

**How a stage is opened:**
- Each stage has its own owner unlock file: `outputs/approvals/unlock/<STAGE>/UNLOCK_<seal>.json`, with fields owner_approval_id, approved_utc, scope = stage, and files.
- A read is refused if any of these holds:
  - the file of that stage is missing;
  - its scope does not match the stage;
  - the requested file is not allowed for the stage;
  - the requested file is not listed in the unlock file.
- A capture unlock that lists `"*"` still cannot open `worlds.jsonl`.
- Each permitted read is hash-checked and logged with its stage.
- Before reading anything, the evaluator refuses to start if the access log shows any earlier read outside the capture scope, or any earlier read of the activations.

**Code placement:**
- Only `src/sealed_stage/` (plus `build_d5.py` and `sealed.py`) names the sealed paths.
- DEV modules in `src/` are still scanned by the frozen `tests/test_d5_dev.py`, and by a new test that applies the same scan.

**Residual limit:** a script could still call `sealed.sealed_read` directly with a capture unlock directory, bypassing the stage check. Mitigation: no code does this (scan), and every access is logged.

## 4. Item 4: the single evaluation (prepared, not run)

Script: `src/sealed_stage/h1_eval_test.py` (sha `a91d386d…`). It runs on CPU, and its own hash is checked at start.

**Sequence:**
1. **One evaluation only.** `EVAL_STARTED.json` is written *before* the first sealed read, so a crash cannot be followed by a silent second attempt. If the file exists, the script refuses to run.
2. **Prechecks:** predictor hashes, standardisation hash, frozen code, both seals intact, and no earlier read of the oracle or the activations.
3. **Admissibility:** the forced A/B choice from `logp_A` vs `logp_B` (an exact tie is not admissible) must equal the oracle optimum.
4. **Primary analysis:**
   - population: admissible depth-4 `upath` pairs;
   - insufficiency rule: fewer than **180** pairs or fewer than **100** families → `INSUFFICIENT` (descriptive only);
   - H1 MET iff, for each k ∈ {B1, **B2 (max over the 3 permutations inside each replicate)**, B3, B6}, the 95 % family-cluster percentile bootstrap lower bound of probe − B_k is > 0 (10,000 resamples, seed 0, the same resamples for every k).
5. **L2, only if H1 is MET:**
   - (a) same-layer order test (`order_u1`, `primary_u2`, `order_u3`; joint distinct assignment) vs max(1/6, B3-order);
   - (b) alias generalisation vs max(1/6, B3);
   - (c) shuffle generalisation vs max(1/6, B3).
6. **Reported, not thresholds:** B5; B7 (`cpath` pairs); B4 (0 by construction); per-head accuracies; secondaries (§5, O-2).
7. **Outputs:** `outputs/h1_test/EVAL_RESULT.json` and `per_world.jsonl`.

**Synthetic tests:**
- Predictors are trained by the real runner. The fake sealed pools and activations are built from PROBE-DEV worlds in temporary directories and read through `staged_read`.
- Results:
  - planted signal → **MET**, with L2 evaluated;
  - no signal → **NOT MET**, with L2 not evaluated;
  - frozen thresholds on a small sample → **INSUFFICIENT**;
  - decoys are excluded and counted;
  - a second evaluation is refused before any read;
  - stage scopes behave as specified.

**Test results:**
- `tests_cpucheck`: **42/42** (10 new).
- Main suite: **88/88** (4 new: static checks of the capture script against the DEV script).
- Logs: `outputs/logs/d3test_prep_tests_{cpucheck,main}.log`.
- All new hashes are recorded in `BTC-H1_FREEZE_MANIFEST.json → prepared.BTC-D3-TEST`.

## 5. Points that need an owner decision **before** any unlock

These are not tuning: v0.3 leaves them operationally open, and they must be fixed before TEST is read.

| ID | Point | Proposal (implemented as prepared) |
|---|---|---|
| **C-1** | v0.3 does not say against which target B2 is **scored** | B2_k is scored against **its own permuted target** π_f,k(u2). This is the usual control-task (Hewitt–Liang) reading of a "capacity" threshold, and it is conservative: when π(u2) is an easy slot (c1 or u1), B2 is high. The alternative, scoring against the true u2, would sit near chance. |
| **C-2** | L2(a) population: "admissible depth-4 base and cf worlds" | Read literally: only the `base` and `cf` variants. |
| **O-2** | §8 secondaries are listed without an operational form | **S1** depth 3 + 4 pooled `upath` and **S2** `xswap` (depth 4): probe − B3, paired, one-sided bootstrap p-values, **Holm** over S1–S2. **S3** per-depth head accuracies: descriptive. **B7** `cpath`: reported. |
| **N-1** | Decoy control (S4) | **Not computable** with the frozen predictors. Decoy worlds show 10 candidates at depth 4 (7 at depth 3), but the decoders are 6-candidate softmaxes, and restricting the candidates would require the oracle. Reported as NOT COMPUTABLE. |
| **N-2** | H2 layer profiles | **NOT RUN.** Each head is frozen at a single vector index, so a profile would need new per-layer training on DEV, which is not authorised. |
| **T-1** | `tests/test_d5_dev.py::test_locked_test_sealed_read_only` asserts that the access log is absent | After an authorised capture this assertion will fail, as expected. I propose a dated amendment: the log may contain only capture-scope reads of `model_inputs.jsonl` until the H1-TEST-EVAL unlock. Until then the failure would be reported, not hidden. |

## 6. GPU request

```text
AUTORISATION DEMANDÉE — BTC-D3-TEST
Objectif scientifique : capturer les activations pré-réponse de Gemma-4-31B-it sur les 4 800 mondes de LOCKED-TEST, préalable unique à l'évaluation confirmatoire H1 (aucune analyse dans ce run).
Why needed / alternative : les prédicteurs gelés ne peuvent être évalués que sur des activations TEST capturées à l'identique de DEV ; aucune alternative CPU (le modèle 31B ne tient pas en local).
Exact change/command : `modal run src/sealed_stage/d3_run_gemma_test.py` (sha 3318895a…), même cœur de capture gelé (d3_capture_core 00a3fa22…), même image et même révision 842da379… ; entrée = model_inputs.jsonl seul (8fbc920d…) via la portée D3-TEST-CAPTURE ; sorties scellées immédiatement (LOCKED-TEST-D3-ACT) ; worlds.jsonl reste illisible.
Compute and cost : 1 × A100-80GB, cpu (4,4), mémoire 16–96 GiB, max_containers=1, retries=0 ; durée attendue ≈ 12–15 min (D3-DEV-R1 : 700 s d'exécution distante, 861 s au total) ; timeout strict 1 500 s ; coût attendu ≈ 0,60 $ (D3-DEV-R1 : 0,579 $) ; plafond 2,00 $ (pire cas calcul 1,50 $ + transfert ≈ 3,4 Go) ; cumul après run ≤ ≈ 3,75 $ sous la limite de 5 $.
Risk and stop condition : arrêt sans relance sur tout échec de pré-vérification, de fidélité (max|Δ| > 0,25, décision A/B divergente hors règle, valeur non finie), de rejeu, de nombre de tokens, de transfert ou d'intégrité (hash, formes, 4 800 identifiants) ; Volume éphémère supprimé même en cas d'erreur ; coût relevé par `modal billing report` aussitôt après. Aucun déverrouillage analytique, aucune évaluation, aucun accès à worlds.jsonl.
I will not execute this step until you explicitly approve it.
```

**What approval would need to include:**
- the go-ahead for BTC-D3-TEST;
- decisions on C-1, C-2, O-2 and T-1;
- the capture-stage unlock. The owner can write this file, or authorise me to write it with their approval ID:

```json
// outputs/approvals/unlock/D3-TEST-CAPTURE/UNLOCK_LOCKED-TEST-D5.json
{"owner_approval_id": "<BTC-D3-TEST approval id>", "approved_utc": "<UTC>", "scope": "D3-TEST-CAPTURE", "files": ["model_inputs.jsonl"]}
```

The analytical unlock (stage H1-TEST-EVAL) and the single evaluation would each need a **later**, separate authorisation.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (H1 v0.3 unchanged)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (LOCKED-TEST sealed, unread; capture stage cannot read the oracle)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (CPU preparation authorised; GPU requested, not run)
[YES] Preserved negative results and raw logs
[YES] Labeled unrun stages NOT RUN and limitations honestly (S4 not computable, H2 profiles not run, residual access limit)
Any deviation: none. New derived artifact: frozen standardisation file (recomputed from DEV, bitwise-identical transform).
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decision on BTC-D3-TEST and on C-1, C-2, O-2, T-1.
```
