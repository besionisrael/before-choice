# BTC-D3-DEV-R1 — corrected fidelity check (D3 technical amendment A2) + rerun request (NOT RUN)

**Date:** 2026-11-13 · **Status:** CPU work done. **No GPU, no training on real activations, no LOCKED-TEST processing.**
**Unchanged:** the H1 v0.3 freeze (document sha `09523e0f…`, verified unchanged), its criteria, PROBE-DEV, the sealed LOCKED-TEST and all baselines.

## 1. Technical diff (D3 only)

### `src/d3_capture_core.py` (sha now `4e574172…`)

| Before (incident) | After (amendment A2, owner decision) |
|---|---|
| per reconstruction: `max_abs_diff ≤ 0.25` **and** `top1(recon) == top1(orig)` **and** same A/B decision | new pure function **`check_fidelity`**, applied separately to the pre-norm and post-norm reconstructions |
| — | (1) **max \|recon − orig\| over the full vocabulary ≤ 0.25** (unchanged) |
| — | (2) **identical A/B decision**: if the original is strict (A or B), the reconstruction must have the same strict sign |
| strict top-1 equality | (3) top-1 accepted iff `r1 == t1` **or** `orig[t1] − orig[r1] ≤ 2·max_abs_diff + τ`, **for any token** (not limited to the top-2); accepted mismatches are flagged `near_tie_top1` |
| — | **τ = 0.125, fixed before any GPU run.** It is the largest bf16 spacing (1 ULP) reachable by Gemma-4 logits under soft-cap 30 (\|x\| ∈ [16, 32)), i.e. one quantisation step of the model's own logits |
| assertion without world | `FidelityError("capture fidelity failed [<world_id>] (<reconstruction>): {…}")` |
| — | per-world record: `world_id`, original top-1, reconstructed top-1, margin, max diff, tolerance, `near_tie_top1`, `ab_tie_original` |
| — | run-level `fidelity_summary`: max diff per reconstruction, τ, limit, **list of every divergence** (near-tie top-1 or original A/B tie) |
| — | progress line every **400 observations** (with the count of divergences logged) |

- Conformity metrics (log p(A), log p(B), top-1, top-5, `top1_in_AB`) are **always computed from the original logits**. This was already the case; it is now tested.
- Guard and token-count errors also carry the `world_id`.

### `src/d3_run_gemma.py` (sha now `bb2145cf…`)

Diff against the frozen version: `outputs/logs/d3_amendment_A1_run_script.diff`.
- **A1** (run conditions of the first authorisation, kept): verify shard hashes and shapes **before** the ephemeral Volume is deleted; wait up to 300 s for the Volume commit; `scaledown_window=2`; record GPU memory, environment and timing.
- **A2:** run id `BTC-D3-DEV-R1`.

### Tests

- `tests_cpucheck/test_d3_fidelity.py` (new), 9 tests: constants (0.25, τ = 0.125); identical; **exact tie of two tokens**; **near tie of three tokens** (third-ranked token accepted, beyond top-2); **excessive logit divergence** (+0.30 aborts, message contains `[world_id]`); **A/B inversion** (aborts); **genuine reconstruction error** (random vector aborts); original A/B tie logged; mathematical check over 200 random cases.
- `tests_gemma/test_d3_capture.py` (updated): new fidelity fields; **conformity fields equal those recomputed from original logits**; `fidelity_summary` present with τ and limit; divergence count consistent.

## 2. Two points for the owner's validation

1. **Original exact A/B tie.** The rule "identical A/B decision" is undefined when the original logits of A and B are exactly equal in bf16. C1 had 2 such worlds in 1,320, so about 7 are expected in 4,800, and the strict implementation would abort on any of them. Implemented: for an original tie only, the reconstruction must also be numerically indiscernible, \|rA − rB\| ≤ 2·max_abs_diff + τ. Each case is logged (`ab_tie_original`). Admissibility is unchanged: computed from the original logits, so a tie is not admissible (C1 rule). **Please confirm or replace.**
2. **The top-1 rule is implied by rule (1).** If d is the maximum difference over the **whole** vocabulary and the reconstruction prefers r1 over t1, then orig[t1] − orig[r1] ≤ 2d (tested on 200 random cases). Rule (3) therefore can never be the binding constraint: it documents and logs near-ties. Genuine reconstruction errors are caught by rule (1) or (2), as the tests show. This follows from the owner's formula as specified; I report it so the rule's role is not overstated.

## 3. Verifications (all PASS)

| Check | Result |
|---|---|
| Tests | main **84/84** (`outputs/logs/r1_tests_main.log`); cpucheck **24/24** (`r1_tests_cpucheck.log`); gemma **5/5** (`r1_tests_gemma.log`) |
| Files changed vs freeze manifest | only `src/d3_capture_core.py`, `src/d3_run_gemma.py`, `tests_gemma/test_d3_capture.py` (+ new `tests_cpucheck/test_d3_fidelity.py`), recorded as amendment **D3-A2** in `reports/BTC-H1_FREEZE_MANIFEST.json` |
| H1 files | `h1_probe.py`, `h1_features.py`, `build_d5.py`, `sealed.py` and the H1/D5 tests are **unchanged**; the H1 document is unchanged (sha `09523e0f…`) |
| PROBE-DEV inputs | sha `a3c81436…` = manifest |
| LOCKED-TEST | seal intact (all hashes), read-only, **access log absent (never read)**; not referenced by any D3 code |
| Failed-run artefact | `remote_error.txt` moved to `outputs/logs/d3dev_attempt1_remote_error.txt`; `outputs/d3/probe_dev/` empty, so the run guard allows exactly one new capture |

## 4. Request (budget and stop conditions identical to BTC-D3-DEV)

```text
AUTORISATION DEMANDÉE — BTC-D3-DEV-R1
Objectif scientifique : capturer les activations pré-réponse de Gemma-4-31B-it sur PROBE-DEV (4 800 mondes) pour le développement des sondes H1 (protocole v0.3 gelé), avec le contrôle de fidélité corrigé (amendement D3-A2).
Why needed / alternative : la première exécution s'est arrêtée sur un contrôle top-1 trop strict (quasi-égalité bf16) ; aucune activation n'a été conservée ; aucune alternative CPU.
Exact change/command : modal run src/d3_run_gemma.py --pool PROBE-DEV — identique à BTC-D3-DEV sauf la règle de fidélité (max|Δ| ≤ 0,25 ; décision A/B identique ; top-1 égal ou marge ≤ 2·max|Δ| + τ, τ = 0,125 ; divergences journalisées avec world_id).
Compute and cost : 1 × A100-80GB, CPU 4/4, RAM 16/96 Gio, retries=0, max_containers=1, timeout 1 500 s, garde 1 420 s ; attendu ≈ 0,9–1,0 $ ; pire cas ≈ 1,69 $ ; objectif 2,00 $ (non garanti techniquement).
Risk and stop condition : arrêt sans relance sur toute erreur d'intégrité (empreintes, tokens, template, fidélité pré/post-normalisation selon A2, répétition bit à bit, visibilité ou vérification des shards avant suppression du Volume) ou garde 1 420 s ; coût relevé aussitôt ; ARRÊT après capture : aucun entraînement de sonde, aucun traitement de LOCKED-TEST, aucun D3-TEST.
I will not execute this step until you explicitly approve it.
```

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (H1 v0.3 freeze unchanged)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (LOCKED-TEST sealed, unread)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (CPU correction authorised; R1 requested, not run)
[YES] Preserved negative results and raw logs (attempt-1 error archived)
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: one rule extension for original exact A/B ties, submitted for validation (§2.1).
Next step: owner decision on §2 and on BTC-D3-DEV-R1.
```
