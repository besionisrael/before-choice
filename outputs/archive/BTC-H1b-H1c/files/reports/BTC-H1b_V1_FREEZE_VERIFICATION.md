# BTC-H1b v1.0 — freeze preparation: verification report

**Date:** 2026-11-14 · **Authorisation:** owner, "DÉCISION SCIENTIFIQUE — BTC-H1b v1.0" (CPU freeze preparation only) · **Status:** freeze artifacts prepared and hashed; **STOPPED.**

> **Not done:** LOCKED-TEST-H1b is not generated (`data/d5/locked_test_h1b/` absent), no GPU capture was run (`outputs/d3/locked_test_h1b/` absent), and there was no evaluation.
> **Unchanged:** LOCKED-TEST (consumed) was not touched, and the H1 v0.3 verdict stays NOT MET.

## Owner conditions

| # | Condition | Result |
|---|---|---|
| 1 | Train and check C_lin on DEV; keep the competitive variant | AdamW: DEV-val 0.717 / paired 0.584 (inner-val 0.725). L-BFGS λ = 0.01: 0.723 / 0.571 (inner-val 0.727). **Frozen variant: L-BFGS**, chosen on inner-val. Both variants are saved |
| 2 | C_max = max(C_bil, C_lin) inside every bootstrap resample | implemented in `h1b_eval_test.evaluate`; tested |
| 3 | Joint power with this exact rule, assumptions and adverse scenarios | `power_h1b_cmax.json`: with 1,000 families, joint power **0.82–0.96** when both contrasts are ≥ 0.05 (base, equal controls, low correlations, low admissibility); **0.32–0.48** if Δ_perm = 0.025; ≤ 0.043 under Δ_perm = 0. INSUFFICIENT risk flagged under low admissibility (about 900 expected contributing families vs the 875 threshold) |
| 4 | Elimination control via c2, trained on DEV only, not confirmatory | `P_c2_L40` frozen; S4 secondary. **DEV-val: elimination 0.883 / paired 0.818, which is above P (0.815 / 0.688)** |
| 5 | Freeze decoders, seeds, exclusions, statistics, scripts, hashes, protocol | `outputs/h1b_freeze/V1_MANIFEST.json` (sha `b4248392…`): 24 code files, 22 artifacts (9 decision decoders, statistics, power files, the control-C result, the PROBE-DEV capture manifest, positions, the protocol, test logs). All hashes re-verified; frozen files set read-only |
| — | Mandatory interpretation | stated in the protocol (header), the evaluator output and §1 of this report: MET ≠ an independent representation of the unchosen branch |

## Tests

| Suite | Result |
|---|---|
| cpucheck | **57/57**, including the new `test_h1b_v1.py`: tiny generator on another seed; generation refused without approval; evaluator with planted signals (MET, sensitivity gate, INSUFFICIENT, C_max rule, between-family permutation, frozen constants) |
| main | **93/94** (the only failure is the historical D3-A4 test, as documented), including the new static capture checks |
| gemma | 9/9 |
| post-evaluation | 4/4 |

Logs: `outputs/logs/h1b_v1_tests_*.log`, hashed in the manifest.

## Cost so far

**No cloud spend in this step** (BTC cumulative ≈ $3.24). The planned capture is ≈ $0.95 with a $1.50 cap, so the cumulative stays ≤ $4.74.

## Point for the owner before any generation

On DEV, chosen-branch information explains P's u2 decoding: the elimination control is stronger than P.
- As frozen, H1b compares P only with C_max and P_perm. A MET outcome would therefore most likely reflect chosen-branch information.
- **Option:** add or promote "P − elimination > 0" as an intersection-union contrast. This is the contrast that addresses the unchosen-branch question; on DEV it is unlikely to be met.
- If chosen, the freeze would be regenerated before the pool exists.

HORS PÉRIMÈTRE OU AUTORISATION REQUISE — je m'arrête ici et propose une décision au responsable scientifique.
