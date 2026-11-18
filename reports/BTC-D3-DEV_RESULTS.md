# BTC-D3-DEV — INCIDENT REPORT: capture **FAILED** (integrity check stop; no activations retained)

**Date:** 2026-11-13 · **Authorisation:** owner, BTC-D3-DEV, single GPU execution · **Status:** **FAILED, stopped, no retry.** No probe training, no LOCKED-TEST processing, no D3-TEST.

## 1. Pre-launch verifications (all PASS)

| Check | Result |
|---|---|
| Code vs freeze manifest | 20 files checked. **Only `src/d3_run_gemma.py` and `tests/test_d3_run.py` differ**: the changes implementing the owner's run conditions (verify shards before Volume deletion; wait for the Volume commit; `scaledown_window=2`; record GPU memory and timing). Recorded as an amendment in `reports/BTC-H1_FREEZE_MANIFEST.json`, pre-amendment copy at `outputs/logs/d3_run_gemma_as_frozen.py`. No H1 criterion, probe, baseline or data changed |
| PROBE-DEV model inputs | sha `a3c81436…` = freeze manifest = D5 manifest |
| LOCKED-TEST | seal intact (all hashes), all files read-only, **access log absent (never read)**, `SEAL_MANIFEST.json` sha = freeze manifest; the D3 script does not reference the sealed directory and supports only PROBE-DEV |
| Final frozen H1 document sha256 (archived in the manifest) | `09523e0f230e6b74d6c1b28a934b366d66356b91b0a7a30d791eeb144cdb69c7` |
| Tests before launch | main 84/84 (`outputs/logs/d3dev_prelaunch_tests_main.log`); gemma 5/5; cpucheck 15/15 |

## 2. What happened

| Item | Value |
|---|---|
| Command | `modal run src/d3_run_gemma.py --pool PROBE-DEV` |
| Modal app | `ap-xkgw5RYfrUkLZ9iWXZNCh1`, stopped, 0 tasks running |
| UTC | 22:11:25 – 22:15:17 (local wall ≈ 3.9 min); image cached |
| Model load | 105.8 s |
| Stop | an in-container assertion: `capture fidelity failed (post_norm): max_abs_diff = 0.1409, top1_same = False, ab_decision_same = True` |
| Worlds processed before the stop | **not recorded.** The assertion message has no `world_id` and progress lines were not printed. Estimated from billed GPU time (≈ 212 s − 106 s of load − start-up) at ~127 ms per world: **about 600–750 worlds passed the check before one failed** |
| Data retained | **none.** The function raised before returning, so no shard was downloaded, and the ephemeral Volume was deleted automatically. `outputs/d3/probe_dev/` contains only `remote_error.txt` |
| Retry | none |

## 3. Root cause (diagnosed from the recorded values, not re-run)

- For one world, the logits re-derived from the **post-norm** vector `lm_head([60]) → soft-cap` differ from the model's logits by at most **0.14**, within the 0.25 tolerance. They give the **same A/B decision**, but a **different top-1 token** over the full vocabulary.
- With |Δ| ≤ 0.14, a top-1 flip is only possible if the model's two most probable tokens are within ~0.28 logit of each other: **a near tie that bf16 rounding can reorder.** The recomputation runs one vector through `lm_head` and the in-model computation runs the whole sequence, so the kernels and rounding differ.
- The captured representation itself is not implicated: the difference is within tolerance and the A/B decision is preserved.
- **The design flaw is mine:** I required exact top-1 equality on the 262,144-token vocabulary, a criterion that legitimate bf16 near-ties can fail. I also did not put the `world_id` in the assertion message. The CPU tests used a tiny random model, where such near-ties did not occur.

## 4. Cost and resources (`modal billing report --show-resources`)

| Resource | $ |
|---|---|
| A100-80GB | 0.1472 |
| CPU | 0.0111 |
| Memory | 0.0075 |
| Network egress | 0.0041 |
| **Total** | **$0.170** |

- Below the $1.69 worst case and the $2.00 objective. $0 charged (credits).
- GPU memory and per-phase timing were **not returned**, because the function failed before returning its manifest.

## 5. Proposed correction (NOT applied; requires owner approval before any new run)

1. **Fidelity criterion** (D3 design element, not an H1 criterion). Keep:
   - (a) max |Δ logit| ≤ 0.25, for both pre-norm and post-norm reconstructions;
   - (b) **identical A/B decision**, the quantity used for admissibility and H1.

   Replace the strict "identical top-1" with:
   - (c) the reconstructed top-1 is the model's top-1 **or**, when the model's top-2 logit margin is < 2 × max |Δ|, one of the model's top-2 tokens. Such worlds are counted and listed as `near_tie_top1`.

   Keep the full-vocabulary top-1 conformity field computed from the model's own logits, as in C1, so nothing changes for the format statistics.
2. **Diagnostics:**
   - include `world_id`, the model's top-2 margin and both reconstructions' top-1 in every fidelity record and assertion message;
   - print progress every 400 worlds.
3. **CPU tests** to add before any rerun:
   - a synthetic near-tie case passes (c) and is counted;
   - a genuine mismatch (|Δ| > 0.25, A/B flip, or top-1 outside top-2) still aborts.
4. **Rerun request** (separate approval): identical command, resources and limits; same objective $2.00. Expected cost ≈ $0.9–1.0.

## 6. Status

- No activations exist for PROBE-DEV. The H1 protocol, its data (PROBE-DEV, sealed LOCKED-TEST) and all criteria are unchanged.
- **Cumulative cloud cost of the project** ≈ $1.17 (D2 ≈ $0.09, V2 ≈ $0.86 + image builds, D3-DEV $0.17).

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (unchanged)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (LOCKED-TEST sealed, unread)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (BTC-D3-DEV; run conditions amendment recorded)
[YES] Preserved negative results and raw logs (outputs/logs/d3dev_*, remote_error.txt)
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none beyond the recorded run-condition amendment of d3_run_gemma.py.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decision on the fidelity-criterion correction (§5) and on a separate rerun authorisation.
```
