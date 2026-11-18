# BTC-D3-TEST-R1 — LOCKED-TEST activation capture: **SUCCESS, sealed**

**Date:** 2026-11-14 · **Authorisation:** owner, "AUTORISATION SCIENTIFIQUE — BTC-D3-TEST-R1" (one attempt; capture and sealing only) · **Status:** completed, **STOPPED.**

> **Scope:** capture and sealing only. The oracle (`worlds.jsonl`) was not read, the activations were not read after sealing, no score was computed, and nothing was trained. The predictors and H1 criteria are unchanged.

## 1. Pre-launch conditions: all PASS

The checks are in `outputs/d3/d3_test_r1_prelaunch.json`.

| # | Condition | Result |
|---|---|---|
| 1 | Fix `d7936624…`, frozen D3 core, model | Fix sha `d793662422f3…`, equal to the prepared R1 version. `d3_capture_core` `00a3fa22…`, `v2_common`, `v2_local` and `sealed` all frozen. `google/gemma-4-31B-it` @ `842da3794eaa0b77d5f08bae87a17459d91ff475` |
| 2 | Import from a path equivalent to `/root/d3_run_gemma_test.py` | A copy outside the project imports cleanly with `ROOT = None`. `/root/<name>.py` has 2 parents, which the guard handles. The regression test passes |
| 3 | LOCKED-TEST seal; attempt-1 traces kept | Pool seal intact. Access log unchanged since attempt 1 (sha `bcd1c64f…`). The attempt-1 log is archived read-only (`outputs/logs/d3test_attempt1_run.log`, original kept) |
| 4 | R1 authorisation recorded without overwriting | The attempt-1 unlock is archived unchanged and read-only (`history/UNLOCK_LOCKED-TEST-D5.attempt1.json`, sha `6f66ce34…`). The new unlock carries the R1 approval id and embeds the previous approval (`previous_approvals`) |
| 5 | Only `model_inputs.jsonl` in the D3-TEST-CAPTURE scope | Enforced by `sealed_access`. `worlds.jsonl` and `manifest.json` are refused |
| 9 | Startup failure leads to an immediate stop | A watchdog (`outputs/logs/d3test_r1_watchdog.sh`) was armed to run `modal app stop --yes` at the first startup failure. It was not triggered |

## 2. Run

| Item | Value |
|---|---|
| Command | `modal run src/sealed_stage/d3_run_gemma_test.py`, app `ap-e6GnzMcEj5dOt6gVMmUiod` |
| UTC | 13:35:44 → 13:53:51 (**17 min 27 s** local wall) |
| Remote | model load 244.7 s (DEV: 100 s); capture 632.6 s for 4,800 worlds + 20 replays + candidate embeddings; remote total 877.4 s, under the 1,420 s guard and the 1,500 s timeout |
| Transfer | commit visible after 10.9 s; download 125.8 s; 3.43 GB |
| Resources | 1 × NVIDIA A100-SXM4-80GB, cpu (4, 4), memory 16–96 GiB, retries = 0, max_containers = 1; peak GPU memory 63.35 GB |
| Environment | torch 2.6.0+cu124, transformers 5.23.0, CUDA 12.4 |

## 3. Integrity of the 4,800 worlds

| Check | Result |
|---|---|
| Input | `model_inputs.jsonl` sha = frozen `8fbc920d…`; 4,800 unique ids; schema identical to PROBE-DEV; every prompt hash recomputed |
| Shards (`verify_shards`, **before** Volume deletion) | sha256 and shapes OK for all 12 shards; 4,800 worlds; acts uint16 (bf16), `mean_emb` float32 |
| Completeness | captured world ids = input world ids (4,800/4,800) |
| Candidate embeddings | 5,800 labels (TEST label set), hashes verified with the shards |
| Token counts | asserted per world against the local tokenizer (in container); no failure |
| Non-finite values | none: rejected in container (D3-A3), and the run would have aborted |
| Replays | 20 replays **bitwise identical**: asserted in container, and the run would have aborted otherwise |

## 4. Fidelity (D3-A2 / D3-A3)

| Statistic | TEST (R1) | DEV (R1) for reference |
|---|---|---|
| max\|Δ\| post-norm logits (full vocabulary) | **0.1952** | 0.195 |
| max\|Δ\| pre-norm logits | **0.1952** | 0.205 |
| Limit / τ | 0.25 / 0.125 | same |
| Logged divergence entries | **32**, from 16 worlds × 2 reconstructions (pre and post) | 15 worlds (11 original A/B ties, 4 near-ties) |

**What the 32 entries are**:
- **9 worlds** have an **exact original A/B tie** (logp_A = logp_B in the model itself). 4 of them also have a top-1 near-tie.
- **7 worlds** have only a top-1 near-tie: the full-vocabulary top-1 differs within the 2d + τ tolerance (largest tolerance 0.505; largest |Δ| among them 0.190).
- All are within the approved rule; none triggered a failure.
- **Consequence for the frozen analysis:** under H1 v0.3, a world whose original A/B choice is an exact tie is **not admissible**. This rule is applied by the evaluator, not here. I have not looked at which variants these 9 worlds belong to.

## 5. Sealing and access logs

| Item | Value |
|---|---|
| Activation seal | `LOCKED-TEST-D3-ACT`, sealed 2026-11-14T13:53:47Z, **40 files**, all read-only; `hash_only` all True; `SEAL_MANIFEST.json` sha `fe7c5756…` |
| Activation access log | **absent** (never read since sealing) |
| Integrity summary outside the seal | `outputs/d3/d3_test_capture_summary.json` (sha `7956f521…`): counts, fidelity, environment, timing. No per-world content beyond the divergence ids |
| LOCKED-TEST pool | seal intact. **Access log: 2 entries, both `model_inputs.jsonl`, scope D3-TEST-CAPTURE**: attempt 1 (06:00:20) and R1 (13:35:47), each with its own approval id. **No read outside the capture scope**; `worlds.jsonl` never read |
| Amended seal test (D3-A4) | **PASS** with this log |
| Tests after capture | main **89/89**, cpucheck **44/44** (`outputs/logs/d3test_r1_post_tests_*.log`) |
| Manifest | `BTC-H1_FREEZE_MANIFEST.json → artifacts.D3-TEST-R1_activations` (hashes of seal, summary, logs, unlocks, cost) |

## 6. Billing (`modal billing report --for today --show-resources`)

| App | A100-80GB | CPU | Memory | Network | **Total** |
|---|---|---|---|---|---|
| R1 `ap-e6GnzM…` | 0.6250 | 0.0473 | 0.0320 | 0.0414 | **$0.746** |
| Attempt 1 `ap-GmiXKv…` (reconfirmed more than 7 h later) | 0 | 0 | 0 | — | **$0.00** |

- Within the $2 objective.
- Cumulative cloud spend is about **$2.50**, under the $5 limit.
- R1 cost more than DEV-R1 ($0.579) mainly because the model took longer to load (245 s vs 100 s).

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (H1 v0.3 unchanged)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (prompts only; oracle never read; activations sealed unread)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (BTC-D3-TEST-R1, single attempt)
[YES] Preserved negative results and raw logs (attempt-1 log, unlock and access entries kept)
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decision on BTC-H1-TEST-EVAL (separate request).
```
