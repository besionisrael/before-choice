# BTC-D3-DEV-R1 — PROBE-DEV activation capture: **SUCCESS** (all integrity checks passed)

**Date:** 2026-11-13 · **Authorisation:** owner, BTC-D3-DEV-R1, single GPU execution · **Status:** captured, verified, **STOPPED**. No probe training, no D3-TEST, no access to LOCKED-TEST.

> This is a data-capture result. It contains **no H1, H2 or H3 evidence**: activations have not been analysed and no probe has been trained.

## 1. Pre-launch verifications (all PASS)

| Check | Result |
|---|---|
| Code | D3-A2 plus **D3-A3** (owner condition 4: reject non-finite logits and activations; added before launch, tested, recorded in the freeze manifest). `d3_capture_core.py` sha `00a3fa22…`, `d3_run_gemma.py` sha `bb2145cf…` |
| H1 freeze | document sha `09523e0f…` unchanged; `h1_probe.py`, `h1_features.py`, `build_d5.py`, `sealed.py` unchanged |
| PROBE-DEV inputs | `a3c81436…` = manifest |
| LOCKED-TEST | seal intact, read-only, **access log absent (never read)**, not referenced by D3 code |
| Tests | main 84/84, cpucheck 25/25 (incl. NaN/±inf rejection), gemma 5/5 |

## 2. Run

| Item | Value |
|---|---|
| Command | `modal run src/d3_run_gemma.py --pool PROBE-DEV` |
| Modal app | `ap-wPTKSUy2kXQTtegmMVWRma`, exit 0, stopped. Single attempt, `retries=0`, `max_containers=1`, `scaledown_window=2` |
| Model / env | Gemma-4-31B-it @ `842da379…`, bf16; NVIDIA A100-SXM4-80GB; torch 2.6.0+cu124, transformers 5.23.0, CUDA 12.4 |
| **GPU memory peak** | **63.35 GB** |
| Observations | 4,800 worlds + 20 bitwise replays = 4,820 |

**Timing:**

| Phase | Seconds |
|---|---|
| Model load | **100.2** |
| Capture (4,820 forwards with 61 hidden states + hook) | **590.3** (≈ 122 ms each) |
| Remote total | 690.6 (guard 1,420 s, timeout 1,500 s) |
| Return → files visible on the Volume | 11.1 |
| Download (3.43 GB) | 126.6 |
| Local wall time | 861.3 |

## 3. Fidelity statistics (corrected rule D3-A2/A3, applied separately to pre- and post-norm)

| Reconstruction | Max \|Δ logit\| over the vocabulary (worst world) | Median / p99 per world | Limit |
|---|---|---|---|
| post-norm ([60] → lm_head → soft-cap) | **0.1952** | 0.1304 / 0.1952 | 0.25 |
| pre-norm ([61] → norm → lm_head → soft-cap) | **0.2051** | 0.1304 / 0.1952 | 0.25 |

- **No world exceeded the tolerance.**
- **No non-finite value** in the logits, the 62 vectors or `mean_emb`, checked in the container and again locally.
- **No A/B inversion.**

**Logged divergences: 28 records = 14 distinct worlds × 2 reconstructions.** All are in `MANIFEST.json → fidelity_summary.divergences`, with `world_id`.

| Type | Worlds |
|---|---|
| Original exact A/B tie only (`ab_tie_original`), **non-admissible per H1 v0.3** | 10 |
| Top-1 near-tie only (`near_tie_top1`; diagnostic, implied by the full-vocabulary bound) | 3 |
| Both | 1 |
| **Total original A/B ties** (also counted from the index) | **11** |

**Replay:** the first 20 worlds were re-run and their 62 × 5,376 vectors are **bitwise identical**.

## 4. Verified files (`outputs/d3/probe_dev/`, 3,432,740,755 bytes, git-ignored)

| File(s) | Content | Verification |
|---|---|---|
| `shard_000…011_acts.npy` | 12 × (400, 62, 5,376) uint16 (bf16 words) | sha256 = manifest **before** Volume deletion, and again locally; shape and dtype checked |
| `shard_000…011_mean_emb.npy` | 12 × (400, 5,376) float32 | same |
| `shard_000…011_index.jsonl` | per world: tokens, log p(A/B) and top-1/top-5 from **original** logits, fidelity records | same; 4,800 unique `world_id`s, **in the exact order of the PROBE-DEV inputs** |
| `candidate_embeddings.npy` / `candidate_labels.json` | (5,800, 5,376) float32 / labels + token ids | sha256 = manifest; **5,800 labels = all labels in the PROBE-DEV prompts**; finite |
| `MANIFEST.json` | per-file sha256, fidelity summary, layout | sha `306573de…`; recorded in `reports/BTC-H1_FREEZE_MANIFEST.json → artifacts` |
| `d3_run_manifest.json` | timings, env, resources, verify result, actual cost, local post-check | — |
| `local_postcheck.json` | independent local re-check | — |

**Layout (unchanged):**
- `acts[i, 0..60]` are the HF hidden states: 0 = embeddings, 60 = **post-final-norm**.
- `acts[i, 61]` is the raw block-60 output **pre-norm**.

## 5. Descriptive behaviour fields (from original logits; not an H1 analysis)

- Top-1 ∈ {A, B} in **4,329/4,800 = 0.902** of DEV worlds.
- Of the 471 non-conformant worlds, **396 are decoy worlds**, matching the C1 pattern. The others are spread across variants: xswap 14, upath 13, cpath 9, base 8, alias 8, swap 8, shuffle 8, cf 7.
- Under H1 v0.3 (§3), format violations are counted and never used for inclusion. Decoy worlds are excluded from training. Admissibility uses the forced A/B choice from the original logits.

## 6. Cost (actual, `modal billing report --show-resources`)

| Resource | $ |
|---|---|
| A100-80GB | 0.5097 |
| CPU | 0.0386 |
| Memory | 0.0261 |
| Network egress | 0.0047 |
| **Total** | **$0.579** |

- Below the $1.69 worst case and the $2.00 objective. $0 charged (credits). The ephemeral Volume was deleted automatically.
- **Cumulative project cloud cost:** ≈ $1.75 (D2 ≈ $0.09; V2 ≈ $0.86 + image builds; D3-DEV failed attempt $0.17; D3-DEV-R1 $0.58).

## 7. Status and next decision

- PROBE-DEV activations exist and are integrity-verified. The H1 protocol v0.3 and its criteria are unchanged. LOCKED-TEST is sealed and unread.
- **Next decision (owner):**
  - validate the captured activations;
  - give a **separate authorisation** for probe training on DEV, under H1 v0.3 §3–5: grid of 2,130 fits, 48 CPU-hour budget, selection on DEV-val only.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (H1 v0.3 unchanged)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently (pinned 842da379…)
[YES] Preserved oracle isolation and train/val/test independence (oracle-free payload; LOCKED-TEST sealed, unread)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (BTC-D3-DEV-R1; D3-A3 implements condition 4)
[YES] Preserved negative results and raw logs
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner validation of the activations and a separate decision on DEV probe training.
```
