# BTC-D3_ACTIVATION_DESIGN — v0.2 (revised after owner review; NOT run)

**Date:** 2026-11-13 · **Status:** prepared and CPU-tested. **No GPU run, no activation captured, no probe.**
**Owner decisions applied (D3):**
- Keep the 61 HF hidden states plus the raw last-block vector.
- Also store the B1 mean embedding.
- Reproducible frozen candidate embeddings.
- Separate logit reconstruction from the pre-norm and post-norm vectors.
- Tests for capture, storage, download, hashes and Volume deletion.

**Model:** `google/gemma-4-31B-it` @ `842da3794eaa0b77d5f08bae87a17459d91ff475`, bf16; torch 2.6.0, transformers 5.23.0.

## 1. Captured per world (last pre-answer position, token `<channel|>`)

| Field | Shape / type | Content |
|---|---|---|
| `acts` | 62 × 5,376, **uint16 words of bf16** (lossless) | [0] scaled input embeddings; [1…59] outputs of blocks 1–59; **[60] block-60 output after the final norm** (HF `hidden_states[-1]`); **[61] raw block-60 output before the final norm** (forward hook) |
| `mean_emb` | 5,376, float32 | **mean over all prompt positions of `hidden_states[0]`**: the input for baseline B1. No other prompt positions are stored |
| index record | JSON | `world_id`, token count, log p(A), log p(B), full-vocabulary top-1 id and conformity, top-5, **fidelity** (below) |

**Per pool (once):** `candidate_embeddings.npy` (float32, n_labels × 5,376) and `candidate_labels.json` (labels and their token ids).
- **Definition** (frozen, context-free): e(c) is the mean of the **scaled input-embedding** rows (`embed_tokens` output, the same scaling as `hidden_states[0]`) of `encode(" " + c)`.
- **Label list:** parsed from the prompts on the local side (regex on rule and reward lines), so no oracle is involved.
- **Reproducibility:** e(c) depends only on the pinned weights and tokenizer. sha256 values are recorded in the manifest. Identical across calls (tested).
- **Validity:** on all 75,400 label occurrences in PROBE-DEV, the in-prompt token ids of a label are **identical** to `encode(" " + label)`, with aligned token boundaries (`outputs/v2_audit/d3_candidate_tokenization_check.json`). A first, looser check reported 72 %; that came from a span-selection bug in my audit script, corrected and re-run strictly.

## 2. Fidelity checks (in every world; abort on failure)

- **(a) Post-norm path:** `lm_head([60])`, then soft-cap.
- **(b) Pre-norm path:** `final_norm([61])`, then `lm_head`, then soft-cap.
- Each must reproduce the model's logits: same top-1 id, same A-vs-B decision, max |Δ logit| ≤ 0.25. The diffs are recorded per world.
- Also: bitwise replay of the first 20 worlds; payload integrity (fields, prompt hashes, digest, token counts vs local); A/B ids.

## 3. Storage, transfer, deletion

- **Shards** of 400 worlds: `shard_k_acts.npy` (uint16), `shard_k_mean_emb.npy` (float32), `shard_k_index.jsonl`. Every file is hashed into `MANIFEST.json`, together with the candidate files.
- They are written to an **ephemeral Modal Volume** (`modal.Volume.ephemeral()`), which **exists only inside the local `with` block and is deleted automatically when it exits, including on error**.
- **No named or persistent Volume** is used (tested).
- **Download:** streamed with `vol.read_file` into `outputs/d3/<pool>/`, then `verify_shards` re-hashes every file and checks shapes and dtypes. Any mismatch reports the run FAILED.
- **Size per pool** (4,800 worlds): activations 4,800 × 62 × 5,376 × 2 B ≈ **3.2 GB**, mean embeddings ≈ 0.1 GB, candidate table ≈ 0.15 GB.
- `outputs/d3/` is git-ignored.
- **Supported pools in code:** **PROBE-DEV only.** LOCKED-TEST capture will be added after the H1 freeze is approved; its outputs will go to a sealed directory (BTC-D5 R3).

## 4. CPU verification performed

**`tests_gemma/test_d3_capture.py`:** run with `.venv-gemma4`, tiny random model with the real Gemma-4 config structure, real tokenizer and template, real PROBE-DEV prompts. **5/5 OK.**

| Test | Verifies |
|---|---|
| `test_world_layout_and_fidelity` | shape (L+2) × d uint16; **post-norm and pre-norm logit reconstructions each pass**; stored bf16 equals HF hidden states exactly; `mean_emb` equals the mean of `hidden_states[0]`; [60] ≠ [61] |
| `test_candidate_embeddings_frozen_and_reproducible` | identical across calls; equals the manual mean of embedding rows |
| `test_run_capture_shards_hashes_replay` | shard sizes (5, 5, 2); bitwise replay; manifest hashes verified; **one flipped byte is detected** |
| `test_payload_integrity_enforced` | wrong digest, extra field (`oracle`) and wrong token count are all rejected |
| `test_shardwriter_roundtrip` | multi-shard write and verify |

**`tests/test_d3_run.py`** (main environment, static): the script imports without any remote call; timeout 1,500 s, guard 1,420 s, shard 400, replay 20; resources; `Volume.ephemeral` used, no `Volume.from_name`; no sealed path; worst case ≤ 85 % of the $2.00 objective.

**Earlier checks still valid** (`outputs/v2_audit/d3_capture_check.json`): hook equality on blocks 1…L−1; `hidden_states[L]` is post-final-norm; determinism.

## 5. Memory, time, cost (unchanged from v0.1, re-checked)

| Item | Value |
|---|---|
| GPU memory | 63.3 GB measured without hidden states (C1), plus ≈ 0.2 GB transient full-sequence states; fits in 80 GB |
| Host memory | one shard ≈ 0.27 GB, well under the 16 GiB request |
| Expected remote time | ≈ 850–960 s (load ~170 s + 4,820 forwards × ~0.13–0.15 s + writes) |
| Timeout / guard | 1,500 s / 1,420 s |
| Worst case | (1,500 + 60) × $0.00096 + $0.04 + egress ≈ $0.15 → **≈ $1.69** |
| Objective | **$2.00 per pool**. Not a technical guarantee: it relies on Modal honouring the timeout and published rates |

## 6. Decisions requested

| ID | Decision |
|---|---|
| D3-R1 | Accept the v0.2 capture specification (§1–3), including `mean_emb` for B1 and frozen candidate embeddings |
| D3-R2 | Later: authorise **D3-DEV** (PROBE-DEV capture, one GPU run, objective $2.00) as a separate request, after the H1 freeze is approved |

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (oracle-free payload; LOCKED-TEST unsupported in code)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (none executed)
[YES] Preserved negative results and raw logs
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none. My first tokenization audit had a bug (reported 72 %); corrected strict check = 100 %.
Next step: owner validation of D3-R1 together with the revised H1 freeze.
```
