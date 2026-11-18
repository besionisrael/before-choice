# BTC-V2-PROTOCOL_FREEZE — DRAFT v1.0 (awaiting owner validation)

**Date:** 2026-11-13 · **Status:** draft for validation. **No GPU run is authorised by this document.**
**Basis:**
- `reports/BTC-V2_DESIGN_REVIEW.md`, with owner decisions V2-1, V2-2 (in principle), V2-3, V2-5 and V2-7 approved.
- V2-4 and V2-6 are corrected here (§5, §7).
- PROTOCOL_FREEZE v0.3 remains in force for everything not restated here: H1–H3, D-1′, evidential ladder L1–L4, P0, reporting §6b and failure criteria §7.

## 0. Invariants (unchanged)

- **Research question and hypotheses H1–H3: unchanged.** The H1 primary endpoint is D-1′ (unchosen depth-2 state under choice-preserving `upath` pairs, depth-4 primary stratum).
- **Evidential ladder:**
  - L1: decodable relational information.
  - L2: prospective path representation.
  - L3: functional prospective computation.
  - L4: sequential forward sweep, **UNTESTED by design**.
- **Task:**
  - Same symbolic two-branch maze generator.
  - Same prompt template (`oracle.format_prompt`), byte-identical in structure to v1.
  - Depths 2–4. D-7 (depths 5–6) stays deferred until competence results.
- **Decision format:** a single first answer token, `A` vs `B`, with **no textual reasoning**. Each model uses its official chat template in non-thinking mode.
- **The only factor changed relative to v1 is the model.**

## 1. Models (pinned)

| ID | Repo @ revision | Image / stack | Non-thinking format (template kwargs) |
|---|---|---|---|
| M1 | `Qwen/Qwen3-32B` @ `9216db5781bf21249d130ec9da846c4624c16137` | torch 2.5.1, transformers 4.57.1 (as D2) | `enable_thinking=False` (see §5.2) |
| M2 | `allenai/Olmo-3.1-32B-Instruct` @ `ac0587e4a7744a551c059d8cd17ba220bc940dae` | torch 2.5.1, transformers 4.57.1 | official template (injects default system prompt; kept, documented) |
| M3 | `google/gemma-4-31B-it` @ `842da3794eaa0b77d5f08bae87a17459d91ff475` | **separate image, transformers 5.x** (exact version pinned in its run request; V2-2 approved in principle only) | `enable_thinking=False` (empty thought channel) |

Weights are bf16. Inference uses `eval()`, `no_grad`, a single forward pass, and no generation.

## 2. Data (generated 2026-11-13, CPU, owner-authorised)

Generator `src/build_v2.py`, master seed **2026111302**. Same family builder and path-counterfactual code as v1.

| Pool | Families | Worlds | Variants | sha256 (data) | sha256 (model inputs) |
|---|---|---|---|---|---|
| **SCREEN** | 60 (20/depth; optimal A/B 10/10 per depth) | 240 | base, cf, swap_base, swap_cf | `cd3499d67f3be6fe03ee936f0c607134b5e14ae7b201ed26441b5c995eda34cd` | `42cc0517246c101e676d517066001e359fe83d301149970b8e940f6602b5dc20` |
| **COMPETENCE** | 90 (30/depth; 15/15) | 1,320 | 10 v1 variants + upath/cpath (d3–4) + xswap | `b1df46cd67e87e938a9ace023f4c46dccac154968b160467113e9cecf4e245d6` | `dc6c7115bcb2b15e21af952ca623562ded6708b5212d9f0df75a20d836fea0da` |
| PROBE-DEV | — | — | **not generated** (reserved family ids ≥ F20000, separate seed, only after competence passes and D-5 approval) | — | — |
| LOCKED-TEST | — | — | **not generated** (same reservation) | — | — |

**Audits:** `tests/test_v2_dataset.py`, 15/15 pass.
- **Determinism:** rebuild is identical; files equal the generator output; manifest hashes match; LF-normalised writer.
- **Ground truth:** independent DFS oracle on all 1,560 worlds, 0 violations.
- **Prompts:** every prompt equals `format_prompt(world)`; prompt hashes are correct; model-input files contain no oracle or reward field.
- **Isolation:**
  - Family ids are F10000–F10149, with no overlap with v1 (F0000–F0029).
  - No family appears in both pools (60 SCREEN / 90 COMPETENCE).
  - Every state label belongs to exactly one family.
  - **0 of the 480 v1 labels are reused.** For this exclusion, v1 is regenerated in memory from its own seed; the builder opens no v1 file and the v1 locked test is never read.
- **Balance:** optimal A/B exactly balanced per depth in each pool.
- **Variants:**
  - base/cf-type pairs flip the optimum, with identical rewards, prompt length and 2 differing lines.
  - swap pairs differ only in the action letters.
  - shuffle has the same edge set; alias labels are disjoint; decoy is unreachable with maximal reward and identical within its pair.
  - Path counterfactuals preserve the choice: 120 upath, 120 cpath, 180 xswap. upath changes the non-optimal depth-2 state and leaves the optimal path intact.

**Tokenisation audit** (`src/v2_token_audit.py` → `outputs/v2_audit/token_audit.json`; tokenizer/template files only):

| | M1 Qwen3-32B | M2 Olmo-3.1-32B | M3 Gemma-4-31B |
|---|---|---|---|
| `A` / `B` ids (single tokens) | 32 / 33 | 32 / 33 | 236776 / 236799 |
| Pre-answer tail (last tokens) | `assistant Ċ <think> ĊĊ </think> ĊĊ` | `<\|im_start\|> assistant Ċ` | `model \n <\|channel> thought \n <channel\|>` |
| Prompt tokens (min–max) | 121–271 | 175–320 | 128–285 |
| Minimal pairs with unequal token length (of 1,140) | **0** | **0** | **0** |

**Data-use rules:**
- SCREEN: once per candidate, for model selection only.
- COMPETENCE: exactly once, for the selected model only. Burnt if it fails.
- v1 train/val: pipeline debugging only. The v1 test is retired and never opened.

## 3. Scoring rules (common to S and C)

- First-answer-token log-probabilities of the model's `A` and `B` token ids at the last template position (§2 tails).
- **Ties:** a tie is `log p(A) == log p(B)` exactly. Primary score counts a tie as 0.5. Strict (0) and lenient (1) bounds are always reported. Same rule as BTC-D2-002.
- **CIs:** 95 % percentile bootstrap over families, 10,000 draws, seed 0.
- Reported per model, never pooled across models. Chosen and unchosen branches are always reported separately (v0.3 §6b).

## 4. Stage S — screening (V2-4 as approved, plus format correction)

**Order fixed:** M1 → M2 → M3, **all screened** (no optional stopping). M3 is screened only if its stack is approved; otherwise this is recorded.

| ID | Criterion (SCREEN, per model) |
|---|---|
| S1 | depth-2 base+cf primary accuracy ≥ 0.70 (n = 40 worlds, 20 families) |
| S2 | depth-2 base/cf pairs both correct ≥ 0.50 (chance 0.25) |
| S3 | A+B probability mass ≥ 0.5 in ≥ 95 % of worlds |
| **S4 (new)** | **top-1 next token ∈ {`A`, `B`} in ≥ 95 % of worlds** (conformity threshold) |

- **Pass = S1 ∧ S2 ∧ S3 ∧ S4.**
- **Selection:** among passers, the highest depth-4 base+cf primary accuracy on SCREEN. Tie-break: lower estimated cost, then order.
- **Stop:** no passer → STOP V2. Report all profiles. No new model, prompt or format without an owner scientific decision.
- **Also reported:** depth 3/4, swap consistency, A/B bias, ties, confident errors, top-1 token distribution.

## 5. Format and template verification (V2-4 correction)

### 5.1 Top-1 conformity

S4 above, and C3b below. A model that puts most of its first-token probability elsewhere (e.g. the `Let` behaviour seen in 9/352 Qwen3-4B worlds) fails conformity **regardless of accuracy**. Argmax(A, B) scoring is still computed and reported, but it cannot rescue a conformity failure.

### 5.2 Qwen3-32B chat template, verified at the pinned revision (documented before any GPU run)

- Source: `tokenizer_config.json` at `9216db57…`.
  - File sha256: `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101`.
  - Template sha256: `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`.
  - Saved copy: `outputs/v2_audit/qwen3_32b_chat_template.jinja`.
- Generation-prompt logic (verbatim):
  ```jinja
  {%- if add_generation_prompt %}
      {{- '<|im_start|>assistant\n' }}
      {%- if enable_thinking is defined and enable_thinking is false %}
          {{- '<think>\n\n</think>\n\n' }}
      {%- endif %}
  {%- endif %}
  ```
- Rendered with `add_generation_prompt=True`:

  | kwargs | String ending | Last tokens |
  |---|---|---|
  | none | `…<\|im_start\|>assistant\n` | `<\|im_start\|> assistant Ċ` → the model would be free to open `<think>` and reason |
  | `enable_thinking=True` | identical to none | same |
  | **`enable_thinking=False`** | `…assistant\n<think>\n\n</think>\n\n` | `<think> ĊĊ </think> ĊĊ` |

- **Rule:**
  - M1 is always called with `enable_thinking=False`, asserted in code: the rendered string must end with `<think>\n\n</think>\n\n`.
  - The pre-answer position is the final `ĊĊ` token after `</think>`. This empty block is part of the official non-thinking format.
  - This differs from Qwen3-4B-Instruct-2507, which has no think block. It is a documented template difference, not a protocol change.
- M2 has no thinking switch. M3 with `enable_thinking=False` ends with an empty thought channel. Each run asserts its expected tail (§2 table).

## 6. Stage C — competence checkpoint (V2-5 as approved, plus format correction)

Selected model only, COMPETENCE pool, run once.

| ID | Criterion | Role |
|---|---|---|
| C1 | depth-2 base+cf primary accuracy ≥ 0.70 **and** family-bootstrap 95 % lower bound > 0.50 (n = 60, 30 families) | competence |
| C2 | depth-2 base/cf both correct ≥ 0.50 | competence |
| C3a | A+B mass ≥ 0.5 in ≥ 95 % of worlds | competence (format) |
| **C3b (new)** | **top-1 ∈ {A, B} in ≥ 95 % of worlds** | competence (format) |
| C4 | depth-4 base+cf primary accuracy ≥ 0.70 | D-1′ feasibility |
| C5 | admissible depth-4 `upath` pairs ≥ 50 % of constructed (60) | D-1′ feasibility |

**Decision table:**
- Any of C1, C2, C3a or C3b fails → **STOP**. Competence not met; D3+ stay stopped; COMPETENCE is burnt.
- C1–C3 pass, but C4 or C5 fails → shallow competence only. **D-1′ primary is infeasible** with this model. Report and STOP; the owner decides.
- All pass → request D-5 (generate PROBE-DEV / LOCKED-TEST) and then D3, each separately approved.

## 7. Compute budget (V2-6 correction)

### 7.1 Calibration (measured, `modal billing report --show-resources`, 2026-11-13)

| Run | GPU s billed (cost ÷ L4 rate) | Remote function time | CPU | Memory | Egress | Total |
|---|---|---|---|---|---|---|
| D2-001b | 56 s | 46 s | $0.0025 | $0.0002 | $0.0068 | $0.0220 |
| D2-002 | 78 s | ~70 s | $0.0029 | $0.0003 | $0.0067 | $0.0272 |

**Observations:**
- GPU billing ≈ remote function time + ~10 s, covering container start and stop.
- The ~20 min allocation wait was **not** billed.
- Egress (~$0.007/run) and image builds (~$0.01 each) are small but non-zero.

### 7.2 Rates and worst-case assumptions (Modal list prices, 2026-11-13)

| Resource | Rate | Worst-case assumption | $/s |
|---|---|---|---|
| A100 80 GB | $0.000694/s | 1 GPU | 0.000694 |
| CPU | $0.0000131/core/s | request = limit = 4 cores (`cpu=(4, 4)`) | 0.0000524 |
| Memory | $0.00000222/GiB/s | limit 96 GiB, billed at the limit in the worst case (`memory=(16384, 98304)` MiB) | 0.000213 |
| **Worst-case total** | | | **0.000960** |

**Fixed overheads per run** (worst case): container start/stop 60 s, egress $0.02, image build $0.02.

Weights are loaded directly to the GPU (`device_map="cuda"`, safetensors), so expected RAM is far below the 96 GiB limit.

### 7.3 Caps, timeouts and margins

| Run | Spending objective (owner) | Timeout / internal guard | Worst case = (timeout + 60 s) × 0.000960 + $0.04 | Safety margin vs objective | Expected time | Expected cost |
|---|---|---|---|---|---|---|
| Screening (each, 240 worlds) | **$1.00** | **780 s / 720 s** | **$0.85** | 15 % | ~430 s: ~310 s download of ~65 GB at the measured ~210 MB/s, 30–60 s to GPU, ~60 s of forwards | ≈ $0.40 |
| Competence (1,320 worlds) | **$1.50** | **1,240 s / 1,160 s** | **$1.29** | 14 % | ~700 s: ~370 s load + ~330 s of forwards at an estimated 0.25 s/world | ≈ $0.61 |
| Programme (3 S + 1 C) | $4.50 | — | **$3.83** | 15 % | | ≈ $1.8 |

The previous design review proposed timeouts of 1,800 s and 2,400 s, which allowed worst cases above the stated caps. **They are replaced by the values above.**

**What these numbers are and are not:**
- They are **spending objectives with explicit safety margins**, derived from list prices, measured overheads and the timeouts.
- They are **not demonstrated technical guarantees.** Enforcement rests on three assumptions:
  1. Modal terminates the function at `timeout`.
  2. Billing follows the published per-second rates and the requested CPU/RAM limits.
  3. No unexpected billable resource appears.
- The workspace **$5 spend limit applies to charges after credits are exhausted**. With ~$12.9 of credits left this month, it would **not** stop credit consumption. It is not a per-run guard.
- Mitigations:
  - Every run is individually approved.
  - `retries=0`, `max_containers=1`, no Volume, no endpoint.
  - Each run's actual cost is read from `modal billing report --show-resources` immediately after it and recorded.
  - If any run's billed cost exceeds its worst-case estimate, **all further V2 runs stop** pending owner review.
- **Timeout risk:** if the weight download is much slower than measured (< ~110 MB/s), a run may hit the timeout and produce no result. This costs at most the worst case and is reported as FAILED, not retried.

## 8. Not measured / limitations

- Forward time per world at 32B is estimated, not measured.
- Allocation wait is unpredictable and unbilled.
- Template differences across models (empty think block, injected system prompt, thought channel) make the models differ in more than capability. Results are per model.
- If every candidate fails S or C, V2 cannot separate insufficient capability from a single-forward-pass limit of the format. Changing the format is outside V2.
- PROBE-DEV and LOCKED-TEST do not exist yet. No H1 analysis is possible before C passes.

## 9. Runs that would follow (each requires its own explicit authorisation)

| Run ID | Model | Data | GPU | Timeout | Objective cap |
|---|---|---|---|---|---|
| BTC-V2-S1 | M1 Qwen3-32B | SCREEN (240) | 1 × A100-80 | 780 s | $1.00 |
| BTC-V2-S2 | M2 Olmo-3.1-32B | SCREEN | 1 × A100-80 | 780 s | $1.00 |
| BTC-V2-S3 | M3 Gemma-4-31B (if stack approved) | SCREEN | 1 × A100-80 | 780 s | $1.00 |
| BTC-V2-C1 | selected model | COMPETENCE (1,320) | 1 × A100-80 | 1,240 s | $1.50 |

The run scripts are not yet written. They will reuse the validated BTC-D2-002 structure (JSON-only return, assertions, no retries), with per-model template and tail assertions and the CPU/RAM limits above. Scripts and tests will be prepared on CPU and submitted with the S1 request.

## 10. Decisions requested

| ID | Decision |
|---|---|
| F-1 | Validate S4 / C3b (top-1 ∈ {A, B} ≥ 95 %) and the Qwen3-32B template rule (§5.2) |
| F-2 | Validate the corrected budget (§7): timeouts 780 s / 1,240 s, CPU/RAM limits, stop-on-overrun rule; caps presented as objectives, not guarantees |
| F-3 | Validate the V2 data as generated (§2 hashes) and the data-use rules |
| F-4 | Confirm the M3 stack decision (transformers 5.x image) or drop M3 |
| F-5 | Authorise preparation (CPU only) of the S/C run scripts and tests |

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (unchanged)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently (candidates pinned; no run)
[YES] Preserved oracle isolation and train/val/test independence (fresh sealed pools; v1 test unopened)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (no GPU run; data generation owner-authorised)
[YES] Preserved negative results and raw logs
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none. PROBE-DEV / LOCKED-TEST not generated, as instructed.
Next step: owner validation of F-1 … F-5.
```
