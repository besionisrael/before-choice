# BTC-V2_DESIGN_REVIEW — model escalation after the BTC-D2-002 competence failure

**Status:** Owner decision 2026-11-13: V2-1, V2-2 (in principle), V2-3, V2-5 and V2-7 approved. **V2-4 and V2-6 corrected in `reports/BTC-V2-PROTOCOL_FREEZE.md` (§5, §7), which supersedes §2 timeouts/caps and §5 of this review.** Originally a PROPOSAL for owner scientific validation. **Nothing in this document has been run on GPU.** No dataset has been generated for V2.
**Date:** 2026-11-13 · **Preceding result:** BTC-D2-002 pre-registered FAIL. Qwen3-4B-Instruct-2507, depth-2 base+cf accuracy 0.5625 < 0.70; the choice changed in 1/24 base/cf pairs; D3–D6 stopped.

**Fixed by the owner for V2:**
- Same symbolic maze task.
- Same A/B single-token decision with no textual reasoning.
- H1–H3 unchanged.
- No GPU run without per-run approval.

## 0. Scope statement

V2 changes **one factor only: the model**, to test whether the competence precondition can be met by a more capable open-weight model.

Unchanged:
- Task family, prompt template (`oracle.format_prompt`), answer format ("Respond with A or B only"), first-answer-token scoring, the H1 primary endpoint D-1′, controls, and the evidential ladder L1–L4.
- H1–H3 are not modified.

A scale-only change has a known limit. If larger models also fail, V2 cannot tell "insufficient capability" apart from "single-forward-pass multi-hop composition is not available in this format". That ambiguity would be reported, not resolved by changing the format (§8).

## 1. Candidate models (facts verified 2026-11-13)

Sources:
- Hugging Face API: `model_info`, `config.json` at the listed revisions.
- Local CPU tokenizer checks, tokenizer/config files only.
- modal.com/pricing.

Gemma 4 could not be loaded by our pinned `transformers==4.57.1`; I parsed its `tokenizer.json` and `chat_template.jinja` directly.

| | **M1 Qwen3-32B** | **M2 Olmo-3.1-32B-Instruct** | **M3 Gemma-4-31B-it** |
|---|---|---|---|
| HF repo @ revision | `Qwen/Qwen3-32B` @ `9216db5781bf21249d130ec9da846c4624c16137` | `allenai/Olmo-3.1-32B-Instruct` @ `ac0587e4a7744a551c059d8cd17ba220bc940dae` | `google/gemma-4-31B-it` @ `842da3794eaa0b77d5f08bae87a17459d91ff475` |
| Licence / gating | Apache-2.0, ungated | Apache-2.0, ungated | Apache-2.0, ungated |
| Architecture | dense decoder, `Qwen3ForCausalLM`, full attention | dense decoder, `Olmo3ForCausalLM`; 48 sliding (window 4096) + 16 full | dense, `Gemma4ForConditionalGeneration` (multimodal wrapper, text path only); 50 sliding (window 1024) + 10 full; final logit soft-cap 30 |
| Layers × d_model | 64 × 5120 | 64 × 5120 | 60 × 5376 |
| Weights (bf16 safetensors) | 65.5 GB | 64.5 GB | 62.5 GB (incl. vision tower) |
| `A` / `B` single tokens | yes (32 / 33, same ids as baseline) | yes (32 / 33) | yes (236776 / 236799) |
| Non-thinking A/B format | `enable_thinking=False` → template appends an empty `<think>\n\n</think>\n\n` | no thinking mode; **template injects a default system prompt** ("You are Olmo…") | `enable_thinking=False` → empty thought channel `<\|channel>thought\n<channel\|>` |
| Prompt length (F0000-base) | 131 tokens | 183 tokens (system prompt) | 140 tokens |
| Supported by our stack (transformers 4.57.1) | **yes** | **yes** | **no**: needs transformers 5.x (latest 5.23.0), i.e. a stack change |
| Sliding window vs our prompts (≤ ~300 tokens) | n/a | 4096 > prompt → effectively full attention | 1024 > prompt → effectively full attention |
| Published interpretability resources (HF search) | community SAEs (`adamkarvonen/qwen3-32b-saes`, `bcywinski/qwen3-32b-saes`; quality unverified) | none found for 32B (fully open data and training make contamination checks possible) | none found for Gemma 4 31B |
| Why include | same lineage, tokenizer and token ids as the failed baseline → **cleanest scale contrast** | **fully open** model (data, code, checkpoints): best for reproducibility and later auditing | **different lab/architecture**: tests that a result is not Qwen-specific |
| Main risk | hybrid-thinking checkpoint run in non-thinking mode: the answer follows an empty think block, unlike 2507-Instruct | default system prompt changes the input (kept as the official template; documented) | stack change (transformers 5.x) needs approval; multimodal wrapper; soft-capped logits (ties less likely, margins compressed) |

**Considered and not recommended for V2:**

| Model | Reason |
|---|---|
| Qwen3.8-27B / Qwen3.6-27B / Qwen3.5-27B | Newest Qwen, but **hybrid linear attention** (48 Gated-DeltaNet-style + 16 full layers per config). A recurrent state changes what "pre-answer residual" and patching mean, and the model needs transformers 5.x. Better kept for a later, separately justified study. |
| Llama-3.3-70B-Instruct | Gated licence (owner account action), 141 GB → 2 × A100-80 or larger, ≈ 2–3× cost |
| Qwen3-30B-A3B-Instruct-2507 | MoE (128 experts, d_model 2048); routing complicates interventions (H3) |
| Qwen3-14B | Cheap (29.5 GB, L40S), same lineage; may not be "capable enough". **Fallback only** if the owner prefers a lower-cost first step. |

## 2. Compute, GPU and cost estimates (Modal list prices, 2026-11-13)

| GPU | $/s | $/h | Fit for 62–66 GB bf16 |
|---|---|---|---|
| A100 80 GB | 0.000694 | 2.50 | yes (~14 GB headroom; prompts ≤ 300 tokens, no generation) |
| RTX PRO 6000 (96 GB) | 0.000842 | 3.03 | yes (more headroom) |
| H100 80 GB | 0.001097 | 3.95 | yes (faster, not needed) |

Memory is billed at $0.00000222/GiB/s and CPU at $0.0000131/core/s. Volumes are $0.09/GiB-month; the page footnote mentions a free allowance, which I do not rely on.

**Recommended GPU: 1 × A100 80 GB.** Concurrency 1, `retries=0`, no Volume (weights downloaded into the ephemeral container, as in BTC-D2-001b/002).

**Per-run estimate**, extrapolated from BTC-D2-002's measured throughput (8.06 GB fetched + loaded in ~40 s ≈ 200 MB/s; 67 ms per forward at 4B):

| Phase | Screening run (240 worlds) | Competence run (~1,320 worlds) |
|---|---|---|
| Weight download + load (~65 GB) | ~5–7 min | ~5–7 min |
| Forwards (est. ~0.15–0.3 s each at 32B) | ~1–2 min | ~4–7 min |
| GPU time | ~7–9 min | ~10–14 min |
| **Estimated cost** (GPU + CPU/RAM) | **≈ $0.35–0.50** | **≈ $0.50–0.75** |
| Proposed hard cap | **$1.00** | **$1.50** |
| Timeout / internal guard | 1,800 s / 1,620 s | 2,400 s / 2,200 s |

**Notes:**
- GPU allocation wait was ~20 min for L4 in our runs and is not billed, as far as observed. A100 availability is not predictable.
- Whole V2 programme: 3 screening runs plus 1 competence run ≈ **$1.5–2.3 expected**, with a hard cap of **$4.50**.
- That cap is within the existing $5 workspace spend limit. About $12.9 of this month's $30 Starter credits remain, so no charge is expected.
- Each run still needs its own approval.

## 3. Instrumentation (for D3+ only if competence passes; not used in V2 screening)

| | M1 Qwen3-32B | M2 Olmo-3.1-32B | M3 Gemma-4-31B |
|---|---|---|---|
| Residual stream at last prompt position | `output_hidden_states` → 65 × 5120 per world | 65 × 5120 | 61 × 5376 (language model of the wrapper; to verify under transformers 5.x) |
| Activation storage, fp16, last position, all layers | ≈ 0.67 MB/world → ~0.9 GB for 1,320 worlds | same | ≈ 0.66 MB/world |
| Hooks / patching | PyTorch forward hooks on decoder layers; same code path as BTC-D2-001b | same | needs new code for the Gemma 4 module tree (stack change) |
| Optional tools | community SAEs (diagnostic only, never primary) | — | — |

D3 capture uses the same convention as BTC-D2-001b, which was validated as bitwise reproducible on 4B. Activations would be returned as bytes, or written to a short-lived Volume deleted after download; the storage cost would be stated in that run's request.

## 4. Data strategy: no reuse of observed data for confirmation

**Status of existing data:**

| Set | Seen by a model? | V2 use |
|---|---|---|
| pilot_v1 + pathcf **train/val** (352 worlds) | yes (Qwen3-4B, BTC-D2-002) | **pipeline debugging only** (dry runs, format checks). Never used for a V2 decision, threshold or selection. |
| pilot_v1 **test** (60) + pathcf test (28) | no model has seen it | **retired, locked** for V2 and never opened. Its generator, labels and design informed our thinking, so it is not "fresh" for confirming V2 decisions. Kept as an archived artefact only. |

**New data: `v2`**, generated on CPU after this review is approved, before any V2 GPU run:
- Same generator code (`src/build_dataset.py`, `src/build_path_cf.py`) and same template; **no task change**.
- New pre-registered master seed **2026111302** (fixed here, before generation).
- **All state labels disjoint from every v1 label** (exclusion list from pilot_v1 + pathcf).
- Depths 2–4 only. D-7 (depth 5–6) stays deferred until competence results, as instructed.
- Family-level partition into **four sealed pools**, hashed and recorded before any model sees them:

| Pool | Families | Worlds | Used for | Uses allowed |
|---|---|---|---|---|
| **SCREEN** | 60 (20/depth) | 240: {base, cf, swap_base, swap_cf} | model screening (§5), each candidate once | 1 per model |
| **COMPETENCE** | 90 (30/depth) | ~1,320: 10 v1 variants + pathcf | pre-registered competence checkpoint for the **selected model only** | **exactly 1** |
| **PROBE-DEV** (D4 train/val) | reserved IDs; generated with D-5 only if competence passes | — | probe development | — |
| **LOCKED-TEST** (H1 confirmatory) | reserved IDs; generated with D-5 | — | D-1′ confirmatory test | once, at the end |

**Rules:**
- SCREEN informs model selection only.
- COMPETENCE is never used for selection, so it is free of winner's-curse bias from screening.
- If COMPETENCE fails, it is **burnt**: no second model may be evaluated on it.
- PROBE-DEV and LOCKED-TEST are generated only after competence passes. They come from a separate seed stream, so no model has seen them and no decision has been informed by them.

## 5. Stage S — screening (one run per candidate, each separately approved)

**Order** (fixed): M1 → M2 → M3. **All three are screened** whatever the earlier outcomes, to avoid optional-stopping bias in selection. If the owner declines M3's stack change, it is screened only with M1 and M2 and this is recorded.

Criteria, pre-registered, evaluated on SCREEN with the BTC-D2-002 tie rule (tie = 0.5; strict and lenient bounds reported):

| ID | Criterion | Rationale |
|---|---|---|
| S1 | depth-2 base+cf primary accuracy ≥ **0.70** (n = 40, 20 families) | same threshold as the v0.3 checkpoint |
| S2 | depth-2 base/cf pairs **both correct ≥ 0.50** (chance 0.25) | targets the 4B failure mode (choice insensitive to the decisive edge) |
| S3 | A+B probability mass ≥ 0.5 in ≥ 95 % of worlds | the answer format is actually followed |

- A model **passes screening** iff S1 ∧ S2 ∧ S3.
- Also reported, not decision-relevant: depth-3 and depth-4 accuracy, swap consistency, A/B bias, ties and confident errors.
- **Selection rule** (pre-registered): among passing models, pick the highest **depth-4** base+cf primary accuracy on SCREEN, because D-1′'s primary stratum is depth 4. Ties go to the lower estimated cost, then the earlier order.
- **Stop S:** if **no** candidate passes, **STOP V2**. Report all three profiles. No further models, prompts or formats without a new owner scientific decision.

## 6. Stage C — competence checkpoint (selected model only, one run, separately approved)

Evaluated once on COMPETENCE (90 families), same tie rule:

| ID | Criterion | Decision role |
|---|---|---|
| **C1** | depth-2 base+cf primary accuracy ≥ **0.70** **and** 95 % family-bootstrap lower bound > 0.50 (n = 60 worlds, 30 families) | **competence** (the lower bound strengthens v0.3, which used the point estimate on 16 worlds) |
| **C2** | depth-2 base/cf both-correct rate ≥ 0.50 | **competence** (counterfactual sensitivity) |
| **C3** | A+B mass ≥ 0.5 in ≥ 95 % of worlds | **competence** (format) |
| **C4** | depth-4 base+cf primary accuracy ≥ 0.70 | **D-1′ feasibility** (primary stratum is depth 4) |
| C5 | admissible depth-4 `upath` pairs ≥ 50 % of constructed | **D-1′ feasibility** (enough truly unchosen pairs) |

**Decision table:**
- C1 ∧ C2 ∧ C3 fail → **STOP**. Competence not met; D3+ remains stopped; the COMPETENCE pool is burnt.
- Competence passes but C4 or C5 fails → competence is shown only at shallow depth. **D-1′ primary is infeasible with this model.** Report and stop. The owner decides; the pooled 3+4 secondary is not promoted to primary.
- All pass → request approval for D-5 (generate PROBE-DEV / LOCKED-TEST) and then D3. Each is a separate approval.

**Power** (descriptive): with 30 depth-2 families, a true accuracy of 0.80 gives an observed rate ≥ 0.70 with high probability, and a lower CI bound of about 0.68. A true accuracy of 0.60 would typically fail C1. The point estimate rests on 60 worlds, against 16 in BTC-D2-002.

## 7. What V2 does not change (confirmations)

- H1, H2, H3, D-1′, the evidential ladder L1–L4, P0 and the reporting format (§6b) are unchanged.
- Prompt text is byte-identical in template to v1. Only labels and seeds differ (new data). Each model's **official** chat template is used in non-thinking mode.
- No fine-tuning and no closed API models.
- The v1 locked test is never opened. No V2 confirmatory claim uses any already-observed item.

## 8. Risks and honest limits

1. **Format ceiling.** Without reasoning tokens, a depth-d decision needs d-hop composition plus a comparison inside a single forward pass. Scale may not remove this limit; Hanna & Ameisen (ICLR 2026) report that latent planning emerges with scale, but not necessarily for this task. If all candidates fail, the justified conclusion is limited to "not met in this format by these models". Changing the format (e.g. reasoning before the answer) changes the "pre-answer" construct and is **outside V2**.
2. **Template confounds across models.** Empty think block (M1), injected system prompt (M2), thought channel (M3): models differ in more than size. Results are reported per model, never pooled.
3. **M3 stack change.** It requires transformers 5.x, a new instrumentation stack, so it needs explicit approval. If declined, V2 runs with M1 and M2 only.
4. **Allocation wait** for A100-80 is unknown, and long waits are possible. No retries.
5. **Cost estimates** are extrapolated from a 4B run, not measured at 32B. Hard caps and timeouts bound them.

## 9. Decisions requested from the owner

| ID | Decision | Recommendation |
|---|---|---|
| V2-1 | Candidate set {M1 Qwen3-32B, M2 Olmo-3.1-32B-Instruct, M3 Gemma-4-31B-it} at the listed revisions | approve |
| V2-2 | Allow transformers 5.x **only** for M3 (separate pinned image; M1/M2 stay on 4.57.1) | approve, or drop M3 |
| V2-3 | Data strategy §4: fresh v2 data (seed 2026111302, labels disjoint from v1), four sealed pools, v1 test retired | approve |
| V2-4 | Screening criteria S1–S3, screen all candidates, selection rule §5 | approve |
| V2-5 | Competence criteria C1–C5 and decision table §6 (adds a CI lower bound and depth-4 feasibility) | approve |
| V2-6 | GPU: 1 × A100 80 GB, caps $1.00 per screening run and $1.50 for the competence run, programme cap $4.50, no Volume | approve |
| V2-7 | After approval: generate v2 SCREEN + COMPETENCE on CPU, with tests and hashes (free, no GPU) | approve |

GPU runs that would follow, each requiring its own explicit authorisation: **BTC-V2-S1** (M1), **BTC-V2-S2** (M2), **BTC-V2-S3** (M3), then **BTC-V2-C1** (selected model).

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (unchanged)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently (model change proposed for approval only)
[YES] Preserved oracle isolation and train/val/test independence (new sealed pools; v1 test retired unopened)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (none executed; all listed as decisions)
[YES] Preserved negative results and raw logs
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none. CPU-only fact checks (HF metadata, tokenizer/config files, Modal pricing page).
Next step: owner validation of V2-1 … V2-7.
```
