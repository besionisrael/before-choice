# BTC-V2-S1 — Qwen3-32B screening: **SCREEN FAIL** (S1 ✗, S2 ✗, S3 ✓, S4 ✓)

**Date:** 2026-11-13 · **Authorisation:** owner, BTC-V2-S1, single GPU execution · **Status:** executed once, scored, **STOPPED**. S2, S3, C1 and D3+ are NOT RUN and not authorised.

> This is a screening result for one candidate model under the frozen V2 protocol. It says nothing for or against H1, H2 or H3. No probes were trained and no activations were captured.

## 1. Run

| Item | Value |
|---|---|
| Command | `modal run src/v2_run_std.py --run-id BTC-V2-S1` (script sha `19bc60d9…`, `v2_remote` `fef0dcda…`, `v2_common` `8f6d6a53…`) |
| Modal app | `ap-JfQvAzr6gTXsVrri1n7c98`, exit 0, stopped. Single attempt, `retries=0`, `max_containers=1` |
| Model | `Qwen/Qwen3-32B` @ `9216db5781bf21249d130ec9da846c4624c16137`, bf16, `enable_thinking=False` (empty `<think>` block; tail asserted) |
| Data | SCREEN only: 240 worlds (60 families, data `cd3499d6…`, inputs `42cc0517…`, digest `6e52d9dc…`, all verified in container) + 20 replays = 260 forwards |
| Hardware | 1 × NVIDIA A100-SXM4-80GB, cpu=(4,4), memory=(16 GiB, 96 GiB), timeout 780 s, guard 720 s |
| Environment | torch 2.5.1+cu124, transformers 4.57.1, CUDA 12.4. GPU memory peak **65.7 GB** |

**Timing:**

| Phase | Seconds |
|---|---|
| Image build (new image, CPU) | 78 |
| Wait from call to remote start | 10.9 |
| Tokenizer | 2.8 |
| Weights download + load to GPU (65.5 GB) | **178.6** |
| 260 forward passes | **24.7** (≈ 95 ms each) |
| Remote function total | 206.2 |
| Local wall time | 216.8 |

The run ended well before the 720 s guard. No assertion fired: template tail, A/B ids, token counts, hashes and fields all matched.

## 2. Pre-registered criteria (applied without adaptation)

| ID | Criterion | Observed | Result |
|---|---|---|---|
| S1 | depth-2 base+cf primary accuracy ≥ 0.70 (n = 40, 20 families) | **0.6125** (strict 0.600, lenient 0.625; 1 tie); 95 % family CI [0.50, 0.725] | **FAIL** |
| S2 | depth-2 base/cf pairs both correct ≥ 0.50 | **5/20 = 0.25** (= chance) | **FAIL** |
| S3 | A+B mass ≥ 0.5 in ≥ 95 % of worlds | 240/240 = 1.00 (min mass 0.836) | PASS |
| S4 | top-1 over the full vocabulary ∈ {A, B} in ≥ 95 % of worlds | 240/240 = 1.00 | PASS |

**SCREEN_PASS = False.** S1 would also fail under the lenient tie bound (0.625).

## 3. Performance

**{base, cf} by depth:**

| Depth | n | Ties | Primary | 95 % CI |
|---|---|---|---|---|
| 2 | 40 | 1 | 0.613 | [0.50, 0.73] |
| 3 | 40 | 0 | 0.525 | [0.50, 0.58] |
| 4 | 40 | 0 | 0.500 | [0.50, 0.50] |

**By variant** (n = 60 each): base 0.558 · cf 0.533 · swap_base 0.567 · swap_cf 0.608. All CIs include 0.5.

**Counterfactual pairs:**

| Pair | Optimal changes? | Pairs | Choice changed | Both correct |
|---|---|---|---|---|
| base → cf, depth 2 | yes | 20 | — | **5** (F10005, F10010, F10024, F10038, F10048) |
| base → cf, all depths | yes | 60 | 8 | — |
| swap_base → swap_cf | yes | 60 | 12 | — |
| base → swap_base (A↔B letters) | yes | 60 | 12 | — |

## 4. First-token conformity, biases and errors

- **Format:** fully conformant. Top-1 is A or B in 240/240 worlds, and the A+B mass is ≥ 0.836 everywhere. Unlike Qwen3-4B, no "Let" outputs.
- **Response bias: very strong preference for `A`.** Choices are A 215, B 23, tie 2. The bias grows with depth:

| Depth | Choices A / B / tie | Accuracy when optimal = A | Accuracy when optimal = B | Median margin log p(A) − log p(B) |
|---|---|---|---|---|
| 2 | 60 / 19 / 1 | 0.95 | **0.42** | +5.37 |
| 3 | 77 / 3 / 0 | 0.97 | **0.05** | +4.00 |
| 4 | 78 / 1 / 1 | 0.97 | **0.00** | +3.87 |

- **Of the 23 B choices, 19 are at depth 2.** At depths 3–4 the model answers A almost always and with confidence, so its ~0.5 accuracy there is the response bias meeting a balanced design, not task solving.
- Exchanging the action letters (swap) changes the choice in only 12/60 pairs: the model tends to say "A" whatever path A leads to.
- **Confident errors** (|margin| ≥ 3): 74 of 240 worlds.
- **Determinism:** the 20 replays have identical choices and max |Δ log p| = 0.0.

**Descriptive reading (not a criterion):** at depth 2 there is weak evidence of some sensitivity to the graph. Accuracy is 0.42 when B is optimal against ~0 at depth ≥ 3, and 5/20 depth-2 pairs are both correct. That is at the chance rate for pairs, however, so no competence claim follows. At depth ≥ 3 behaviour is dominated by a default "A" response. This mirrors the D2 picture for Qwen3-4B (an A preference and insensitivity to the decisive edge), with stronger confidence.

## 5. Cost (actual, `modal billing report --show-resources`)

| Resource | $ |
|---|---|
| A100-80GB | 0.1194 |
| CPU | 0.0123 |
| Memory | 0.0062 |
| Network egress | 0.0200 |
| **Run total** | **$0.158** |
| Image build (billed separately) | ≈ $0.01–0.02 |

- That is far below the computed worst case of $0.85 and the $1.00 objective. **No overrun.**
- Charged to the payment method: $0.00 (credits).
- The billing report covers full daily intervals and may still update; a later re-check is advisable.

## 6. Artifacts

| Artifact | Path / hash |
|---|---|
| Raw outputs | `outputs/v2/BTC-V2-S1/behavior_raw.jsonl` (`9cb2aa51…`) |
| Manifest (timing, env, actual cost, decision) | `outputs/v2/BTC-V2-S1/manifest.json` |
| Scored metrics + per-world | `outputs/v2/BTC-V2-S1/scored.json` (`ff0ec8d9…`) |
| Logs | `outputs/logs/v2s1_run.log`, `v2s1_score.log`, `v2s1_billing_resources.csv`, `v2s1_{start,end}_utc.txt` |

## 7. Status and what is NOT concluded

- **M1 Qwen3-32B does not pass V2 screening** (S1, S2 fail; format S3/S4 pass).
- **Per the protocol, M1 is not eligible for selection.** V2 does **not** stop yet: the frozen rule requires screening of all approved candidates (M2, M3) before selection, and **each requires a new authorisation**. None is requested automatically here.
- **Not concluded:**
  - That Qwen3-32B "cannot plan".
  - That larger models or other formats would fail.
  - Anything about prospective representations, which were not measured.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently (pinned 9216db57…)
[YES] Preserved oracle isolation and train/val/test independence (SCREEN only; oracle-free payload; COMPETENCE untouched)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (BTC-V2-S1)
[YES] Preserved negative results and raw logs
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decision on BTC-V2-S2 (Olmo) / S3 (Gemma 4).
```
