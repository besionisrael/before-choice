# BTC-V2-S2 — Olmo-3.1-32B-Instruct screening: **SCREEN FAIL** (S1 ✗, S2 ✗, S3 ✓, S4 ✓)

**Date:** 2026-11-13 · **Authorisation:** owner, BTC-V2-S2, single GPU execution · **Status:** executed once, scored, **STOPPED**. S3, C1 and D3+ are NOT RUN and not authorised.

> A screening result for one candidate under the frozen V2 protocol. It is not evidence for or against H1, H2 or H3, which remain unchanged and untested. No probes were trained and no activations were captured.

## 1. Pre-inference verification (owner condition)

Official template checked on CPU at the pinned revision, before the run (`outputs/v2_audit/olmo31_32b_chat_template.jinja`):

| Item | Value |
|---|---|
| Revision | `ac0587e4a7744a551c059d8cd17ba220bc940dae` |
| `chat_template.jinja` sha256 | `49a944c9814e130bd3cc80a2420c55ff22e24a4e4ba97009aa60000909b5eac3` |
| Default system prompt | **injected by the official template**: "You are Olmo, a helpful AI assistant built by Ai2. Your date cutoff is December 2024, …" (kept, as frozen in §1/§5 of the protocol) |
| Generation tail / pre-answer token | `…<\|im_end\|>\n<\|im_start\|>assistant\n`; last token `Ċ` |
| `A` / `B` | single tokens 32 / 33 |
| Frozen tail on SCREEN prompts | 240/240 |

The same assertions ran again inside the container: tail, ids, token counts, data hashes and fields. None fired.

## 2. Run

| Item | Value |
|---|---|
| Command | `modal run src/v2_run_std.py --run-id BTC-V2-S2` (same code as S1: `19bc60d9…` / `fef0dcda…` / `8f6d6a53…` / `555eff1f…`) |
| Modal app | `ap-krDl9AnSNOqObuyIA5IAUO`, exit 0, stopped. Single attempt, `retries=0`, `max_containers=1`; cached image reused |
| Data | SCREEN only: 240 worlds + 20 replays = 260 forwards (hashes `cd3499d6…` / `42cc0517…` / digest `6e52d9dc…`) |
| Hardware / env | 1 × A100-SXM4-80GB, cpu=(4,4), memory=(16, 96) GiB, timeout 780 s / guard 720 s; torch 2.5.1+cu124, transformers 4.57.1; GPU memory peak **64.6 GB** |

**Timing:**

| Phase | Seconds |
|---|---|
| Call → remote start | 7.3 |
| Weights download + load (64.5 GB) | **250.6** |
| 260 forwards | **26.1** (≈ 100 ms each) |
| Remote total | 277.3 |
| Local wall time | 284.6 |

The `rope_scaling` type warning (integer beta values) was logged, as anticipated in the readiness report; it did not interrupt the run.

## 3. Pre-registered criteria (applied without adaptation)

| ID | Criterion | Observed | Result |
|---|---|---|---|
| S1 | depth-2 base+cf primary accuracy ≥ 0.70 (n = 40, 20 fam.) | **0.550** (strict 0.525, lenient 0.575; 2 ties); CI [0.49, 0.64] | **FAIL** |
| S2 | depth-2 base/cf both correct ≥ 0.50 | **2/20 = 0.10** (below the 0.25 chance rate) | **FAIL** |
| S3 | A+B mass ≥ 0.5 in ≥ 95 % | 240/240 (min 0.743) | PASS |
| S4 | top-1 over the full vocabulary ∈ {A, B} in ≥ 95 % | 240/240 | PASS |

**SCREEN_PASS = False.** Both S1 bounds (strict and lenient) are below 0.70.

## 4. Performance by depth and base/cf sensitivity

| Depth | {base, cf} n | Ties | Primary | Strict–lenient | 95 % CI | base/cf both correct | base/cf choice changed |
|---|---|---|---|---|---|---|---|
| 2 | 40 | 2 | 0.550 | 0.525–0.575 | [0.49, 0.64] | **2/20** (F10005, F10021) | 4/20 |
| 3 | 40 | 1 | 0.463 | 0.450–0.475 | [0.40, 0.50] | 0/20 | 2/20 |
| 4 | 40 | 6 | 0.550 | 0.475–0.625 | [0.46, 0.64] | 2/20 | 5/20 |

**By variant** (n = 60 each): base 0.550 · cf 0.492 · swap_base 0.567 · swap_cf 0.483. All CIs include 0.5.

**Choice changes across pairs:**

| Pair | Optimal changes? | Choice changed |
|---|---|---|
| base → cf (decisive edge swapped) | yes | **11/60** |
| swap_base → swap_cf | yes | 11/60 |
| base → swap_base (A↔B letters exchanged, graph identical) | yes | **30/60** |

The choice is about as likely to change when only the **letters** are exchanged (30/60) as by chance. It rarely changes when the **decisive transition** changes (11/60). This is the same insensitivity pattern seen in Qwen3-4B and Qwen3-32B.

## 5. A/B distribution, conformity, errors, reproducibility

**Choices:** A 83 · B **146** · tie **11**. This is a B preference, the opposite of Qwen3-32B's A preference.

| Depth | A / B / tie | Accuracy when optimal = A | Accuracy when optimal = B | Median margin log p(A) − log p(B) |
|---|---|---|---|---|
| 2 | 24 / 54 / 2 | 0.38 | 0.78 | −0.75 |
| 3 | 26 / 52 / 2 | 0.30 | 0.65 | −0.63 |
| 4 | 33 / 40 / 7 | 0.40 | 0.50 | −0.06 |

- **Low-confidence decisions:** median |margin| < 1 nat at every depth, only 4 confident errors (|margin| ≥ 3), and **11 exact ties** (bf16), 7 of them at depth 4. Contrast Qwen3-32B: confident, A-biased, 74 confident errors.
- **Format conformity:** perfect. Top-1 is A or B in 240/240 worlds; A+B mass is ≥ 0.743 everywhere.
- **Reproducibility:** the 20 replays have identical choices and max |Δ log p| = 0.0.

**Descriptive reading (not a criterion):**
- Olmo follows the answer format but its choices are near-indifferent between A and B, with a B lean.
- The choices are largely insensitive to the graph's decisive transition and sensitive to the letter labels.
- There is no sign of depth-2 competence: S2 at 0.10 is below the chance rate for pairs.

## 6. Cost (actual, `modal billing report --show-resources`)

| Resource | $ |
|---|---|
| A100-80GB | 0.1958 |
| CPU | 0.0148 |
| Memory | 0.0100 |
| Network egress | 0.0241 |
| **Run total** | **$0.245** |

- No image rebuild: the cached S1 image was reused.
- Below the $0.85 worst case and the $1.00 objective. **No overrun.** $0.00 charged (credits).
- Billing data may still update.
- **V2 spend so far:** S1 $0.158 + S2 $0.245 + one image build ≈ **$0.42**.

## 7. Artifacts

| Artifact | Path / hash |
|---|---|
| Raw outputs | `outputs/v2/BTC-V2-S2/behavior_raw.jsonl` (`1bfbd02b…`) |
| Manifest (template check, timing, env, actual cost, decision) | `outputs/v2/BTC-V2-S2/manifest.json` |
| Scored metrics + per-world | `outputs/v2/BTC-V2-S2/scored.json` (`7cee596f…`) |
| Logs | `outputs/logs/v2s2_run.log`, `v2s2_score.log`, `v2s2_billing_resources.csv`, `v2s2_{start,end}_utc.txt` |
| Template | `outputs/v2_audit/olmo31_32b_chat_template.jinja` |

## 8. Status

**Two of the three approved candidates have now failed screening:**

| Model | S1 | S2 | S3 | S4 | Bias |
|---|---|---|---|---|---|
| Qwen3-32B | 0.613 | 0.25 | ✓ | ✓ | strong A, confident |
| Olmo-3.1-32B | 0.550 | 0.10 | ✓ | ✓ | B lean, low confidence, 11 ties |

- Neither is eligible for selection.
- **Per the frozen rule, the selection decision cannot be made until M3 (Gemma-4-31B) is screened or formally excluded with a recorded reason.** If M3 also fails, or is excluded, the rule gives STOP V2 (no passer).
- BTC-V2-S3 requires a new authorisation and the Gemma 4 stack decision (F-4 is approved in principle; its CPU environment is verified). None is requested automatically.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (unchanged, untested)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently (pinned ac0587e4…)
[YES] Preserved oracle isolation and train/val/test independence (SCREEN only; COMPETENCE untouched)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (BTC-V2-S2)
[YES] Preserved negative results and raw logs
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decision on BTC-V2-S3 (Gemma 4) or on formally excluding M3 (which would yield STOP V2).
```
