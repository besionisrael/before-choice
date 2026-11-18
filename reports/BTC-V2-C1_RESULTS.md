# BTC-V2-C1 — Gemma-4-31B-it competence checkpoint: **COMPETENCE PASS; D-1′ FEASIBLE** (C1–C5 all ✓)

**Date:** 2026-11-13 · **Authorisation:** owner, BTC-V2-C1, single execution on the frozen COMPETENCE pool · **Status:** executed once, scored, **STOPPED**. D3, D5 and later stages are NOT launched; only a validation request is submitted (§10).

> A **behavioural** competence result under BTC-V2-PROTOCOL_FREEZE v1.0. It establishes that the precondition for H1 testing is met for this model, format and task. It is **not** evidence for or against H1, H2 or H3, which remain unchanged and untested. No probe was trained and no activation was captured.

## 1. Pre-run verification

| Check | Result |
|---|---|
| COMPETENCE pool hashes | data `b1df46cd…` ✓, model inputs `dc6c7115…` ✓, prompt digest `cf3c6e26…` ✓ (verified locally and inside the container) |
| Template (Gemma 4 @ `842da379…`, transformers 5.23.0) | frozen tail ✓, single `<bos>` ✓, empty thought channel ✓, **pre-answer token `<channel\|>`** on 1,320/1,320 |
| Token counts vs CPU-check reference | 1,320/1,320 |
| A / B ids | 236776 / 236799 |
| Resources | 1 × A100-80GB, cpu=(4,4), memory=(16384, 98304) MiB, timeout 1,240 s, guard 1,160 s, `retries=0`, `max_containers=1`; computed worst case $1.287 |
| Selection | `outputs/v2/selection.json` = `gemma4_31b` (enforced by the driver) |
| Code | identical to S3 (`v2_run_gemma` `5b25f5ee…`, `v2_remote` `fef0dcda…`, `v2_common` `8f6d6a53…`, `v2_local` `555eff1f…`); 67/67 tests |

## 2. Run

| Item | Value |
|---|---|
| Command | `modal run src/v2_run_gemma.py --run-id BTC-V2-C1 --model gemma4_31b` |
| Modal app | `ap-Fl6k865yhqxoWcvFmdD6Bs`, exit 0, stopped. Single attempt; cached image |
| Forwards | 1,320 worlds + 20 replays = 1,340 |
| Environment | NVIDIA A100-SXM4-80GB; torch 2.6.0+cu124; transformers 5.23.0; bf16 |
| **GPU memory peak** | **63.3 GB** |

**Timing:**

| Phase | Seconds |
|---|---|
| Call → remote start | 12.8 |
| Tokenizer | 7.8 |
| Weights download + load | **161.7** |
| 1,340 forwards | **170.1** (≈ 127 ms each) |
| Remote total | 339.5 |
| Local wall time | 352.1 |

The run ended far below the 1,160 s guard. No assertion fired and no error occurred.

## 3. Pre-registered criteria (applied without adaptation)

| ID | Criterion | Observed | Result |
|---|---|---|---|
| **C1** | depth-2 base+cf primary accuracy ≥ 0.70 **and** family-bootstrap 95 % lower bound > 0.50 (n = 60, 30 families) | **1.000** (60/60; 0 ties); CI [1.00, 1.00] | ✓ |
| **C2** | depth-2 base/cf both correct ≥ 0.50 | **30/30 = 1.00** | ✓ |
| **C3a** | A+B mass ≥ 0.5 in ≥ 95 % of worlds | **1,255/1,320 = 0.9508** (threshold 1,254) | ✓ (**margin: 1 world**) |
| **C3b** | top-1 over the full vocabulary ∈ {A, B} in ≥ 95 % of worlds | **1,255/1,320 = 0.9508** | ✓ (**margin: 1 world**) |
| **C4** | depth-4 base+cf primary accuracy ≥ 0.70 | **0.800** (48/60); CI [0.72, 0.88] | ✓ |
| **C5** | admissible depth-4 `upath` pairs ≥ 50 % of constructed | **45/60 = 0.75** | ✓ |

**Decision** (pre-registered table): **COMPETENCE = PASS; D-1′ FEASIBLE.** Per the owner's instructions, nothing further is launched.

> **Caveat on C3a/C3b:** both are met by exactly one world above the threshold. A single additional non-conformant world would have produced **COMPETENCE FAIL**. The criterion is pooled over all variants, as pre-registered, and is applied as written. Its fragility and its concentration in decoy worlds (§6) are reported for the owner's scientific judgement; they do not change the decision.

## 4. Depth-2 competence and base/cf pairs (items 1–2)

| Depth | {base, cf} n | Primary | 95 % family CI | base/cf both correct | Choice changed base → cf |
|---|---|---|---|---|---|
| 2 | 60 | **1.000** | [1.00, 1.00] | **30/30** | 30/30 |
| 3 | 60 | 0.950 | [0.88, 1.00] | 27/30 | 27/30 |
| 4 | 60 | **0.800** | [0.72, 0.88] | **18/30** | 18/30 |

## 5. Depth-4 accuracy and counterfactual sensitivity (item 4)

Choice changed across pairs (all depths, n = 90 each):

| Pair | Optimal changes? | Choice changed |
|---|---|---|
| base → cf | yes | 75 |
| alias_base → alias_cf | yes | 78 |
| shuffle_base → shuffle_cf | yes | 74 |
| swap_base → swap_cf | yes | 76 |
| base → swap_base (letters exchanged) | yes | 79 |
| decoy_base → decoy_cf | yes | **38** |
| base → shuffle_base (rule lines reordered) | **no** | **14** (should be 0 for a perfectly order-invariant solver) |

**Accuracy by variant** (n = 90; upath/cpath n = 60):

| Variant | Accuracy | Variant | Accuracy |
|---|---|---|---|
| base | 0.922 | cf | 0.911 |
| alias_base | 0.978 | alias_cf | 0.889 |
| swap_base | 0.911 | swap_cf | 0.933 |
| shuffle_base | 0.883 | shuffle_cf | 0.911 |
| **decoy_base** | **0.656** | **decoy_cf** | **0.744** |
| upath_base | 0.900 | upath_cf | 0.817 |
| cpath_base | 0.833 | cpath_cf | 0.817 |
| xswap_base | 0.939 | xswap_cf | 0.889 |

**Depth-4 behaviour** (all variants): accuracy 0.925 when A is optimal and **0.613 when B is optimal**. The A-default on failure seen in S3 persists: 136 of the 168 errors are "A" choices, and 111 of the errors are at depth 4.

## 6. Format conformity (item 3)

- **All 65 non-conformant worlds are decoy worlds:** decoy_base 31, decoy_cf 34, at depth 4 (52) and depth 3 (13).
- In each of them the top-1 token is **"To"**: the model begins a sentence instead of answering. Their A+B mass is < 0.5 in every case, with a minimum of 0.005.
- **Within decoy variants, conformity is 115/180 = 0.64.** Outside decoy variants it is 1,140/1,140 = 1.00.
- The forced A/B choice in those 65 worlds is correct in 43.

**Descriptive reading (POST HOC, not pre-registered):**
- Adding an unreachable chain whose terminal carries the highest reward:
  - degrades accuracy (decoy_base depth 4: 0.50);
  - breaks the answer format;
  - reduces counterfactual sensitivity (38/90 changes, against 75/90 for base/cf).
- This is compatible with the model's decision process being **partly anchored on the maximum printed reward**, the shortcut P0 identified. It does not establish that, and a decoy prompt is also longer and contains extra rules.
- It is directly relevant to the forward-vs-backward question (H2) and to interpreting any future D-1′ result.

## 7. Chosen vs unchosen branches, separately (items 5–6)

Choice-preserving pairs (source world → manipulated world). "Admissible" means the model chose the optimal action on **both** members.

| Branch manipulated | Pair type | Depth | Constructed | Choice invariant | **Admissible** |
|---|---|---|---|---|---|
| **Unchosen** | upath (base + cf sources) | 3 | 60 | 58 | 56 |
| **Unchosen** | upath | **4** | **60** | **56** | **45** (base-source 23/30, cf-source 22/30) |
| **Chosen** | cpath | 3 | 60 | 56 | 53 |
| **Chosen** | cpath | 4 | 60 | 58 | 46 |
| Both (membership) | xswap | 2 / 3 / 4 | 60 / 60 / 60 | 60 / 58 / 54 | 60 / 56 / 45 |

**For D-1′:**
- The admissible primary population would be **45 depth-4 unchosen-branch pairs** in COMPETENCE.
- These are **behavioural counts on the COMPETENCE pool, which is not used for H1**.
- The H1 probe analyses would run on PROBE-DEV / LOCKED-TEST, which are not yet generated, and their admissible counts will differ.
- Choice invariance under unchosen-path reordering is high (56/60 at depth 4), as expected: the manipulation does not change the optimum.

## 8. Reproducibility, errors, cost (item 7)

- **Reproducibility:** the 20 replays have identical choices and max |Δ log p| = 0.0.
- **Choices:** A 766 · B 552 · tie 2 (1 in shuffle_base, 1 in xswap_base).
- **Errors:** 168/1,320. 111 are at depth 4, 43 at depth 3 and 14 at depth 2 (the depth-2 errors are in variants other than base/cf, mostly decoy). There are **140 confident errors** (|margin| ≥ 3).
- **Margins:** median |log p(A) − log p(B)| is 21.8 at depth 2, 17.4 at depth 3 and 12.6 at depth 4.

**Cost** (actual, `modal billing report --show-resources`):

| Resource | $ |
|---|---|
| A100-80GB | 0.2417 |
| CPU | 0.0183 |
| Memory | 0.0124 |
| Network egress | 0.0118 |
| **Run total** | **$0.284** |

- Below the $1.287 worst case and the $1.50 objective. **No overrun.** $0.00 charged (credits). Cached image.
- **V2 programme total:** S1 0.158 + S2 0.245 + S3 0.169 + C1 0.284 = **$0.856**, plus 2 image builds (≈ $0.02–0.04). That is well under the $4.50 objective.

## 9. What is and is not concluded

**Justified:**
- With first-token A/B answering and no textual reasoning, Gemma-4-31B-it meets the pre-registered competence criteria on fresh, sealed data. The COMPETENCE pool was used exactly once.
- Its choices track the decisive transition at depth 2 (30/30 pairs) and largely at depths 3–4.
- The behavioural precondition for D-1′ is met: depth 4 at 0.80, and 45/60 admissible unchosen-path pairs.

**Not justified:**
- Any claim about prospective path representations: H1 has not been tested.
- Any claim about forward computation or sweep (H2) or causal use (H3).
- That the model "plans".
- That the competence is robust to decoys: it is not (§6).

**Open concerns to weigh before any H1 work:**
1. The format criterion passed by one world, with all failures in decoy worlds.
2. Decoy sensitivity is compatible with reward-anchored (backward) computation. Together with P0, this means D-1′ alone cannot discriminate forward from backward construction; this was already stated in PROTOCOL_FREEZE §4b and §8.
3. A rule-order effect remains (14/90 choice changes under shuffle with the same optimum).
4. Depth-4 errors are biased toward "A".

## 10. Request for scientific validation (no execution requested)

All criteria pass. As instructed, **no D3/D5 run is launched or requested for execution here.** I ask the owner to validate scientifically which step comes next. Under the existing protocol, the order would be:

1. **D-5:** generate PROBE-DEV and LOCKED-TEST on CPU, from a separate seed, with reserved family ids ≥ F20000. This includes decisions on size (power for D-1′ given ≈ 75 % depth-4 admissibility) and on D-7 (depth 5–6), now eligible since competence results exist.
2. Then a **D3 activation-capture design** for Gemma 4 (61 hidden states × 5,376, pre-answer position `<channel|>`), as a separate GPU request.

Points I recommend the owner decide first:
- How the C3 margin and the decoy-format failures should be treated: for example, whether decoy worlds are excluded from H1 probe populations, decided **before** any activation is seen.
- D-8: the minimum-reward decoy that blocks the backward shortcut on the unchosen branch. Its relevance has increased given §6.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (unchanged, untested)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently (gemma-4-31B-it @ 842da379…)
[YES] Preserved oracle isolation and train/val/test independence (COMPETENCE used once; PROBE-DEV/LOCKED-TEST not generated)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (BTC-V2-C1)
[YES] Preserved negative results and raw logs
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none. Decoy/format analysis labelled POST HOC.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner scientific validation of the next stage (D-5 data generation and D3 design); nothing launched.
```

## 11. Artifacts

| Artifact | Path / hash |
|---|---|
| Raw outputs | `outputs/v2/BTC-V2-C1/behavior_raw.jsonl` (`c4f87aa0…`) |
| Manifest (pre-check, timing, env, cost, decision) | `outputs/v2/BTC-V2-C1/manifest.json` |
| Scored metrics + per-world | `outputs/v2/BTC-V2-C1/scored.json` (`f670dc5b…`) |
| Logs | `outputs/logs/v2c1_run.log`, `v2c1_score.log`, `v2c1_billing_resources.csv`, `v2c1_{start,end}_utc.txt` |
