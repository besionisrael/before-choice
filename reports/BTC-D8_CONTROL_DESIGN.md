# BTC-D8_CONTROL_DESIGN — matched reward-decoy controls (design; NOT adopted as an endpoint)

**Date:** 2026-11-13 · **Status:** design for owner validation. No GPU, no model exposure. **Not an endpoint**, and **not evidence of forward computation** (owner decision 3).

## 1. Motivation (from existing results)

- **P0 (prompt-only):** for the unchosen branch, backward chaining from the **lowest** printed reward (BWD−) is always the cheapest route. It reaches the D-1′ target in 2 rule applications.
- **The existing (max-reward) decoy** blocks backward anchoring only for the **chosen** branch.
- **BTC-V2-C1 (POST HOC):** max-decoy worlds degraded Gemma's accuracy (0.66/0.74) and its format conformity (65 non-A/B outputs, all decoy). This is compatible with reward-anchored processing, and is confounded by prompt length and out-of-range rewards (10–12).

D-8 targets the unchosen-branch backward shortcut. It also controls the confounds of the existing decoy.

## 2. Design: three matched decoys per family

For each family (depths 3–4), one unreachable chain of `depth` edges is added, with **identical labels and insertion positions** in all three variants. Only the decoy terminal's reward differs:

| Variant | Decoy terminal reward | Blocks prompt-only shortcut |
|---|---|---|
| `decoy_max` | **9** (> both real rewards) | BWD+ (chosen branch, from the max reward) |
| `decoy_min` (**D-8 proper**) | **1** (< both real rewards) | **BWD− (unchosen branch, from the min reward)** |
| `decoy_mid` | strictly between the real rewards | none (presence / length / position control) |

- Real rewards are drawn from **2–8 with gap ≥ 2**, so all three decoy rewards are single-digit and in-range.
- This is a **separate exploratory pool**. Its reward range differs from v1/v2 (1–9) and is not mixed with PROBE-DEV or LOCKED-TEST.

## 3. Prototype audit (DESIGN SAMPLE only)

`src/d8_prototype.py` builds a throwaway design sample (seed 909090; 80 families, 40 at depth 3 and 40 at depth 4; ids F90000+). No model has seen it. Output: `outputs/v2_audit/d8_prototype_audit.json`.

| Check | Result |
|---|---|
| Independent oracle violations | 0 |
| Optimal action identical across the 3 variants | ✓ (80/80) |
| Prompt length identical across variants | ✓ |
| Prompts differ in **exactly one character** (decoy reward digit) | ✓ |
| Gemma-4 token count identical across variants (official template) | ✓ |
| Decoy reward ordering as designed | ✓ |

**Prompt-only strategy recovery of the depth-2 target** (fraction of worlds; budget = depth):

| Variant | Depth | Unchosen (non-optimal) d2: FWD / BWD− / BWD+ / ELIM | Chosen (optimal) d2: FWD / BWD− / BWD+ / ELIM |
|---|---|---|---|
| decoy_max | 3 | 1.00 / **1.00** / 0 / 0 | 1.00 / 0 / **0.00** / 0 |
| decoy_max | 4 | 1.00 / **1.00** / 0 / 1.00 | 1.00 / 0 / **0.00** / 0 |
| **decoy_min** | 3 | 1.00 / **0.00** / 0 / 0 | 1.00 / 0 / 1.00 / 0 |
| **decoy_min** | 4 | 1.00 / **0.00** / 0 / **0.00** | 1.00 / 0 / 1.00 / 1.00 |
| decoy_mid | 3 | 1.00 / 1.00 / 0 / 0 | 1.00 / 0 / 1.00 / 0 |
| decoy_mid | 4 | 1.00 / 1.00 / 0 / 1.00 | 1.00 / 0 / 1.00 / 1.00 |

The ELIM strategy (2 composed hops) only applies at depth 4 in the P0 implementation; its 0 at depth 3 is not informative.

**Reading:**
- `decoy_min` removes the min-reward backward anchor for the unchosen branch at depth 4 (BWD− and ELIM both fall from 1.00 to 0.00).
- `decoy_max` removes the max-reward anchor for the chosen branch.
- `decoy_mid` removes neither, so it isolates the effect of merely adding a chain.
- Forward chaining from S recovers every target in every variant.

## 4. What D-8 can and cannot show (limits stated in advance)

- **A reachability-aware backward strategy is not blocked by any decoy.** Tracing back from each terminal until S is reached, then comparing rewards, costs ≈ depth hops per terminal. D-8 raises the cost of backward reasoning (the cheapest anchor becomes misleading); it does not make backward reasoning impossible.
- So differences between decoy_min, decoy_mid and decoy_max can show **sensitivity to reward anchoring**. They **cannot establish forward computation**, and must not be reported as evidence for H2's forward organisation or for L4.
- Single-digit in-range rewards remove the "large number" and length confounds of the existing decoy. The reward *distribution* (2–8 plus decoy), however, differs from the main pools, so D-8 results are compared only within D-8.

## 5. Proposed use (exploratory, if later approved)

- **Behavioural contrast:** accuracy and format conformity, decoy_min vs decoy_mid vs decoy_max, per depth, with family bootstrap.
- **Representational contrast** (only after H1 work is approved): D-1′-style unchosen-d2 decoding in decoy_min vs decoy_mid. This is labelled **EXPLORATORY**, never confirmatory, never pooled with the primary.
- **Pool size if approved:** 100 families per depth (3, 4) × {base, cf, decoy_max, decoy_min, decoy_mid, and their cf counterparts} ≈ 1,200 worlds, with a separate seed and labels disjoint from all pools.

## 6. Decisions requested

| ID | Decision |
|---|---|
| D8-1 | Accept the matched three-decoy design (max/min/mid, single-digit, real rewards 2–8, gap ≥ 2) as a **secondary exploratory control** |
| D8-2 | Keep D-8 out of PROBE-DEV / LOCKED-TEST and out of any confirmatory endpoint |
| D8-3 | (Later) authorise generation of a D-8 exploratory pool and a behavioural GPU run, each separately |

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (design sample never shown to a model)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (none executed)
[YES] Preserved negative results and raw logs
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none. D-8 is not presented as evidence of forward computation.
Next step: owner validation of D8-1 … D8-3.
```
