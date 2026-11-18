# BTC-H1c — design proposal: dead ends and dead-end-swap pairs (separating chosen- from unchosen-branch information)

**Date:** 2026-11-14 · **Owner decision:** "DÉCISION SCIENTIFIQUE — BTC-H1c" (O3 retained; CPU design and checks only) · **Status:** PROPOSAL.

> **Not done:** no pool generated, no GPU, no probe training.
> **Kept as is:** BTC-H1b v1.0 stays an untouched historical archive (`outputs/h1b_freeze/V1_MANIFEST.json`); H1 v0.3 stays NOT MET.

## 1. Hypothesis

**H1c:** on pairs of worlds in which the chosen branch, its text, c2, the rewards, the candidate set, the depth-2 set, branch membership and every label's occurrence count are **identical**, and **only which unchosen depth-2 state continues to the reward changes**, the pre-answer activation identifies the unchosen continuing state (u2) above:
- chance between the two possible states, and
- every control that cannot see the continuation.

**What a positive result would establish:**
- The pre-answer state carries information identifying the **unchosen** continuation.
- That information is **not reducible** to the choice, c2, 1-hop structure, label frequencies or the label representations available at their arrival.

**What it would not establish:**
- A forward sweep: backward inference from the rewards is still possible (P0).
- Sequential exploration.
- Causal use (H3).

## 2. Task variant (`src/build_h1c.py`; design tool, no pool written)

**World** (route depth 4 on both branches):

```
S -a-> d1 -> d2 -> d3 -> T (reward)        d1 -> e1 -> e2   (e2: dead end, no move, no reward)
```

- **Template:** every non-root edge reads "From X, you can move to Y." The instruction explains dead ends: "Choose the initial action from which the highest reward can be reached. A state with no listed move and no reward is a dead end."
- **Why a dead-end chain of length 2.** A first version used a single dead end e. A residual-cue check showed that u2 then appeared twice in the prompt and e only once, so **an occurrence-count shortcut** could identify u2. With e1 → e2, both d2 and e1 appear exactly twice (arrival + departure).
- **Candidates:** the 10 prompt-derived intermediate labels (d1, d2, d3, e1, e2 of both branches), in canonical order.

**Variants per family:**

| Variant | Definition |
|---|---|
| `base_L` | random line order, with both unchosen depth-2 departure lines (u2 → u3, ue1 → ue2) **after** both u1 lines ("late") |
| `dswap_L` | the **sources** of those two lines are exchanged, so u2 and ue1 swap roles. **Two lines change, one label each** |
| `base_E` / `dswap_E` | the same pair with both departure lines **before** the u1 lines ("early"; secondary: tests whether keys can carry the continuation) |
| `cswap_L` | the same swap on the **chosen** branch (reference: c2 changes, u2 does not) |

## 3. Verified invariants (`src/h1c_design_checks.py` → `outputs/h1c_design/checks.json`; 400 design families, 2,000 worlds, design seed 777; `tests/test_h1c_design.py` 3/3)

| Check, on every pair | `dswap_L` | `dswap_E` |
|---|---|---|
| same model-optimal choice | ✓ | ✓ |
| same rewards | ✓ | ✓ |
| same candidate set (10) | ✓ | ✓ |
| same chosen-branch text, same c2 | ✓ | ✓ |
| same depth-2 set and branch membership (1-hop view) | ✓ | ✓ |
| same occurrence count for every label | ✓ | ✓ |
| u2 changes | ✓ | ✓ |
| number of differing lines | 2 | 2 |
| causal order (late / early) respected | 400/400 | 400/400 |

**`cswap_L`:** the choice is the same, c2 changes and u2 does not (400/400).

**Tokenisation** (Gemma tokenizer, 100 families): **12,000 / 12,000** label occurrences end on a token boundary; prompts are 261–295 tokens.

## 4. Candidate space and chance levels (requirement 4)

| Reference | Single world | Paired (both members correct) |
|---|---|---|
| uniform over the 10 candidates | 0.10 | 0.01 |
| uniform over the 2 possible unchosen states {u2, ue1} | 0.50 | **0.25** (independent guesses) |
| any **deterministic** control that sees only the 1-hop view, counts, the choice or the arrival keys | 0.50 | **0.00** (identical inputs, so identical output, so exactly one member correct; verified on 800/800 design pairs) |

## 5. Controls and what each can reconstruct without the pre-answer query (requirements 2, 3, 5)

**Prompt-only controls (no activations):**

| Control | Inputs | Paired accuracy on late pairs |
|---|---|---|
| c2 / branch elimination | c2, the depth-2 set, branch membership | **0** by construction (verified) |
| B3+ with counts | per-label text features, occurrence counts, choice | 0 by construction (identical features across members) |
| **Full-structure control** (strong; requirement 3) | follows the continuation rules to the reward lines | **1.000** (exact on 2,000/2,000 worlds) |

The full-structure control documents that the answer is **fully reconstructible from the prompt by 3-hop path following**. It is a ceiling for prompt reconstructibility, not a threshold. The question is whether the **activation** carries it.

**Activation-based controls (keys vs query):**

| Control | What it uses | Expected on late pairs | Purpose |
|---|---|---|---|
| C_max (choice + arrival keys, bilinear-matched and linear) | choice, contextual arrival keys | ≈ 0 paired: u2's and ue1's arrival keys sit **before** both departure lines, so by causal attention they are identical across members | can keys alone carry the continuation? |
| key-only | arrival keys | ≈ 0 | — |
| P_perm | pointer with the query from another family (same choice, same variant) | ≤ 0.25 | world-specific query information |
| Early-order pairs (`*_E`, secondary) | the same decoders when departure lines **precede** arrival | C_max may succeed here | measures how much the keys *can* carry when causality allows |
| Position sweep (secondary) | query at the ends of the departure lines, the last rule line, the reward lines, Q_END, ANS | — | where the continuation information first appears (no causal reading) |
| `cswap_L` reference | the same pipeline for c2 on chosen-branch swaps | — | symmetry: chosen vs unchosen continuation |

**Proposed primary rule (to be frozen later):**
- Statistic: paired u2 accuracy of the pointer P (ANS query × arrival keys) on admissible `(base_L, dswap_L)` pairs.
- **H1c MET iff** the lower 95 % family-bootstrap bound of P − max(0.25, C_max, P_perm) > 0.
- The secondary analyses above are reported.

## 6. Power (`src/power_h1c.py` → `outputs/metrics/power_h1c.json`)

**Model:**
- 2 primary pairs per family (two late line orders).
- A pair is admissible iff both worlds are, with per-world admissibility *a*.
- Family ICC 0.15; threshold 0.25; 2,000 simulations per cell.

| Families (pairs at a = 0.75) | δ = 0.05, a = 0.60 / 0.75 / 0.90 | δ = 0.10, a = 0.60 / 0.75 / 0.90 | δ = 0.15 |
|---|---|---|---|
| 300 (337) | 0.35 / 0.49 / 0.63 | 0.87 / 0.97 / 1.00 | ≥ 0.995 |
| **600 (675)** | **0.59 / 0.79 / 0.92** | **0.99 / 1.00 / 1.00** | 1.00 |
| 800 (900) | 0.73 / 0.89 / 0.98 | ≈ 1 | 1.00 |
| 1,000 (1,125) | 0.82 / 0.95 / 0.99 | 1.00 | 1.00 |

**Proposal: 600 TEST families × 2 late pairs = 2,400 worlds.**
- Power is **≥ 0.99 for δ ≥ 0.10** at any plausible admissibility.
- For δ = 0.05 it is 0.59–0.92, depending on admissibility, which must first be measured on DEV.

## 7. Competence conditions

- Dead ends make the task harder; competence is unknown.
- Measured on the new DEV pool from the same capture pass (forced-choice logp), so no separate run is needed.
- **Gate before any TEST:**
  - per-world choice accuracy ≥ 0.70 overall, **and**
  - ≥ 0.60 on `base_L` / `dswap_L`.
- Otherwise H1c **stops** (reported as a negative behavioural result), with no further spending.

## 8. Budget (the $5 BTC cumulative limit is unchanged)

| Step | Size | Expected | Hard cap (timeout) |
|---|---|---|---|
| H1c-DEV capture (competence + training activations): ANS + 10 arrival + 10 departure keys, 5 layers {24, 32, 40, 48, 56} | 400 families × 5 variants = 2,000 worlds; ≈ 2.3 GB | ≈ $0.62 | timeout 850 s → ≈ $0.82 |
| H1c-TEST capture: ANS + 10 arrival keys + position-sweep slots, 3 layers around the DEV-selected layer | 600 families × 4 = 2,400 worlds; ≈ 1 GB | ≈ $0.68 | timeout 850 s → ≈ $0.82 |
| **BTC cumulative** | — | $3.24 → **≈ $4.54** | worst case ≈ **$4.88** (≤ $5) |

**Consequences of the tight budget:**
- No second full attempt is possible for either capture.
- A startup failure costs about $0 (watchdog).
- A late failure would end H1c under the current limit.

## 9. Identifiability risks that remain

1. **Backward inference.** The continuing unchosen state can be found by tracing back from the reward (T ← u3 ← u2). Decodability says nothing about the direction of construction.
2. **Structural summary vs prospection.** A positive result could reflect a reading-time summary such as "which depth-2 states lead to rewards", for both branches, available at the end of the prompt. That is still unchosen-branch information, but not specifically *prospective*. The `cswap` symmetry and the position sweep only describe this.
3. **Template change.** "you can move to" and the dead-end instruction differ from H1, so the H1 / H1b results are not directly comparable.
4. **Decoder retraining.** K = 10 candidates; all decoders are re-fitted on the new DEV pool. The optimism risk on DEV applies again; it is mitigated by the late-pair construction, where the controls are 0 by construction rather than estimated.
5. **Competence.** If admissibility is low, power for δ = 0.05 drops (0.59 at a = 0.60), and the competence gate may stop the study.

## 10. Next decisions (none taken)

| ID | Proposal |
|---|---|
| C-1 | adopt the late dead-end-swap design (length-2 dead-end chains, 10 candidates) and its template |
| C-2 | primary rule: P − max(0.25, C_max, P_perm) > 0 on admissible late pairs; secondaries as in §5 |
| C-3 | sizes: DEV 400 families × 5 variants; TEST 600 families × 2 late pairs |
| C-4 | competence gate (§7) before any TEST |
| C-5 | budget and caps (§8) under $5 |

After approval, the next CPU step would be: positions code for the new template, the DEV pool generator with seed and sealing rules, the DEV capture script, and a protocol draft. Each GPU step would be requested separately.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task (controlled extension resolving a documented confound)
[YES] Kept H1 primary and H2/H3 conditional
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently (no training at all)
[YES] Preserved oracle isolation and train/val/test independence (no pool generated; design seeds only)
[YES] Did not equate token availability with prospective path representation (counts and full-structure controls)
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (CPU design only)
[YES] Preserved negative results and raw logs (first design's occurrence-count cue documented and fixed)
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decisions C-1 … C-5.
```
