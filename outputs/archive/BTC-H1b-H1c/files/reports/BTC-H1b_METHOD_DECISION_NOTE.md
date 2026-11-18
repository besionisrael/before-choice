# BTC-H1b — methodological decision note: separating chosen-branch from unchosen-branch information

**Date:** 2026-11-14 · **Owner decision:** "DÉCISION SCIENTIFIQUE — BTC-H1b". The v1.0 freeze is kept as a historical prepared version; generation of the new TEST is suspended. **CPU reasoning only:** no new TEST, no GPU, no training.

## 1. The identifiability problem (precise statement)

**Current task:**
- At depth 4 there are exactly **two** depth-2 labels, c2 and u2.
- The depth-2 set can be read from the prompt in one hop: the targets of rule lines whose source is an action-line label.

**Consequence:**
- u2 is a **deterministic function of (prompt structure, c2)**.
- Any decoder that knows c2 and the 1-hop structure recovers u2. The DEV elimination control did exactly this: 0.818 paired, against 0.688 for the pointer.
- This holds for **every** world of the current design, including the choice-preserving `upath` pairs. There u2 and u3 swap labels but the chosen branch is identical, and the prompt-derived depth-2 set changes with the prompt, so elimination still succeeds on both members.

**So no accuracy-based contrast on the current task can show unchosen-branch information that is not reducible to chosen-branch information plus reading-time structure.**

## 2. Options and their discriminating power

| Option | What it tests | Discriminating power | Cost / risk |
|---|---|---|---|
| **O1. Add "P − elimination > 0" as a decisive IUT contrast** (current task, frozen v1.0 plus one contrast) | whether P beats the *best available* c2-based reconstruction | **Weak and asymmetric.** P ≤ elimination (likely, given DEV) is uninformative about u2, because elimination is near the ceiling the structure allows. P > elimination could arise just because P uses c2-type information better than the frozen P_c2 decoder; it would still not prove unchosen-specific information | no new data; TEST capture ≈ $0.95; most likely NOT MET with little scientific yield |
| **O2. Within-pair analysis on existing `upath` pairs** (chosen branch byte-identical, u2 changes) | whether the query moves between the members of a pair | **Insufficient alone.** The prompt-derived depth-2 set changes between members, so c2 + structure still predicts both u2s | CPU on existing data; no confirmatory value |
| **O3. New task variant with dead-end distractors plus "dead-end-swap" pairs** (recommended) | whether the pre-answer state carries information about **which unchosen depth-2 state continues the path**, with c2 and the chosen branch fixed | **Strong** (details in §3): in each pair, c2, the chosen branch text, the depth-2 set and the branch membership of every depth-2 label are identical. Only the unchosen continuation differs. A c2-based or structure-based decoder gives **the same answer for both members**, so its paired accuracy is ≈ 0 by construction | new DEV and TEST pools, competence re-check and captures (GPU). **Over the $5 limit if done in full** (§4) |
| O4. Causal interventions (patch or ablate unchosen-branch keys) | causal use | out of scope (H3), and conditional on H1 | — |

## 3. Option O3 in detail (proposal for discussion; nothing generated)

**Task modification** (stays within the two-action symbolic task; a controlled extension that resolves a documented confound, as CLAUDE.md §0.3 allows):
- Each depth-1 state gets **two** successors.
  - One continues the route to the branch terminal.
  - The other is a **dead end**: a state with no outgoing rule and no reward.
- The route to each rewarded terminal stays unique, and the optimal action is unchanged.
- At depth 4 the prompt now has 4 depth-2 labels: c2, c-dead, u2, u-dead.

**Dead-end-swap pairs:**
- Swap which of u1's two successors continues (u2 ↔ u-dead) by relabelling the **source** of the unchosen depth-2 → depth-3 rule line only.
- Line positions, the label multiset, the rewards, the chosen branch text, c2 and the choice are all unchanged; u2 changes.
- **Elimination via c2 and 1-hop structure:** identical inputs for both members, so identical predictions, so paired accuracy ≈ 0.
- **Branch-membership elimination** (u-dead vs c-dead is 1 hop from the action lines): it reduces the candidates to {u2, u-dead}. It still gives the same answer for both members, so paired ≈ 0, single ≈ 0.5.

**Residual confounds that remain and must be controlled:**

| Confound | Control |
|---|---|
| Contextual keys may encode "has an outgoing edge" (a continuation cue available at reading time, by forward or backward propagation) | keep C_max (choice + keys) and key-only. Use **arrival** keys only. Randomise line order so the departure line of u2 is not always before its arrival |
| Backward inference from rewards (P0) | stays a live alternative: decodability does not establish the direction of construction. To be stated, not solved |
| Competence | dead ends add load. Admissibility must be re-measured on a new DEV pool before any freeze; insufficient competence would stop the design |
| Decoder change | K = 8 candidates instead of 6 (6 path labels + 2 dead ends). The decoders must be retrained on the new DEV pool; the H1b decoders do not transfer |

**Primary H1c contrast (sketch):**
- **Paired accuracy on dead-end-swap pairs** of the pointer at ANS, compared with:
  - C_max (choice + keys);
  - the elimination controls;
  - P_perm;
  - and, for chance, 0.25 for a guess between u2 and u-dead.
- **What it can establish:** information in the pre-answer state about the **unchosen** continuation that is not reducible to the choice, to c2, or to reading-time 1-hop structure.
- **What it cannot establish:** that this information was computed forward (a forward sweep), or that it is used (H3).

## 4. Resources (indicative)

| Step | Estimate |
|---|---|
| New DEV pool (CPU) | free |
| DEV competence run + multi-position capture (GPU) | ≈ $0.8–1.0 |
| New TEST pool capture | ≈ $0.9–1.0 |
| **BTC cumulative** | $3.24 → ≈ **$5.0–5.2**: **at or above the $5 limit** |

**Ways to fit the limit:**
- Smaller pools (power would be recomputed).
- Or raise the limit, which is the owner's decision.
- Or drop the separate competence run and capture DEV competence in the same pass.

## 5. Recommendation

1. **Do not run O1 as the confirmatory study.** On the current task it cannot separate chosen from unchosen information, and its likely outcome (NOT MET) would add little.
2. **Keep v1.0 frozen as a historical prepared version** (done; `V1_MANIFEST.json`).
3. **If the question is to be pursued, design O3 ("H1c")** on CPU first:
   - generator extension and dead-end-swap derivation, with tests;
   - a check that elimination and branch-membership controls give identical predictions for both members of every pair;
   - a power analysis with K = 8 and a 0.25 floor;
   - a budget plan under (or with an explicit change of) the $5 limit.
   No GPU until that design is approved.
4. **Interpretation stays bounded either way.** Even a positive H1c result would show decodable information about the unchosen continuation in the pre-answer state. It would not show a forward sweep, sequential exploration or causal use.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task (O3 is a controlled extension resolving a documented confound)
[YES] Kept H1 primary and H2/H3 conditional
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently (no training at all in this step)
[YES] Preserved oracle isolation and train/val/test independence (no TEST generated)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (CPU note only)
[YES] Preserved negative results and raw logs
[YES] Labeled unrun stages NOT RUN and limitations honestly (budget conflict stated)
Any deviation: none.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner choice among O1 / O3 (and the budget question).
```
