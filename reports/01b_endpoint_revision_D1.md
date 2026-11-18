# D1b — Targeted revision of the primary H1 endpoint (D-1)

**Date:** 2026-11-13 · **Stage:** CPU only · **Requested by:** owner review of PROTOCOL_FREEZE v0.1
**Scope:** research question, model, H1–H3 hierarchy and pilot_v1 data **unchanged**. This revises only *how* H1 is measured.

## 1. The shortcut — verified and stronger than reported

**Claim reviewed:** with correct choice, the unchosen terminal is simply the lower-reward outcome.

**Formal statement.** Each world has two real terminals t_A, t_B with visible rewards r(t_A) ≠ r(t_B), and decoy worlds have one extra unreachable terminal with reward greater than both. The non-optimal branch n = argmin over a of r(path_a[-1]). Therefore:

- path_n[-1] = argmin over *all* visible terminals of r(t). The decoy never wins this argmin because its reward is the maximum.
- If the model chooses correctly, unchosen = n, so the "unchosen terminal" equals the label with the lowest printed reward.
- If the model chooses wrongly, unchosen = optimal, so the unchosen terminal is the argmax over real terminals: again a function of answer + rewards.

**Empirical check** (`tests/test_path_cf.py::test_nonoptimal_terminal_is_lowest_visible_reward_everywhere`): holds in **300/300** pilot_v1 worlds, across all variants. The v0.1 primary endpoint can therefore be satisfied by reading the reward lines, **with neither graph traversal nor the answer**. The base/cf pairing does not protect it: reward labels are identical within the pair, but the target terminal is constant too, because only *which branch* reaches it flips.

**A related shortcut was also found.** The depth-1 state of each action is printed verbatim in one line (`At S, action X leads to Y.`), so it is a one-hop lexical lookup (`test_depth1_state_is_one_hop_lexical`).

**Consequence.** Targets at depth 1 and at the final depth are reachable from a lexical **anchor** (the action line, or the reward line) in ≤1 hop. Only intermediate states require composing relations.

## 2. New counterfactuals: unchosen trajectory changes, choice is identical

`src/build_path_cf.py` builds **pilot_v1_pathcf**, a new file. `pilot_v1.jsonl` is read-only, and its hash is checked and recorded as `f9073bdf…`. The new worlds are derived from each family's `base` and `cf` worlds, keeping the same labels, line positions, rewards, prompt length, split and optimal action:

| Variant | Manipulation | Depths | n |
|---|---|---|---|
| `upath_{base,cf}` | swap two intermediate states **inside the non-optimal branch** (depth 3: route depths 1↔2; depth 4: 2↔3, root line untouched) | 3, 4 | 40 |
| `cpath_{base,cf}` | same, inside the optimal branch (chosen-path reference) | 3, 4 | 40 |
| `xswap_{base,cf}` | exchange the depth-2 state between branches (depth 1 for depth-2 families) → branch membership changes | 2, 3, 4 | 60 |

Total 140 worlds. Splits: 84 train / 28 val / 28 test, inherited by family.

**Verified invariants** (10 tests, all pass, 28/28 project-wide):
- Independent oracle passes on all 140 worlds.
- Optimal action, rewards, terminals, label multiset, prompt length and line count are identical to the source.
- Only the targeted branch changes (`upath` leaves the optimal path byte-identical, and vice versa for `cpath`).
- The depth-2 target changes in every `upath` pair.
- Depth-4 `upath` keeps the root line.
- **Any pair-invariant decoder has paired accuracy exactly 0.** "Pair-invariant" means any function of answer, rewards, label inventory or length. It outputs the same label for both members while the target differs.
- **Tokenisation (D-3):** all 40 `upath`, 40 `cpath`, 60 `xswap` and 220 base/cf-type pairs have identical token length under the official chat template.

## 3. Residual confound (not hidden): endpoint + elimination

Chains are short, so every state lies near an anchor:

| Depth | Target (non-optimal, route depth 2) | Hops from S-anchor | Hops from reward-anchor (terminal) |
|---|---|---|---|
| 3 | d2 | 2 | **1** (predecessor of the lowest-reward terminal) |
| 4 | d2 | 2 | 2 |

- In depth-3 `upath`, the target can be found as "the predecessor of the lowest-reward terminal", i.e. one backward hop. This is a *backward* computation, not forward simulation.
- Only **depth-4, route-depth 2** is ≥2 hops from every lexical anchor in the pilot.
- Even there, a combination of one-hop lookups from both ends plus elimination over the branch's labels remains possible. That combination is itself relational (two composed hops), but it is not evidence of forward order specifically.
- Longer chains (depth ≥5) would reduce this, but changing the depth range is a task-design change. It is listed as **decision D-7** and **not implemented**.

## 4. Proposed revised primary endpoint (D-1′)

**Unchosen intermediate-state identification under choice-preserving path counterfactuals.**

- **Population:** `upath` pairs (source world, `upath` world), depth 4, locked test split. Pre-specified secondary: depth 3 + 4 pooled.
- **Branch:** the oracle non-optimal action.
  - Primary analysis: pairs where the model chose the optimal action on **both** members, so the branch really is unchosen.
  - Also reported: all pairs, conditioned on the action letter.
  - Also reported: whether the `upath` manipulation changed the model's choice (it should not).
- **Target:** the label at route depth 2 of that branch.
- **Candidates:** all intermediate labels of both branches, 2·(depth−1) = 6 at depth 4. This gives matched inventory and per-world chance 1/6.
- **Metric:** paired accuracy, meaning the probe picks the correct label in **both** members. Chance ≈ 1/36 for a random probe; **exactly 0** for any pair-invariant decoder.
- **Probe:** candidate-conditioned linear probe on last-pre-answer residual stream. Layer and regularisation are chosen on train/val; test is evaluated once.
- **Baselines to beat** (all v0.1 baselines B1–B5 kept):
  - B4 (answer + rewards), now provably 0 on this metric.
  - **B6 anchor-and-eliminate:** the same probe trained only on targets at depth 1 and the terminal (anchor-adjacent), then applied by elimination.
  - **B7 chosen-path reference:** the same metric on `cpath`. It is reported alongside the unchosen result rather than as a threshold: H1 needs both branches, and unchosen ≪ chosen is informative.

**Secondary endpoints:**
- `xswap` branch-membership paired accuracy, all depths. Depth 2 is flagged lexical.
- Ordered pairwise `before(s_i, s_j)` on `upath`.
- The v0.1 base/cf terminal endpoint, kept but **demoted to a manipulation/positive-control check**, explicitly not H1 evidence.

## 5. Power caveat

- The pilot test contains 2 depth-4 families, i.e. 4 `upath` depth-4 pairs.
- **The pilot cannot test D-1′ confirmatorily.** It can test feasibility, alignment and choice-invariance.
- A confirmatory test needs the main dataset (D-5, deferred by the owner). This is stated so that pilot numbers are never presented as H1 evidence.

## 6. Artifacts

`src/build_path_cf.py`, `tests/test_path_cf.py`, `src/tokenizer_audit.py`, `data/generated/pilot_v1_pathcf.jsonl` (`571444a8…`), `data/generated/pilot_v1_pathcf_model_inputs.jsonl`, `data/splits/pathcf_{train,val,test}.jsonl`, `data/generated/pilot_v1_pathcf_manifest.json`, `outputs/metrics/tokenizer_audit.json`.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently (tokenizer only, revision pinned)
[YES] Preserved oracle isolation and train/val/test independence
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (D-2, D-3 approved; no GPU)
[YES] Preserved negative results and raw logs
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none. New variants are additions in a separate file; pilot_v1 unchanged.
Next step: owner validation of D-1′ (and D-7 decision)
```
