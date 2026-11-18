# D1 — Controlled dataset audit (`pilot_v1`)

**Stage:** D1 (CPU) · **Date:** 2026-11-13 · **Verdict: PASS** (gate met; design choices awaiting owner sign-off in `PROTOCOL_FREEZE.md`)

## 1. What was built

`python src/build_dataset.py --families-per-depth 10 --seed 20261113`

- **30 graph families** (10 per depth 2/3/4) × **10 variants** = **300 worlds**. Size matches the planned "200–300 worlds GPU competence pilot".
- Prompt template is the **unchanged pilot_v0 template** (`oracle.format_prompt`), so v0 and v1 are directly comparable. `pilot_v0` is preserved as `data/generated/pilot_v0.jsonl`.
- Every state label is **globally unique** across the whole dataset (no label reused between families or splits).

| Variant | Construction | Purpose |
|---|---|---|
| `base` / `cf` | v0 rule: swap targets of the two final transitions | **Primary** minimal pair; optimal action flips, label multiset, rewards, prompt length and line count constant; prompts differ in exactly 2 lines |
| `alias_base/_cf` | all non-start labels renamed, same line order | label-identity control, held-out-alias generalisation |
| `swap_base/_cf` | A↔B letters exchanged, everything else byte-identical | dissociate answer letter from path; A/B bias |
| `shuffle_base/_cf` | same graph, different rule order | position/order shortcut control |
| `decoy_base/_cf` | + unreachable chain (same depth) ending in reward 10–12 > both real rewards | breaks "anchor on max reward and trace back"; forward simulation unaffected |

Oracle (private) per world: ordered `paths` for **both** A and B (chosen and unchosen), `states_by_depth`, `rewards`, `optimal_action`, `decoy_chain` where relevant.

## 2. Files and hashes (LF, platform-independent)

| File | Rows | SHA-256 |
|---|---|---|
| `data/generated/pilot_v1.jsonl` | 300 | `f9073bdf…377907` |
| `data/splits/train.jsonl` | 180 (18 fam.) | `1b46f44c…3cf1` |
| `data/splits/val.jsonl` | 60 (6 fam.) | `a172856b…20b1` |
| `data/splits/test.jsonl` **(LOCKED)** | 60 (6 fam.) | `34ab3841…13e8c3` |
| `data/generated/pilot_v1_model_inputs.jsonl` | 300, prompt only, **no oracle/rewards field** | `0fa40be3…1011` |

Full hashes: `data/generated/pilot_v1_manifest.json`, `outputs/logs/manifest_D0D1.json`.

## 3. Split strategy

Family-level split, stratified by depth, 60/20/20 → per depth 6/2/2 families. All 10 variants of a family (including alias and cf members) stay in one split. Verified: no family and no state label crosses splits.

## 4. Balance

| Split | depth 2/3/4 (families) | optimal in `base` A/B | optimal in base+cf |
|---|---|---|---|
| train | 6/6/6 | 11/7 | 50/50 |
| val | 2/2/2 | 0/6 | 50/50 |
| test | 2/2/2 | 4/2 | 50/50 |

Whole dataset: base optimal is exactly 5A/5B per depth; every paired set is 50/50 by construction. Within-split base imbalance (val all-B) is a small-sample artefact; analyses must be **paired** (base+cf, or base+swap), never base-only.

Reward gap (base, n=30): 1→8, 2→11, 3→2, 4→1, 5→3, 6→3, 7→2 (skewed small, **not stratified**).

## 5. Tests (`tests/test_dataset.py`, 16 tests, all OK)

Independent-oracle audit on all 300 v1 worlds and 600 v0 worlds; family completeness; cf flip for all 5 pair types; label multiset/reward/length invariance; cf differs in exactly 2 prompt lines; swap differs only in action letters; shuffle = same edge set, different order; alias isomorphic with disjoint labels; decoy unreachable and max reward; depth/optimal balance; trivial heuristics; split leakage (family and label); split sizes; determinism (same seed → same hash, different seed → different); committed files == generator output; model-input file has no oracle.

**Trivial-baseline result (primary base+cf, n=60):** always-A = 0.50; "first action line listed" = 0.50; "alphabetically-first root label" = 0.50. These are exactly 0.50 because any heuristic blind to the final edges answers identically on base and cf. This is a property of the paired design, not evidence the task is hard to shortcut by other means.

**Bug found and fixed during D1:** first draft resampled decoy reward/positions separately for `decoy_base` and `decoy_cf` (so they were not a minimal pair). Caught by test, fixed (drawn once per family).

## 6. Known limitations / open confounds

1. **Backward chaining is not excluded.** In linear chains, tracing back from each terminal is as effective as forward simulation; decoy only removes the max-reward *anchor*. A representation of "ordered path" in activations could arise from either; H2 must treat backward construction as a live alternative.
2. **Depth ↔ prompt length** perfectly confounded; compare within depth.
3. **cf manipulates only the last edge**: depth-d states for d < depth are identical in base/cf (control), only final-depth membership changes.
4. **Decoy reward values (10–12) are out of the base reward range (1–9)**; a model could ignore large numbers. Also decoy prompts are longer.
5. **Tokenisation not yet checked** (needs Qwen tokenizer files — decision D-3). Labels of 4 random chars will tokenise into variable numbers of tokens; base/cf token alignment should still hold because the cf only permutes existing labels, but must be verified before any patching.
6. Pilot size (6 test families) is a **feasibility pilot**, not powered for H1 claims. A main dataset (~2,000 worlds) would be generated only after pilot competence, with owner approval.
7. Only linear disjoint topologies; no shared states, no in-branch choice.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently (no LLM used)
[YES] Preserved oracle isolation and train/val/test independence
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (none made)
[YES] Preserved negative results and raw logs
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none (new control variants are additions inside the same task; pilot_v0 preserved; listed for sign-off in PROTOCOL_FREEZE.md)
Next step: owner review of PROTOCOL_FREEZE.md
```
