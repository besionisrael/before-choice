# D0 — Environment and oracle audit

**Stage:** D0 (CPU, local) · **Date:** 2026-11-13 · **Experiment ID:** BTC-D0D1-001
**Verdict: PASS** (with documented design limitations; no defect found in the starter code)

## 1. Environment

| Item | Value |
|---|---|
| Machine | Local Windows 11 Pro 10.0.26200, CPU only |
| Python | 3.13.5 (MSC v.1943, AMD64), `.venv` |
| Third-party deps | none (stdlib only); `requirements.txt` records this |
| Git | project is **not** a git repository → provenance by SHA-256 (`outputs/logs/manifest_D0D1.json`) |
| GPU / cloud | none used, $0 |

## 2. Starter inventory (as received)

| File | SHA-256 |
|---|---|
| `generate_worlds.py` | `ad9b783e…f209d435` |
| `oracle.py` | `6b0f6f2d…5ee640` |
| `tests/test_oracle.py` | `a86c4527…573584` |
| `data/worlds.jsonl` (shipped, 50 rows) | `5d63d87e…de7c9a` (copy kept at `reports/_orig/worlds_shipped.jsonl`) |

Starter files were **not modified**.

## 3. Tests and reproduction

- `python -m unittest discover -s tests -v` → original 2 tests **OK** (now 18 tests total, all OK; log `outputs/logs/unittest_D0D1.log`).
- `python generate_worlds.py --pairs 25 --seed 42 --output data/generated/pilot_v0.jsonl` → 50 worlds.
  - Re-running gives a **byte-identical** file (determinism PASS).
  - Content equals the shipped `data/worlds.jsonl` after LF normalisation (both `5d63d87e…`). Byte hash on Windows differs (`411a9bae…`) only because the legacy writer emits CRLF. The v1 writer forces LF.

## 4. Independent oracle re-check

`src/independent_check.py` does **not** import `oracle.py`: it enumerates *all* simple routes start→terminal by DFS over an adjacency map (vs. the starter's unique-successor tracing), then checks:

| Check | pilot_v0 (50 shipped-seed worlds) | pilot_v0 stress (600 worlds, depths 2–4 × 100 seeds) |
|---|---|---|
| exactly one complete route per action | PASS | PASS |
| no cycles | PASS | PASS |
| terminal reachable, unique in-edge per state | PASS | PASS |
| no reward ties | PASS | PASS |
| oracle paths / rewards / optimal action match | PASS | PASS |
| branches disjoint | PASS | PASS |
| base/cf optimal action flips | PASS (25/25) | PASS |
| prompt leakage (`optimal`, `oracle`, `paths`, assembled ≥3-state route) | none | none |

Optimal action in pilot_v0: A 25 / B 25. Depth (pairs): 2→9, 3→7, 4→9 (not balanced).

## 5. Limitations of the starter (not concealed)

1. **Linear, disjoint branches only.** Each action has one chain; no branching after the root. This makes the task solvable by forward chaining *or* backward chaining from each terminal; behaviour cannot discriminate these.
2. **Terminal rewards are explicit in the prompt**, and the high-reward terminal is trivially identifiable. A "find max-reward terminal, trace predecessors back to S" strategy solves every item. (D1 adds a decoy control against the *max-reward anchor* variant of this shortcut; it does not rule out backward chaining in general.)
3. **The cf manipulation touches only the two final edges.** The minimal pair differs in exactly two target tokens, located in two prompt lines; the ordered path differs only at the final depth. Probes for "depth-d state" will have identical targets in base/cf for d < depth — useful as a control, but only the final depth is manipulated.
4. **Labels vary across samples** (random 4-char A–Z0–9). Good against memorised labels, but tokenisation length varies per label (not yet measured — needs the Qwen tokenizer; see PROTOCOL_FREEZE decision D-3).
5. **Prompt length is a deterministic function of depth** (394/468/542 chars at depth 2/3/4) → any depth decoder must be compared within depth.
6. **Small reward gaps dominate** (sampling two distinct values from 1–9). Not a bug; a difficulty covariate.
7. Prompt grammar "1 points" kept unchanged for comparability.
8. Data rows co-locate `prompt` and `oracle`; isolation must be enforced by inference code (D1 provides a prompt-only file).

## 6. Gate

Unit tests pass; no oracle/prompt leakage detected; determinism verified. **D0 PASS.**

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
Any deviation: none
Next step: D1 dataset (done, see 01_dataset_audit.md)
```
