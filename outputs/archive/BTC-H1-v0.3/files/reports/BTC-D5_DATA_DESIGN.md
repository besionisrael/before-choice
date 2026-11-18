# BTC-D5_DATA_DESIGN — PROBE-DEV and LOCKED-TEST pools (design, NOT generated)

**Date:** 2026-11-13 · **Status:** design for owner validation. **No PROBE-DEV or LOCKED-TEST data has been generated.** No GPU, no probe.
**Model of record:** Gemma-4-31B-it @ `842da379…` (COMPETENCE PASS, D-1′ FEASIBLE; BTC-V2-C1).
**Invariants:** H1–H3, D-1′, L1–L4 and the backward-reconstruction alternative (P0) are unchanged. D-7 (depth 5–6) is **excluded** from the primary analysis (owner decision 4).

## 1. What the data must support

| Analysis | Needs |
|---|---|
| **Primary H1 (D-1′):** unchosen depth-2 state, depth-4 `upath` pairs, admissible pairs only, paired accuracy vs B1/B2/B3/B6 | depth-4 families with base, cf, upath_base, upath_cf; the model's own choices (admissibility) |
| B7 chosen-branch reference (always reported separately) | cpath_base, cpath_cf |
| Pre-specified secondary: depth 3+4 pooled | depth-3 families with the same variants |
| Secondary: `xswap` membership, `before(s_i, s_j)` order, per-depth reconstruction | xswap_base, xswap_cf (+ all worlds) |
| Generalisation and controls (H1 rating "moderate" needs alias) | alias, shuffle, swap |
| Decoy: **secondary control, distinct from the primary** (owner decision 2) | decoy_base, decoy_cf (v1/v2 generator, max-reward decoy) |
| Probe training and model selection | PROBE-DEV only, family-level train/val |

## 2. Power analysis (`src/power_d1prime.py` → `outputs/metrics/power_d1prime.json`)

**Model:**
- The unit of inference is the family.
- Each depth-4 family yields 2 `upath` pairs (sources base and cf). Each pair is admissible with probability **a = 0.75**, the planning value observed in BTC-V2-C1 (45/60).
- Probe and baseline "paired-correct" outcomes have a family random effect (ICC 0.05 or 0.15) and a probe/baseline correlation ρ ∈ {0, 0.3}.
- Decision rule: the 95 % family-cluster CI of (probe − baseline) has lower bound > 0.
- 2,000 simulations per cell. The normal approximation was checked against a real family bootstrap (F = 150, +0.15: estimate 0.173, bootstrap CI [0.086, 0.262]).

**Reference baseline levels:**
- **0.25** is the ceiling of B6 (anchor-and-eliminate with perfect anchors, then a coin flip between the two depth-2 labels). It is the most demanding baseline.
- **0.10** is an intermediate value.
- **0.03 ≈ 1/36** is chance.

**Power, worst case (ICC 0.15, ρ = 0):**

| Families F (depth 4) | Expected admissible pairs | Δ = +0.10 over 0.25 | Δ = +0.15 over 0.25 | Δ = +0.20 over 0.25 | Δ = +0.10 over 0.10 |
|---|---|---|---|---|---|
| 90 | 135 | 0.40 | 0.71 | 0.92 | 0.65 |
| 150 | 225 | 0.62 | 0.90 | 0.99 | 0.87 |
| **200** | **300** | **0.73** | **0.97** | **1.00** | **0.94** |
| 300 | 450 | 0.89 | 1.00 | 1.00 | 0.99 |

**Proposal: LOCKED-TEST has 200 depth-4 families.** This gives ≥ 0.97 power for a +0.15 advantage over the strongest plausible baseline, and 0.73 for +0.10.
- The intersection-union rule (probe must beat **each** threshold baseline, §5 of the H1 freeze) needs no multiplicity correction. Its power is governed by the strongest baseline, which the table assumes.
- Effects below +0.10 over B6 will likely be reported as `unsupported` or `inconclusive`. This is stated in advance, not tuned later.

## 3. Pools, sizes, seeds and partitions (pre-registered here; generated only after validation)

| Pool | Seed | Family ids | Depth-4 families | Depth-3 families | Variants / family | Worlds | Partition |
|---|---|---|---|---|---|---|---|
| **PROBE-DEV** | **2026111401** | F20000–F20299 | 200 | 100 | 16 (below) | **4,800** | family-level, stratified by depth and optimal action: **train 75 % / val 25 %** (150 + 75 train, 50 + 25 val) |
| **LOCKED-TEST** | **2026111402** | F30000–F30299 | **200** | 100 | 16 | **4,800** | single sealed pool |

**The 16 variants per family** are the v1/v2 generator variants: base, cf, swap_base, swap_cf, alias_base, alias_cf, shuffle_base, shuffle_cf, decoy_base, decoy_cf, upath_base, upath_cf, cpath_base, cpath_cf, xswap_base, xswap_cf.

**Excluded from both pools:**
- **D-8 decoys:** a separate exploratory pool, if ever approved; see BTC-D8_CONTROL_DESIGN.
- **Depth 2:** no D-1′ target; not needed.
- **Depth 5–6:** D-7 reserved for a distinct exploratory study.

**Generator rules:**
- Code unchanged (`build_dataset.build_family`, `build_path_cf.derive`, `oracle.format_prompt`).
- Same reward range 1–9 and same decoy rule (10–12) as v1/v2, for comparability.
- Optimal A/B exactly balanced per depth and per partition.
- **Labels disjoint** from v1, v2 and each other pool. Exclusion sets are regenerated in memory from their seeds, so no file is read.
- Every state label is private to one family, so no family or label crosses pools or the train/val split.

## 4. Locked-test handling

1. **Generation:** one CPU script `src/build_d5.py` (to be written after validation) builds both pools in one deterministic pass.
   - It writes PROBE-DEV in clear.
   - It writes LOCKED-TEST to `data/d5/locked_test/` together with its sha256 manifest.
   - **No development code may import or read that directory.** A test enforces this by scanning source files.
2. **Seal:** the LOCKED-TEST sha256 values are recorded in the H1 freeze before any activation exists. Any later change voids the test.
3. **Use:**
   - The LOCKED-TEST model-input file is sent once to the D3-TEST capture run (separate approval). Its activations are stored under `outputs/d3/locked_test/` and stay sealed.
   - Scoring code refuses to load them unless `outputs/d3/UNLOCK_<approval-id>.json` exists, a file written only after the owner approves the final analysis.
   - During development only hashes are computed, never contents analysed.
4. **One evaluation:** the frozen probe is applied once. A second evaluation, for any reason, makes the result exploratory.

## 5. Inference rules (summary; full text in the H1 freeze proposal)

- **Primary population:** LOCKED-TEST depth-4 `upath` pairs where the model chose the oracle-optimal action on **both** members, from the same D3 forward pass (first-token argmax over A/B with the C1 tie rule).
- Also reported, never pooled with the primary: all pairs conditioned on the action letter, and the chosen-branch `cpath` reference.
- **Format violations** (top-1 ∉ {A, B}, or A+B mass < 0.5) are counted and listed by world. They **do not** change the pre-registered admissibility rule, which uses the forced A/B choice. The primary endpoint is not modified after observation (owner decision 2).
- **CIs:** 95 % family bootstrap, 10,000 draws, seed 0. Primary decision by intersection-union over B1, B2, B3, B6. Secondaries Holm-corrected.

## 6. Cost and storage implications (for D3; detailed in BTC-D3_ACTIVATION_DESIGN)

- 4,800 worlds per pool × 62 vectors × 5,376 × 2 bytes (bf16) ≈ **3.2 GB** per pool.
- Estimated GPU time per pool ≈ 10–13 min on A100-80GB; expected cost ≈ $0.7–0.9 per pool. Each pool is a separate run with a separate approval.

## 7. Decisions requested

| ID | Decision |
|---|---|
| D5-1 | LOCKED-TEST = 200 depth-4 + 100 depth-3 families (4,800 worlds); power rationale §2 |
| D5-2 | PROBE-DEV = 200 depth-4 + 100 depth-3 families, 75/25 family split |
| D5-3 | Seeds 2026111401 / 2026111402, id ranges, label disjointness, 16 variants, D-8 and depth 5–6 excluded |
| D5-4 | Locked-test sealing and unlock procedure (§4) |
| D5-5 | Authorise writing `src/build_d5.py` + tests and generating both pools on CPU (LOCKED-TEST sealed at generation) |

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (design: sealed test, disjoint labels, family splits)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (none executed)
[YES] Preserved negative results and raw logs
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none. Only C1 admissibility rate used as a planning input (no new data looked at).
Next step: owner validation of D5-1 … D5-5.
```

---

## Revision v0.2 (2026-11-13) — owner conditions on D5

**Owner decision:** D5 is validated under conditions. Sizes and seeds are kept, PROBE-DEV is authorised, and **LOCKED-TEST is not generated** until the revised H1 freeze is validated.

### R1. Extended power analysis (baselines above 0.25)

Source: `outputs/metrics/power_d1prime_extended.json`. Worst case ICC 0.15, ρ = 0, admissibility 0.75, 1,500 simulations per cell.

| Effective baseline | Δ | F = 120 | 150 | **200** | 250 | 300 | 400 |
|---|---|---|---|---|---|---|---|
| 0.25 | +0.10 | 0.52 | 0.60 | **0.76** | 0.84 | 0.88 | 0.95 |
| 0.25 | +0.15 | 0.83 | 0.90 | **0.97** | 0.99 | 1.00 | 1.00 |
| 0.30 | +0.10 | 0.48 | 0.57 | **0.71** | 0.80 | 0.86 | 0.93 |
| 0.30 | +0.15 | 0.79 | 0.86 | **0.95** | 0.99 | 0.99 | 1.00 |
| 0.35 | +0.10 | 0.45 | 0.53 | **0.67** | 0.77 | 0.83 | 0.91 |
| 0.35 | +0.15 | 0.80 | 0.87 | **0.94** | 0.97 | 0.99 | 1.00 |
| 0.40 | +0.10 | 0.44 | 0.52 | **0.65** | 0.74 | 0.81 | 0.91 |
| 0.40 | +0.15 | 0.79 | 0.87 | **0.93** | 0.98 | 0.99 | 1.00 |
| 0.50 | +0.10 | 0.43 | 0.54 | **0.62** | 0.72 | 0.80 | 0.91 |
| 0.50 | +0.15 | 0.78 | 0.86 | **0.94** | 0.98 | 0.99 | 1.00 |
| any of the above | +0.20 | ≥ 0.95 | ≥ 0.98 | **1.00** | 1.00 | 1.00 | 1.00 |

**Reading:**
- With 200 depth-4 families, an advantage of **+0.15 over any effective baseline up to 0.50** is detected with power ≥ 0.93.
- An advantage of **+0.10** is detected with power 0.62–0.76, depending on the baseline. Such effects may be reported as `unsupported`. This sensitivity limit is declared now.

### R2. Insufficiency rule (pre-registered)

Power as a function of admissible pairs (baseline 0.25, Δ = +0.15, ICC 0.15):

| Admissible pairs | 100 | 120 | 150 | **180** | 200 | 240 | 300 |
|---|---|---|---|---|---|---|---|
| Power | 0.60 | 0.64 | 0.75 | **0.82** | 0.85 | 0.92 | 0.97 |

**Rule:** if LOCKED-TEST yields **fewer than 180 admissible depth-4 `upath` pairs**, or **fewer than 100 families contributing at least one admissible pair**, the H1 outcome is **`INSUFFICIENT`**.
- No H1 rating is given and no L-level is attributed.
- Estimates are reported descriptively with their CIs.
- The minimal detectable effect at the observed count is stated.
- The expected count at the C1 admissibility rate is 300 pairs; the rule triggers if admissibility drops below 0.45.

### R3. Sealing procedure, strengthened (`src/sealed.py`, tests in `tests/test_d5_dev.py`)

1. **Generation:**
   - `src/build_d5.py --pool LOCKED-TEST` **refuses to run** without an owner approval file (tested).
   - When approved, it writes only to `data/d5/locked_test/`, then calls `seal_dir`. This records the sha256 of every file in `SEAL_MANIFEST.json` and **marks every file read-only**.
2. **Registration:** the seal manifest hashes are copied into the H1 freeze document and the change log **before** any activation of LOCKED-TEST exists.
3. **Development access:** only `hash_only()` (integrity, no content) is allowed. A test scans every `src/*.py` and fails if any module other than `build_d5.py` and `sealed.py` references the sealed directory.
4. **Analysis access:** only through `sealed_read()`. It requires `outputs/d3/UNLOCK_<seal-id>.json` (fields: owner approval id, UTC date, scope), re-verifies the sha256 of the file read, and **appends an access record** (UTC, file, purpose, approval id) to `SEAL_ACCESS_LOG.jsonl`. Any hash mismatch raises and voids the test.
5. **Activations of LOCKED-TEST** (D3-TEST, future) are written to a separate sealed directory with the same mechanism and the same unlock file.
6. **One evaluation:** the access log makes repeated evaluations visible. Any analysis after the first is labelled POST HOC.
7. `data/d5/locked_test/` and `outputs/d3/` are git-ignored, so activations and test data are never published.

### R4. PROBE-DEV generated (CPU, authorised)

`python src/build_d5.py --pool PROBE-DEV` (seed 2026111401):

| Item | Value |
|---|---|
| Worlds / families | 4,800 / 300 (16 variants each; ids F20000–F20299) |
| Depth × split (families) | depth 4: train 150 / val 50; depth 3: **train 76 / val 24**; optimal A/B exactly balanced in every cell |
| sha256 | worlds `d26b5127…`, model inputs `a3c81436…`, train `4058e539…`, val `b764d3e8…` |
| Audits (`tests/test_d5_dev.py`, 10 tests) | files = generator = manifest; independent oracle 0 violations; template; family/label isolation; labels disjoint from v1 + v2; model inputs oracle-free; 400 depth-4 primary pairs constructible; sealing round trip (refusal, unlock, access log, tamper detection) |

The depth-3 split is 76/24, not 75/25: `round(0.75 × 50) = 38` families per optimal action. This is a deterministic consequence of the pre-registered rule.

**LOCKED-TEST:** not generated (generation refused, verified by test).
