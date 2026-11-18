# BTC-H1 confirmatory protocol — v0.3 **FROZEN** (2026-11-13)

**Date:** 2026-11-13 · **Status: FROZEN.**
- The owner approved H1-R2′, H1-R4′, H1-R5′, and H1-R7 conditional on the two verifications in §11, both PASS.
- Code and test hashes, and the LOCKED-TEST seal hashes, are in §12 and `reports/BTC-H1_FREEZE_MANIFEST.json`.
- **No probe has been trained on real activations, no activation captured, nothing unlocked.**

*The text below §0–§10 is the v0.3 proposal as approved. §3 and §5 are made more precise by §11, without any criterion changing.*

**Owner decisions on v0.2:**
- **Approved:** H1-R1, H1-R3, H1-R6, D3-R1.
- **Approved with corrections:** H1-R2 (order test), H1-R4 (baseline specifications).
- **Pending:** H1-R5 (feasibility), H1-R7 (final freeze).

**Changes in v0.3 (only):**
- §6.3: the order test is fully specified, at a single layer ℓ\*.
- §5: baselines are specified operationally.
- §4 and §4b: benchmark result. **The v0.2 grid is infeasible**, so a **new deterministic grid** is submitted, with the budget definition, instrumentation and overrun behaviour.

**Unchanged:** H1, D-1′ and its confirmatory population; the separation of H1 MET from L2 (§6.2–6.3); training-set rule §3; insufficiency rule; L1–L4; the backward-reconstruction alternative.

## 0. Unchanged commitments

- This freeze concerns **H1 only.** The maximum attainable level is **L2**; L4 is untested by design.
- **Backward reconstruction stays a live alternative** (P0). No result here identifies the direction of construction.

## 1. Model, representation, data (unchanged)

| Item | Value |
|---|---|
| Model | Gemma-4-31B-it @ `842da379…`, bf16, first-token A/B (236776 / 236799) |
| Representation (D3-R1 approved) | 62 vectors at `<channel\|>` (61 HF + raw block-60 pre-norm), `mean_emb`, frozen e(c) |
| DEV | PROBE-DEV, sha `d26b5127…` |
| TEST | LOCKED-TEST, seed 2026111402: **generated and sealed at freeze** (hashes in §12) |

## 2. H1 primary endpoint (unchanged)

- **Population:** depth-4 `upath` pairs (base→upath_base, cf→upath_cf), **admissible** (forced choice oracle-optimal on both members; a tie is not admissible).
- **Target:** route-depth-2 label of the unchosen branch. **Candidates:** the 6 intermediate labels. **Metric:** paired accuracy.
- Chosen-branch (B7) and all-pairs results are reported separately.

## 3. Training-set rule (approved; unchanged)

- **Training and selection worlds:** non-decoy PROBE-DEV worlds of depth 3–4 in which the model's forced choice is optimal. The same set is used for the probe and for every baseline.
- **Decoy worlds** serve only as a distinct secondary control.
- **Format violations** are counted, never used for inclusion or exclusion.

**Slot convention (all decoders).** Candidates are ordered in 6 slots relative to the **model's forced choice**: c1, c2, c3 (chosen branch, depths 1–3) and u1, u2, u3 (unchosen branch). The decoder input is only the activation and the candidates' frozen embeddings. **No oracle answer is ever an input.** The oracle is used only to compute training and evaluation labels.

## 4. Probe and the new deterministic grid (H1-R5 resubmitted)

**Architecture (unchanged):**
- `s(c) = z_ℓᵀ U Vᵀ ê(c)`, softmax over the 6 candidates, cross-entropy.
- z_ℓ and ê(c) are standardised with DEV-train statistics.
- AdamW, learning rate 1e-3, batch 256, at most 60 epochs, early stopping on DEV-val loss with patience 5.
- Implementation: `src/h1_probe.py`.

**New grid** (chosen on cost grounds only, before any activation exists; deterministic):

| Factor | v0.2 (infeasible) | **v0.3 (proposed)** |
|---|---|---|
| Vector index ℓ | all 62 | **{0, 4, 8, …, 56, 60, 61}: 17 values.** Includes the embeddings (0), the post-norm final state (60) and the raw block-60 pre-norm state (61) |
| Rank r | {8, 32, 128} | **{8, 32}** (r = 128 costs ≈ 4–10× more per epoch) |
| Weight decay λ | {1e-4, 1e-3, 1e-2, 1e-1} | unchanged |
| Seeds | {0, 1, 2} | unchanged |
| Primary fits | 2,232 | **408** |

- **Selection (primary):** the mean over the 3 seeds of DEV-val paired accuracy on DEV-val admissible depth-4 `upath` pairs. Ties: smaller r, then larger λ, then lower ℓ. Freeze (ℓ\*, r\*, λ\*) and E\* = the rounded mean best epoch.
- **Single final probe:** refit on all DEV training-rule worlds with (ℓ\*, r\*, λ\*), E\* epochs, seeds 0, 1 and 2. Average the 3 softmax outputs. Hash the weights before unlock.

### 4b. Benchmark and budget (`src/bench_h1_grid.py` → `outputs/metrics/bench_h1_grid.json`; synthetic arrays only)

**Synthetic data with the real shapes:** d = 5,376; label table 5,800 × 5,376; 2,850 train and 930 val worlds; 6 candidates. Measured on this machine (16 logical CPUs, torch 2.5.1 CPU).

| Threads | r = 8 CPU-s/epoch | r = 32 CPU-s/epoch | r = 128 CPU-s/epoch |
|---|---|---|---|
| 1 | ≈ 0.2 | 0.23 | 4.93 |
| 4 | 0.20 | 0.35 | 4.23 |
| 16 | 0.52 | 0.56 | 2.02 |

**Full-programme cost** (all heads and baselines, §5):

| Grid | Fits | Worst case (60 epochs/fit), threads 4 | Worst case, threads 16 | Typical (~20 epochs), threads 4 |
|---|---|---|---|---|
| v0.2 | 11,274 | **≈ 300 CPU-h** | 194 CPU-h | ≈ 100 CPU-h |
| **v0.3** | **2,130** | **9.9 CPU-h** (3.2 h wall) | 19.1 CPU-h (2.9 h wall) | 3.3 CPU-h (1.1 h wall) |

**Budget definition (fixed):**
- **1 CPU-hour = 3,600 s of process CPU time** (user + system, summed over all threads) of the single training process, as returned by `time.process_time()`.
- Thread count fixed at **4** (`torch.set_num_threads(4)`).
- **Budget = 48 CPU-hours** for the whole programme of §4–5, including final refits.
- The worst case of v0.3 is 9.9 CPU-h, a margin of about 4.8×.

**Instrumentation:**
- `BudgetMeter.tick()` runs after every fit. It appends (fit number, configuration, cumulative CPU-s, wall-s) to `outputs/logs/h1_budget.tsv`.
- Fits run in a fixed, pre-declared order: primary grid (ℓ ascending, then r, then λ, then seed), then B6 heads, then B1, B5, the order heads, B2, the final refits.

**Overrun behaviour:**
- If cumulative CPU time exceeds 48 h, `BudgetExceeded` stops the programme immediately and writes `BUDGET_EXCEEDED` with the list of completed fits.
- **No selection or test evaluation is made on a partial grid.** The owner is consulted.
- The grid is **never reduced after observing DEV performance**. Any change needs a new deterministic proposal approved before resuming.

## 5. Baselines: operational specification (H1-R4 corrected)

All baselines:
- are fitted and selected **on DEV only**, using the §3 training-rule worlds;
- **no baseline choice of any kind uses LOCKED-TEST**: all baseline hyperparameters, seeds and predictors are frozen and hashed before unlock;
- **retraining rule:** after selection, every learned baseline is refit once on all DEV training-rule worlds with frozen hyperparameters for E\* epochs (its own mean best epoch) and 3 seeds, then averaged. Nothing is refit after unlock.

| ID | Input / targets | Grid actually applied | Selection | Seeds / averaging | Edge cases |
|---|---|---|---|---|---|
| **B1** | `mean_emb` (standardised), same bilinear head, target u2 | r ∈ {8, 32} × λ (4) × 3 seeds = 24 fits | mean DEV-val paired accuracy over seeds; same tie rule | final = 3-seed softmax average | none |
| **B2** | z at ℓ\*, same head, **paired permuted targets.** For family f and k ∈ {0, 1, 2}: a permutation π_f,k of the 6 slots seeded by `sha256("B2‖f‖k")`; the target of every world of f is the slot π_f,k(u2) | **no search:** (ℓ\*, r\*, λ\*), E\* epochs; 3 permutations × 3 seeds = 9 fits | none (fixed at the primary's choice) | per k, 3-seed average → 3 predictors | **threshold: inside every bootstrap replicate, the B2 value is max_k(paired accuracy of k) on that replicate**, so the comparison probe − max_k B2 is re-maximised per replicate (conservative). The point estimate uses max_k on the full sample |
| **B3** | per-candidate text features only: (i) appears in an action line; (ii) that action letter equals the model's chosen letter (model output, not oracle); (iii) number of rule lines where it is source; (iv) … where it is target; (v) normalised first line index; (vi) appears in a reward line; (vii) its printed reward (0 if none). Conditional-logit softmax over the 6 candidates | λ ∈ {1e-4, 1e-3, 1e-2, 1e-1}, L-BFGS (deterministic, no seed) | DEV-val paired accuracy; ties → larger λ | single deterministic fit | candidates with identical features → equal scores; ties in argmax broken by slot order. **No multi-hop or chained feature** |
| **B3-order** | B3 features, 3 heads (u1, u2, u3) | λ grid as B3, per head | DEV-val exact-order accuracy | deterministic | joint distinct assignment, as in §6.3 |
| **B6** | 4 anchor heads (c1, c3, u1, u3), same bilinear architecture, **each with its own search over the v0.3 grid** (17 ℓ × 2 r × 4 λ × 3 seeds = 408 fits per head) | as stated | each head: mean DEV-val single-world accuracy for its own target | per head, 3-seed average | **Duplicates:** the 4 anchors are assigned **jointly to 4 distinct candidates**, maximising the summed log-probabilities over the 360 ordered assignments; ties go to the lexicographically first. **So exactly 2 candidates always remain; empty sets are impossible.** |
| B4 / B5 / B7 | as v0.2 (B5: e(c) only; r × λ × seeds = 24 fits) | — | — | — | reported, not thresholds |

**B6 probabilistic convention:**
- Per world, p = 1/2 if the true u2 slot is among the 2 remaining candidates, else 0.
- **Paired success of a pair = p₁ × p₂**, as if the two worlds' choices between their 2 remaining labels were independent fair coins. Possible values are 0 and 0.25.
- No random draw is made: expectations enter the paired accuracy and the family bootstrap directly.

## 6. Decision rules

**6.1 Insufficiency (approved, unchanged).** Fewer than 180 admissible depth-4 `upath` pairs, or fewer than 100 contributing families → `INSUFFICIENT`.

**6.2 H1 success (approved, unchanged).** H1 is MET iff the lower bound of Δ_k > 0 for every k ∈ {B1, B2, B3, B6}.
- Δ_k = probe − B_k paired accuracy, with a 95 % family-cluster percentile bootstrap (10,000 resamples, seed 0).
- For B2, the max over permutations is taken inside every replicate.
- **H1 MET means L1 is established; L2 is not yet attributed.**

**6.3 L2 attribution** (only if H1 is MET; all three requirements, intersection-union).

**(a) Order test (corrected):**
- **Layer:** all decoding at the **single vector index ℓ\*** selected for the H1 probe. Multi-layer reconstructions may be reported descriptively but **cannot attribute L2**.
- **Heads:** u2 is the frozen H1 probe. u1 and u3 are **distinct heads**, trained on DEV with the §3 rule at (ℓ\*, r\*, λ\*), E\* epochs (their own early-stopping selection of E within the same 60-epoch rule), 3-seed average. They are **frozen and hashed before TEST**. Fits: 2 heads × 3 seeds, plus refits.
- **Branch:** the unchosen branch is the one **not chosen by the model's forced A/B choice** (slot convention §3). No oracle answer is a decoder input.
- **Decoding:** the 3 heads' log-probabilities are combined by **joint distinct assignment** of (u1, u2, u3) to 3 of the 6 candidates (maximum summed log-probability over 120 ordered triples; ties to the lexicographically first). Implementation: `h1_probe.l2_order_predict`, which **refuses heads trained at different layers** (tested).
- **Population:** admissible depth-4 base and cf worlds of LOCKED-TEST (single worlds; families resampled).
- **Criterion:** exact-match rate of the ordered triple minus max(1/6, B3-order), lower bound > 0.
  - 1/6 is the expected exact match of a decoder that knows the 3 unchosen labels and orders them at random.
  - B3-order is the prompt-only reference.

**(b) Alias generalisation:** single-world u2 accuracy on admissible depth-4 `alias_*` worlds, minus max(1/6, B3 on the same worlds), lower bound > 0. Same frozen probe at ℓ\*.

**(c) Shuffle generalisation:** the same on admissible depth-4 `shuffle_*` worlds.

**Outcome:** `MET + L2` if (a), (b) and (c) all hold; otherwise `MET, L2 not attributed`, naming the failing requirement(s).

## 7. What each outcome permits (unchanged)

| Outcome | Strongest statement |
|---|---|
| INSUFFICIENT | descriptive only |
| NOT MET (unsupported / weak) | H1 not supported / L1 only (not beyond anchor-and-eliminate) |
| MET, L2 not attributed | L1, including beyond anchor-and-eliminate |
| MET + L2 | ordered, label- and order-general unchosen-path information at the pre-answer position. **Not** evidence of forward construction (P0), sequential sweep (L4) or causal use (L3/H3) |

## 8. Secondary analyses and integrity (unchanged)

- **Secondary, Holm-corrected:** depth 3+4 pooled; `xswap`; per-depth reconstruction; decoy control; H2 descriptive layer profiles (not time).
- **Integrity:**
  - DEV-only selection;
  - sealed LOCKED-TEST via `sealed_read` with owner unlock and access log;
  - one evaluation;
  - all predictors and code hashed before unlock.

## 9. Evidence attached

| Item | Result |
|---|---|
| `tests_cpucheck/test_h1_probe.py` (CPU-check env, synthetic only) | **10/10**: planted-signal recovery (≥ 0.5 vs 1/6 chance) and chance without signal; determinism; fixed-epoch refit; joint assignment distinct and deterministic; **B6 duplicate resolution and expectation (0.5 per world, 0.25 paired)**; order exact match (u2/u3 swap → 0); B2 permutations paired per family and varying over k; bootstrap with max over permutations per replicate; budget overrun raises; **order heads must share ℓ\*** |
| Main suite | 80/80 |
| Benchmark | `outputs/metrics/bench_h1_grid.json` (§4b) |

## 11. Pre-freeze verifications (owner conditions for H1-R7) — both PASS

**11.1 No leakage through candidate order, and no oracle in predictive inputs** (`src/h1_features.py`, `src/h1_probe.py`, `tests_cpucheck/test_h1_probe.py`)

- **Canonical input order:** candidates are always presented to every decoder sorted **by label string** (random 4-character ids). Slots c1…u3 are used **only** to compute the supervision target index.
- **All ties are broken by canonical index (label order), never by slot:** `argmax_canonical`, and `joint_assignment` in lexicographic canonical order.
- **Tests:**

| Test | Result |
|---|---|
| Bilinear probe | scores are **equivariant** under arbitrary per-world candidate permutations; predicted **labels identical** after mapping back |
| Canonicalisation | any input order gives the same canonical tensor |
| Joint assignment (B6 anchors, order test) | over 200 random permutations: identical **labels** assigned; identical B6 expectation and order exact-match |
| Ties | resolved by canonical index |
| B3 features | permutation-equivariant |
| Padding (depth 3: 4 candidates) | masked candidates get probability 0 |

- **No oracle in predictive inputs:**
  - The candidate set is built from the **prompt alone** (rule-line labels minus rewarded terminals minus S). It equals the oracle's intermediate states in **4,200/4,200** non-decoy PROBE-DEV worlds (6 at depth 4, 4 at depth 3).
  - The builders `candidates_from_prompt`, `b3_features` and `canonical_candidates` take no world or oracle argument (checked) and run on the oracle-free model-input file.
  - B3 uses the model's forced choice (a model output), not the oracle answer.
  - `supervision_targets` (oracle, slots relative to the **model's** choice) is used only for labels.

**11.2 Epochs of the L2 heads, and B3 within the budget**

- **u1 and u3 heads:**
  - Each is trained at (ℓ\*, r\*, λ\*) with 3 seeds and early stopping on **DEV-val**.
  - **Each selects its own epoch count** E_u1 or E_u3 (rounded mean over its 3 seeds of the best epoch).
  - Each is then **refit on all DEV training-rule worlds with that fixed number of epochs**, 3 seeds, averaged, and **frozen and hashed before TEST**.
  - u2 is the H1 probe, with its own E\*.
  - u3 targets exist only at depth 4, so depth-3 worlds are excluded from u3 and c3 head training.
- **B3 and B3-order:**
  - Their L-BFGS conditional-logit fits (`train_conditional_logit`) call the same `BudgetMeter`.
  - They are counted in the **single global 48 CPU-hour budget**, with all bilinear fits, in the declared order: primary → B6 heads → B1 → B5 → B3 → B3-order → order heads → B2 → final refits.
  - Their cost is negligible (milliseconds to seconds per fit).

**11.3 Test status at freeze**
- `tests_cpucheck` (synthetic and prompt-only): **15/15**.
- Main suite: see §12.
- `tests_gemma`: 5/5.

## 12. Freeze manifest

`reports/BTC-H1_FREEZE_MANIFEST.json` holds the sha256 of 20 protocol code and test files, the 3 test logs (main 83/83, cpucheck 15/15, gemma 5/5) and the data hashes.

**LOCKED-TEST seal** (generated 2026-11-13, seed 2026111402, approval `outputs/approvals/H1-R7_LOCKED-TEST_GENERATION.json`):

| Item | Value |
|---|---|
| Seal id / UTC | `LOCKED-TEST-D5` / 2026-11-13T21:56:25Z |
| `worlds.jsonl` | `0a9938a79494dc5b44aba2e051761878295c5aa822d35ac190dcf84a8d14562f` |
| `model_inputs.jsonl` | `8fbc920d44295a5d54749c7bf9913fa27c882b78c049cd38880895dbbd43bd4f` |
| `manifest.json` | `746586a9b47a1c9245bea2845006004b8fc58051f8fff356029d74c94bb04c19` |
| `SEAL_MANIFEST.json` | `f1d541e297a261ab13573e6f36ea4082847d4fd968cef7a228559f86d0918392` |

- 4,800 worlds and 300 families (ids F30000–F30299). Files are read-only.
- **Access log absent: never read through `sealed_read`.**
- Content was not inspected. Only counts and hashes were printed at generation.
- **Independence from PROBE-DEV** (`tests/test_d5_independence.py`): 0 shared family ids, world ids, labels or prompt hashes. The in-memory regeneration is byte-identical to the sealed file.

## 10. Decisions requested (historical, v0.3 proposal)

| ID | Decision |
|---|---|
| H1-R2′ | Order test as specified in §6.3(a): same ℓ\*, distinct DEV-trained frozen heads, model-choice branch, joint assignment, multi-layer descriptive only |
| H1-R4′ | Baseline operational specifications (§5), including B2 max per replicate, B6 distinct-anchor assignment and product convention, and no baseline selection on LOCKED-TEST |
| H1-R5′ | New deterministic grid (17 vectors × r ∈ {8, 32} × 4 λ × 3 seeds), threads = 4, budget = 48 CPU-h process time, instrumentation and overrun rule (§4, §4b) |
| H1-R7 | Final freeze. On approval: generate and seal LOCKED-TEST (CPU) and record its hashes here, then submit the D3-DEV GPU request |

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (H1, D-1′, L1–L4 unchanged)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (no LOCKED-TEST; synthetic data only for probes)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (none executed)
[YES] Preserved negative results and raw logs (v0.2 grid infeasibility reported)
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none. The grid reduction is cost-based and submitted before any activation exists.
Next step: owner validation of H1-R2′, H1-R4′, H1-R5′ and H1-R7.
```
