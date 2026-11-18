# BTC-H1b X3 + X4 — results: contextual label keys, observation positions and instrument validity (EXPLORATORY)

**Date:** 2026-11-14 · **Authorisation:** owner, "DÉCISION SCIENTIFIQUE — BTC-H1b-X3X4" (GO conditionnel: one capture + one CPU analysis) · **Status:** completed, **STOPPED.**

> **EXPLORATORY.**
> - PROBE-DEV was already used for the H1 v0.3 selection; LOCKED-TEST was not accessed; no new pool was created.
> - **The H1 v0.3 verdict NOT MET is unchanged.** Nothing below is confirmatory.

## 1. What ran

| Step | Result |
|---|---|
| Owner amendments, applied and tested **before** the capture (git `b6adc69`) | choice + contextual-keys control; pre-reward positions treated as a measured control with no causal reading; 3 seeds everywhere; missing positions declared; c1/A1 reported by model choice A/B |
| Pre-launch (`outputs/d3x/plan/prelaunch.json`) | frozen files unchanged; D3 core and `v2_common` frozen; model `842da379…`; DEV inputs and positions hashes; no earlier capture |
| **Capture**, app `ap-8XwptyExv041YgLOhdv9c0` | 15:34:00 → 15:52:05 UTC (18 min local wall); model load 285.6 s, capture 588.1 s, remote total 876.8 s (guard 1,420 s); download 155 s; **7.41 GB**; peak GPU memory 63.1 GB; watchdog not triggered |
| Integrity | **3,480 / 3,480** worlds; shard hashes and shapes OK before Volume deletion; container tokenisation sha equal to the local positions for every world |
| Fidelity (frozen D3 rule) | max\|Δ\| 0.195 (post-norm) / 0.205 (pre-norm) ≤ 0.25; 0 non-finite; **20 replays bitwise identical**; 1 world with a tolerated top-1 near-tie (F20168-base) |
| **ANS identity with D3-DEV-R1** | **3,480 / 3,480 bitwise identical** (max\|Δ\| = 0), so the analysis precondition was met |
| **Cost** (`modal billing report`) | **$0.751** (A100 0.645, CPU 0.049, memory 0.033, network 0.024). BTC cumulative ≈ **$3.24** (under $5) |
| **Analysis** (single run) | 15:52:36 → 16:30:11 UTC; **1.43 CPU-h of 6**; 540 grid cells (3 seeds), 72 key-only, 72 choice + keys, B3+; 22 cells hit the 60-epoch cap (all A1 or pre-reward cells, listed in `results.jsonl`) |

**Note on spend:** the Modal workspace shows $19.68 this month. $17.19 of it comes from an unrelated app (`prospection-evaluative`, 2026-11-12). Correcting my earlier ≈ $2.43, the BTC apps total $2.49 before this run and **$3.24** after it.

**Splits:** identical to H1b. Every choice is made on inner-val (181 / 45 families); results are on DEV-val (74 families, 855 worlds, 77 u2 pairs).

**Chance levels:** about 0.196 per world (0.167 for c3/u3) and 0.028 paired.

## 2. Instrument validity (pre-declared rule, thresholds unchanged)

**Selected cell** (best inner-val c1): **layer 48, query Q_END, contextual arrival key (`ctx_tgt`).**

| Criterion | Value (DEV-val) | Threshold | Result |
|---|---|---|---|
| S-A c1 | **0.958** | ≥ 0.80 | pass |
| S-B c1 − key-only(c1) | 0.958 − 0.538 = **0.42** | ≥ 0.25 | pass |
| S-C A1 | **0.943** | ≥ 0.80 | pass |
| **Verdict** | **SENSITIVE** | | |

**By model choice:**
- c1: A 0.958 (n = 495) / B 0.958 (n = 360).
- A1: A 0.952 / B 0.931.

There is no asymmetry between the choices.

**Accompanying control (owner amendment 1) — choice + contextual keys, no query activation:**
- **c1 = 1.000 and A1 = 1.000** at layer 48 (≥ 0.99 from layer 24 onwards).
- So c1 can be rebuilt **entirely** from the model's chosen letter plus the candidates' contextual keys.
- The pointer's success on c1 therefore **does not show** that the query activation carries label identity beyond the choice.
- The instrument is valid in the sense of the pre-declared rule, but its positive control is fully explained by "choice + key context".

## 3. Representation effect (static vs contextual keys) and position effect

DEV-val accuracy. For each (R, Q), the layer is chosen on inner-val and noted after the @.

| Slot | Representation | ANS | Q_END | ACT_END_A | ACT_END_B | LAST_RULE_END |
|---|---|---|---|---|---|---|
| **c1** | static | 0.172 | 0.213 | 0.317 | 0.227 | 0.213 |
| | **ctx arrival** | **0.966** @48 | **0.958** @48 | 0.520 | 0.501 | 0.439 |
| | ctx departure | 0.736 | 0.732 | 0.463 | 0.407 | 0.386 |
| **A1** | static | 0.213 | 0.209 | 0.712 | 0.239 | 0.225 |
| | **ctx arrival** | **0.994** | 0.979 | 0.991 | 0.998 | 0.792 |
| | ctx departure | 0.792 | 0.766 | 0.887 | 0.715 | 0.577 |
| **u2** (paired) | static | 0.281 (0.026) | 0.251 (0.013) | 0.233 (0.000) | 0.233 (0.013) | 0.208 (0.000) |
| | **ctx arrival** | **0.815 (0.688)** @40 | **0.821 (0.610)** @40 | 0.391 (0.208) | 0.447 (0.234) | 0.426 (0.299) |
| | ctx departure | 0.607 (0.377) | 0.628 (0.481) | 0.342 (0.143) | 0.378 (0.273) | 0.332 (0.130) |
| **c2** | static | 0.250 | 0.278 | 0.228 | 0.225 | 0.206 |
| | **ctx arrival** | 0.788 | **0.827** | 0.458 | 0.395 | 0.382 |

**Representation effect (very large):**
- **Static keys stay at chance at every position and layer.** This reproduces the H1 / X1b failure.
- **Contextual arrival keys** recover c1, A1, c2 and u2 at the pre-answer positions.
- Arrival keys beat departure keys.
- **Reading:** E3 is strongly supported as the cause of the H1 instrument failure. The static mean-token key cannot address labels; the label's own contextual state can.

**Position effect, as a measured control (owner amendment 2):**
- The **choice-dependent** slots (c1, c2, u2) are decoded far better from the post-reward queries (ANS, Q_END) than from the pre-reward ones (ACT_END_A/B, last rule line): u2 is 0.82 vs 0.39–0.45, c1 is 0.96 vs 0.44–0.52.
- The **choice-independent** A1 is decoded equally well at ACT_END (0.99).
- The pattern matches the query mainly contributing **choice** information. Per the amendment, no causal interpretation is attached.

**Layer profile** (ctx arrival, Q_END):
- c1: 0.54 at layer 32 → 0.94 at 40 → 0.96 at 48.
- u2: 0.44 at 32 → **0.82 at 40** → 0.66 at 48 → 0.47 at 61.
- A1 is already at 0.98 by layer 32.
- These are descriptive stages of one forward pass, not time.

## 4. u2: the H1 target with the validated instrument, and its controls

| Decoder (ctx arrival keys) | Layer | u2 DEV-val single | **u2 paired** (77 pairs) |
|---|---|---|---|
| **Pointer, pre-declared selected cell** | 48, Q_END | 0.662 (A 0.685 / B 0.631) | **0.429** |
| Pointer, best u2 cell on inner-val (POST HOC look) | 40, Q_END | 0.821 | 0.610 |
| Pointer at ANS (as H1's position) | 40 | 0.815 | 0.688 |
| **Choice + keys** (no query) | 48 | 0.641 | **0.519** |
| Choice + keys | 40 / 32 | 0.72 / 0.70 | 0.57 / 0.58 |
| Key-only (no query, no choice) | 48 / 40 | 0.435 / 0.47 | 0.208 / 0.36 |
| B3+ (prompt text + model choice) | — | 0.327 | 0.156 |
| H1 static probe (X1b, layer 32) | — | 0.208 | 0.013 |

**Readings:**
1. **The H1 negative result was an instrument failure.**
   - With contextual keys, the unchosen depth-2 label is decodable far above chance, the prompt baseline and key-only (paired 0.43–0.69 vs B3+ 0.156).
   - The H1 v0.3 verdict stands as a statement about the frozen probe. It is not evidence that the information is absent.
2. **Most of the u2 signal is reproduced without any query activation.**
   - Choice + keys reaches paired 0.52–0.58.
   - The contextual arrival states of the labels (rule lines, which come **before** the rewards) already carry branch and depth membership: which initial action a label is reachable from, and at what depth. This is relational, multi-hop information computed by the model while it reads, which the prompt-only B3+ (no chaining) cannot reach.
   - Adding the model's chosen letter then identifies the unchosen depth-2 state.
3. **What the decision-time query adds beyond the choice is small and uncertain.**
   - At the pre-declared selected cell, the pointer does **not** beat choice + keys on u2 (paired 0.429 vs 0.519).
   - At layer 40 it does: single 0.82 vs 0.72, paired 0.61–0.69 vs 0.57. That comparison is a post hoc look over many cells, with 77 pairs (paired SE ≈ 0.055).
   - **A specific representation of the unchosen path at the decision point, beyond the choice bit, is NOT established.**

## 5. E1–E7 after X3 + X4 (DEV, exploratory; none established)

| ID | Status now |
|---|---|
| **E3** (static keys unsuitable) | **Strongly supported** as the cause of the H1 instrument failure: static keys stay at chance; contextual keys recover every slot |
| **E1** (unchosen state not computed) | **Weakened in its broad form.** The structure of both branches (labels' depth and branch membership) is computed in the label representations during reading. **Open in its narrow form:** whether the decision-time state represents the unchosen path beyond the choice |
| **E4** (position) | Refined: the label-identity *structure* lives at the label positions; the query positions mainly add the choice (measured, no causal reading) |
| E2 (capacity) | Not the main factor: the same bilinear capacity goes from chance to 0.96 once the keys change |
| E5 (layer sampling) | A layer-40 peak for u2 shows that layer matters; it lies inside the old v0.3 grid, but at the static-key bottleneck it could not be seen |
| E6 (chosen-branch only) | **Not supported:** c2 (0.83) and u2 (0.82) are decoded comparably |
| E7 (competence) | Not re-tested here (no confidence strata in this block) |

## 6. Limitations

- **Re-use of PROBE-DEV:** it was used for the H1 selection and again for H1b. This is many cells (540), with inner-val selection and a single DEV-val split (77 u2 pairs).
- **Capacity mismatch in the key comparison:** the pointer (bilinear, rank 8) and choice + keys (linear, 2 × 5,376 weights, its own λ) differ in capacity. The pointer − choice + keys comparison is therefore indicative only.
- **Choice + keys uses the model's chosen letter.** It shows what the choice plus reading-time structure suffice for; it is not a model of the mechanism.
- **Epoch cap:** 22 cells hit the 60-epoch cap (A1 and pre-reward cells), so these may be slight underestimates.
- **Scope:** only H1-type decoding. No causal test (H3) and no dynamics claim (H2).

## 7. Recommendation (one next discriminating experiment; needs a new decision)

**A pre-registered confirmatory test of the narrow question:**
- **The question:** does the decision-time activation carry unchosen-path information beyond the choice and the reading-time label structure?
- **Primary contrast, frozen on PROBE-DEV:** u2 paired accuracy of the contextual-key pointer at the pre-answer position minus a **capacity-matched** choice + keys decoder (same bilinear form, with the query replaced by a one-hot of the model's choice). Layer and representation fixed on DEV (e.g. layer 40, arrival keys).
- **Secondary contrasts:** vs B3+ and vs key-only.
- **Data:** a **new, independently generated and sealed TEST pool** (LOCKED-TEST is consumed), with its multi-position capture, under a new frozen protocol with power computed for paired differences of about 0.05–0.10.

This is the cleanest next step: it isolates the prospective, decision-time component from the reading-time graph encoding that this block uncovered.

## 8. Artifacts

| Artifact | sha (prefix) |
|---|---|
| `outputs/d3x/probe_dev/MANIFEST.json` | `7b29e05e…` |
| `outputs/d3x/probe_dev/x_run_manifest.json` | `e75c14c6…` |
| `outputs/h1b_x3x4/PLAN.json` | `ee07a594…` |
| `outputs/h1b_x3x4/SUMMARY.json` | `743a0e24…` |
| `outputs/h1b_x3x4/results.jsonl` (every cell and control) | `6d5ea559…` |
| analysis script | `f92586d0…` |

- The 7.4 GB of tensors are not versioned; they are referenced by these hashes.
- Logs: `outputs/logs/x3x4_*`.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (no H2/H3 work)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (PROBE-DEV only; LOCKED-TEST not accessed)
[YES] Did not equate token availability with prospective path representation (choice+keys and key-only reported; narrow claim NOT established)
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (BTC-H1b-X3X4; $0.751; BTC cumulative ≈ $3.24 < $5)
[YES] Preserved negative results and raw logs (all 540 cells + controls)
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none. Spend accounting corrected ($2.49 before this run, not $2.43).
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decision on a new frozen protocol with a new sealed TEST pool.
```
