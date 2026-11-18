# BTC-H1c-DEVX — results: INSTRUMENT ADEQUATE (pre-declared criteria); unchosen continuation not detected where keys are blind

**Date:** 2026-11-14 · **Owner decision:** "DÉCISION SCIENTIFIQUE — BTC-H1c-DEVX" (one capture, cap $0.99; CPU analysis if G1–G3 pass; cumulative cap raised to $6, **not** a TEST authorisation) · **Status:** completed, **STOPPED.**

> **Status of these results:** EXPLORATORY, on DEV families only. No TEST was created or captured and no confirmatory evaluation was run.
> **Kept as historical:** the initial H1c DEV result (inconclusive) and the diagnostic; nothing was rewritten.

## 1. Capture (single run)

| Item | Value |
|---|---|
| Frozen before the capture | git `7bbe0d4`. The CI reporting was added as a reporting-only change, recorded in `PREPARED.json`. Pre-launch checks passed; BTC spend before the run $3.762 |
| App | `ap-2tmsd0cdAeYGk8OJM73jPl`, 20:24 → 20:35 UTC |
| Timing | load 96 s, capture 466 s (3,000 worlds + 20 replays), remote 562 s (guard 900 s); download 50 s |
| Size | ≈ 1.3 GB; peak GPU memory 63.5 GB; watchdog not triggered |
| **G1** | hashes and shapes OK, 3,000/3,000, verified before Volume deletion |
| Fidelity | max\|Δ\| 0.195 ≤ 0.25; 0 non-finite; **20/20 replays bitwise identical**; 22 divergence entries = 11 worlds with an exact original A/B tie (not admissible) |
| **G2** bitwise identity of the binary arrival keys | **600 / 600 late pairs identical**, so PASS |
| **G3** competence | overall **0.727**, late pairs **0.749**, so PASS. By variant: base_L 0.748, dswap_L 0.750, cswap_L 0.763, base_E 0.678, dswap_E 0.693 |
| **Cost** | **$0.474** (A100 0.417, CPU 0.032, memory 0.021, network 0.004), within the $0.99 cap. **BTC cumulative ≈ $4.24** (cap $6) |

## 2. Analysis (single run; 0.043 CPU-h of 2; 4 threads)

**Data:** H1c DEV and DEVX combined, layers {32, 40, 48}, admissible worlds only. Frozen family roles:

| Role | Admissible worlds | Contributing families |
|---|---|---|
| TRAIN | 2,551 | — |
| SELECT | 364 | 88 |
| EVAL | 711 | 174 (never seen) |

**Layer selection (PC1 on SELECT only; u2 never used):**

| Layer | PC1 on SELECT |
|---|---|
| 32 | 0.585 |
| **40** | **0.909** (selected) |
| 48 | 0.907 |

### 2a. Pre-declared stop criteria, on EVAL

| Criterion | Value | Threshold | Result |
|---|---|---|---|
| **SC1** PC1 (12-way c1) | **0.886**, 95 % CI [0.855, 0.915], 174 families / 711 worlds | ≥ 0.80 | **pass** |
| **SC2** \|PC1(SELECT) − PC1(EVAL)\| | \|0.909 − 0.886\| = **0.023** | ≤ 0.10 | **pass** |
| **SC3** chosen continuation (PC2) | **0.756** [0.720, 0.795], sign-flip p < 1e-4 (174 families) | > 0.5, p < 0.025 | **pass** |
| **Decision** | **INSTRUMENT ADEQUATE** | | |

PC1 on TRAIN is 1.000: the gap to EVAL shows some residual overfitting, but generalisation is adequate.

### 2b. Unchosen continuation, on EVAL (reported; not part of SC1–SC3)

| Population | Accuracy | Uncertainty | Families |
|---|---|---|---|
| **Late pairs: robust rule** (sign-flip on family means; arrival keys identical within pairs, so information can only come from the query) | **0.521** | sign-flip p = **0.18**; late-world CI [0.480, 0.557] | 141 contributing families (158 families with ≥ 1 late world) |
| Late pairs, paired (descriptive) | 0.128 | — | 141 |
| **Early pairs** (departure lines *before* the arrival lines, so the arrival keys can contain the continuation) | **0.703** | sign-flip p < 1e-4 | 118 |
| All EVAL worlds, all variants | 0.605 | [0.575, 0.634] | 174 |

## 3. Reading (exploratory)

1. **The instrument is now sensitive** (SC1–SC3): with about 2,550 training worlds, the pointer reads c1 well (0.89) and decodes the **chosen** continuation (0.76) on never-seen families.
2. **On late pairs, the unchosen continuation is not detected** (0.52, p = 0.18; CI includes 0.5).
   - This is the decisive design, in which only the pre-answer query can carry the information.
   - The DEV point estimate corresponds to δ ≈ +0.02, with an upper CI bound of about +0.06.
3. **On early pairs, the unchosen continuation is decodable** (0.70).
   - Here the candidates' own arrival keys are computed **after** the departure lines, so they can encode the continuation at reading time.
   - Choice + keys and key-only controls on early pairs were **not** part of the pre-declared analysis and were not run. Running them now would be post hoc.
   - The most parsimonious reading, offered as a hypothesis and not established: **the continuation information is present in the label representations when causally available, but there is no evidence that the pre-answer query adds it when the keys cannot.**
4. **Chosen vs unchosen asymmetry:** the chosen continuation is decoded at 0.76 (all worlds); the unchosen one at 0.52 on late pairs.
   - The populations differ: PC2 uses every variant, P uses late pairs only.
   - A like-for-like comparison would require the chosen-side analogue on `cswap_L` with the same robust rule. It is not pre-declared and was not run.

## 4. Confirmatory proposal (for a separate scientific decision; nothing frozen or run)

**Rule (owner): with SC1–SC3 passed, I declare only INSTRUMENT ADEQUATE and propose:**

| Element | Proposal |
|---|---|
| Hypothesis | as C-2′ / A1: on late dead-end-swap pairs, the pre-answer query decodes the unchosen continuation above 0.5 (robust sign-flip rule on family means) |
| Frozen configuration | layer 40 (selected on SELECT via PC1), arrival keys, pointer r = 8, λ = 0.1, 3 seeds, trained on all DEV (TRAIN + SELECT + EVAL, 1,000 families) with E fixed from SELECT |
| New sealed TEST pool | about 600 new depth-4 families × 2 late pairs (2,400 worlds); generator unchanged; seed and sealing rules as for LOCKED-TEST |
| Capture | 13 slots × layer 40 only; ≈ $0.60, cap ≈ $0.85; cumulative ≤ ≈ **$5.09** (cap $6) |
| Controls (unchanged) | key-only, choice × keys, μ_choice query (0.5 by construction); P_perm; PC1 and PC2 as sensitivity checks on TEST |
| **Power, given what DEV shows** | DEV late estimate δ ≈ +0.02 (CI up to about +0.06). Robust-rule power at 600 families: ≈ 0.33–0.39 for δ = 0.025, 0.83–0.93 for δ = 0.05. **If the true effect is near the DEV estimate, a TEST would most likely be NOT MET.** About 1,500–2,000 families would be needed for reasonable power at δ = 0.025 (≈ $1.5–2, beyond the $6 cap) |

**My assessment, for the decision:**
- A confirmatory TEST at 600 families is feasible within budget.
- But given the DEV estimate, its most likely outcome is NOT MET, with power around 0.35 for the effect DEV suggests.
- **Alternatives to weigh:**
  - **(a)** run it anyway, to obtain a pre-registered result;
  - **(b)** close H1c with the exploratory conclusions above;
  - **(c)** first run, on DEV (CPU, post hoc, labelled as such), the early-pair and `cswap_L` controls that would clarify the keys-vs-query and chosen-vs-unchosen readings.

## 5. Artifacts

| Artifact | sha (prefix) |
|---|---|
| `outputs/h1c_devx/capture/MANIFEST.json` | `82dacfd1…` |
| `GATES.json` | `ec024948…` |
| `run_manifest.json` | `62564e90…` |
| `outputs/h1c_devx/analysis/DEVX_ANALYSIS.json` | `8adcf438…` |

The tensors are not versioned. Logs are in `outputs/logs/h1c_devx_*`.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (frozen TRAIN/SELECT/EVAL; layer chosen from PC1 on SELECT; no TEST)
[YES] Did not equate token availability with prospective path representation (late-pair design; early-pair reading flagged as key-borne)
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (BTC-H1c-DEVX; $0.474 ≤ $0.99; cumulative ≈ $4.24 ≤ $6)
[YES] Preserved negative results and raw logs (late-pair null reported; historical DEV kept)
[YES] Labeled unrun stages NOT RUN and limitations honestly (early-pair controls and cswap analogue not run)
Any deviation: none.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — separate scientific decision on the confirmatory proposal (a / b / c).
```
