# BTC-H1c — DEV capture: CPU preparation and GPU request

**Date:** 2026-11-14 · **Owner decision:** "DÉCISION SCIENTIFIQUE — BTC-H1c" (C-2′, C-6, C-7, C-8 validated; DEV preparation on CPU only) · **Status:** PREPARED, **NOT RUN.**

> **Not done:** no GPU, no TEST generated, no real probe training.
> **Kept as is:** BTC-H1b v1.0 untouched; cumulative limit $5.

## 1. Owner conditions

| # | Condition | Result |
|---|---|---|
| 1 | Null calibration of the primary rule with ≥ 5,000 simulations; Monte-Carlo uncertainty | §2. **Calibrated** in every null scenario; no protocol correction needed |
| 2 | Blocking control of the bitwise identity of the two binary keys within each late pair | `src/h1c_gates.py` (G2, blocking). Tested: passes on identical keys and blocks a perturbed copy. **On a real forward pass (tiny random Gemma, real tokenizer, real H1c DEV prompts), the binary keys are bitwise identical within late pairs while the ANS query differs** |
| 3 | DEV generator, positions, capture, synthetic tests, integrity checks | §3 |
| 4 | Probe selection on distinct DEV families; positive controls for sensitivity | §4: inner-train / inner-val within DEV-train families, report on DEV-val families; PC1 (12-way c1 ≥ 0.80), PC2 (chosen continuation), PC3 (choice letter) |
| 5 | $5 cumulative maximum | §5: worst case after DEV ≈ $4.14; DEV + TEST worst ≈ $4.96 |
| 6 | DEV GPU request prepared, not executed | §6 |

## 2. Null calibration (`src/power_h1c_null.py` → `outputs/metrics/power_h1c_null.json`)

**Setup:**
- The exact rule: Δ = P − max(0.25, C_max, P_perm); family bootstrap with B = 2,000 per dataset (the protocol uses 10,000).
- 600 families × 2 late pairs; admissibility 0.75; **5,000 datasets per scenario**.

**Correction to the null model, made during this step:**
- The correct label **flips** between the two members of a pair.
- So without information, it is the decoder's *preference* that may be correlated, not its correctness.
- My first draft wrongly correlated correctness. I stopped it before it produced results and rewrote it.

| Null scenario | Mean P pair success | **Rejection rate** | MC SE | MC 95 % interval |
|---|---|---|---|---|
| N1 boundary: independent preferences | 0.250 | **0.0200** | 0.0020 | 0.016–0.024 |
| N2 family-correlated preferences (ICC 0.40) | 0.185 | 0.0000 | — | — |
| N3 deterministic (same preference in both members) | 0.000 | 0.0000 | — | — |
| N4 boundary, with P_perm at 0.15 (the 0.25 floor binds) | 0.250 | **0.0228** | 0.0021 | 0.019–0.027 |

**Reading:**
- At the least favourable boundary (N1), the one-sided level is ≤ the nominal 0.025.
- With the 0.25 floor binding (N4), the point estimate is 0.023; the MC interval just covers 0.025.
- **The rule is calibrated** (slightly conservative). Realistic nulls (N2, N3) are far below the nominal level.
- Power under alternatives (`power_h1c_binary.json`, unchanged): 600 families → ≥ 0.985 at δ = 0.10, and 0.58–0.88 at δ = 0.05.

## 3. Prepared and tested on CPU

| Item | File | Result |
|---|---|---|
| **H1c DEV pool** (generated on CPU; not a TEST, not sealed) | `src/build_h1c_dev.py` → `data/h1c/dev/` | 400 families × 5 variants = **2,000 worlds**, seed 2026111404, F50000–F50399; families split 300 train / 100 val, stratified by optimal action; labels disjoint from v1, v2, PROBE-DEV, LOCKED-TEST and the H1b design (in-memory regeneration). Model inputs sha `ca2133ca…` |
| **Positions** (30 slots: ANS, 12 arrival keys, 14 rule-line ends, 2 reward-line ends, Q_END) | `src/h1c_positions.py`, `src/h1c_build_positions.py` → `outputs/h1c_dev/plan/` | every slot decodes to the expected token in **2,000/2,000** worlds; binary arrival keys before the first differing token with an identical prefix in **400/400** late pairs; prompts 291–333 tokens; positions sha `0d29b278…` |
| **Capture script** | `src/h1c_dev_capture.py` (sha `9c188f09…`) | same capture core as before (frozen D3 fidelity check, non-finite rejection, 20 replays, tokenisation sha), 30 slots × layers {24, 32, 40, 48, 56}; transfer verified before Volume deletion |
| **Gates** | `src/h1c_gates.py` (sha `cb86889b…`) | G1 hashes and shapes; **G2 blocking bitwise identity** of the binary keys in every late pair; **G3 competence** (≥ 0.70 overall, ≥ 0.60 on late pairs). The analysis is refused unless G2 and G3 pass |
| **DEV analysis** (prepared) | `src/h1c_dev_analysis.py` (sha `38c9a775…`) | binary selection over layer {24…56} × rank {2, 8} on inner-val; controls C_bil, C_lin, key-only, P_perm on DEV-val late pairs; positive controls PC1–PC3; early pairs and position secondaries |
| **Synthetic test of the analysis** (planted unchosen-continuation signal) | `tests_cpucheck/test_h1c_dev.py` | P paired 0.59 on late pairs; **key-only, C_lin and C_bil exactly 0.000** (identical keys, so identical predictions); P_perm 0.27; selection and reporting on disjoint families |

**Test suites:**

| Suite | Result |
|---|---|
| cpucheck | **60/60** |
| main | **98/99** (the historical D3-A4 test only, as documented) |
| gemma | **12/12**, including `test_h1c_capture.py` (bitwise identity on a real forward) and `test_h1c_tokens.py` |
| post-evaluation | 4/4 |

Logs are in `outputs/logs/h1c_dev_prep_tests_*.log`; hashes in `outputs/h1c_dev/plan/PREPARED.json`.

## 4. DEV selection and positive controls (to run after the capture, only if the gates pass)

**Selection:**
- Only on DEV-train families (inner 80/20 split), with early stopping and the choice of layer and rank on inner-val.
- **DEV-val families** (100) are reported only.
- The H1c TEST protocol (layer, rank, decoders) would be frozen from this.

**Positive controls on DEV-val, reported with the selection:**

| ID | Control | Requirement |
|---|---|---|
| PC1 | 12-way c1 pointer at the selected layer | **≥ 0.80**, otherwise the instrument is insufficiently sensitive and no TEST is proposed |
| PC2 | binary decoding of the **chosen** continuation | reported (symmetry reference) |
| PC3 | the model's chosen letter, linear on ANS | reported |

**Primary-type report on DEV-val late pairs:**
- P paired accuracy vs C_bil, C_lin, key-only (0 by construction), P_perm and 0.25.
- Also early pairs and the query-position sweep.
- **All exploratory**; the decision would come only from a later TEST.

## 5. Cost (the $5 BTC cumulative limit is unchanged)

| Item | Estimate |
|---|---|
| Worlds / slots / layers | 2,000 worlds × 30 slots × 5 layers ≈ **3.2 GB** (bf16) |
| Remote time | load 245–290 s + 2,020 forwards × ≈ 0.17 s ≈ 345 s + writes ≈ 40 s → **≈ 650–700 s** (guard 800 s, timeout 850 s) |
| Expected cost | ≈ **$0.68** |
| Hard cap | ≈ **$0.90** (timeout plus overhead at the worst-case rate, plus egress) |
| BTC cumulative | $3.24 → ≈ $3.92 expected (≤ $4.14 worst) |
| Planned TEST later | ≈ $0.70 (cap $0.82) → ≤ **$4.96** worst in total |
| Margin | no second full attempt for either capture under $5 |

## 6. GPU request

```text
AUTORISATION DEMANDÉE — BTC-H1c-DEV
Objectif scientifique : capturer les activations de Gemma-4-31B-it sur le pool H1c-DEV (impasses, paires dead-end-swap) pour mesurer la compétence, vérifier l'identité bit à bit des clés binaires et sélectionner sur DEV le décodeur de la règle C-2′ (exploratoire ; aucune décision confirmatoire).
Why needed / alternative : la nouvelle tâche n'a jamais été vue par le modèle ; compétence, intégrité (G2) et sélection exigent des activations ; aucune alternative CPU.
Exact change/command : `modal run src/h1c_dev_capture.py` (sha 9c188f09…) ; 2 000 mondes DEV (inputs ca2133ca…, positions 0d29b278…), 30 positions × 5 couches {24,32,40,48,56}, même image et révision 842da379… ; contrôle de fidélité D3 gelé, 20 rejeux, empreinte de tokenisation, vérification des shards avant suppression du Volume ; puis portes G2 (identité bit à bit, bloquante) et G3 (compétence) en local.
Compute and cost : 1 × A100-80GB, cpu (4,4), mémoire 16–96 GiB, max_containers=1, retries=0, timeout 850 s, garde 800 s ; ≈ 11–12 min distantes + ≈ 2–3 min de téléchargement ; ≈ 3,2 Go ; coût attendu ≈ 0,68 $, plafond ≈ 0,90 $ ; cumul BTC ≤ ≈ 4,14 $ (limite 5 $).
Risk and stop condition : arrêt sans relance sur tout échec (fidélité, rejeu, tokenisation, transfert, intégrité) ; échec de démarrage → arrêt immédiat (watchdog) ; si G2 échoue → analyse refusée ; si G3 échoue → H1c s'arrête (résultat comportemental négatif), aucun TEST. Aucun pool TEST, aucun accès à LOCKED-TEST.
I will not execute this step until you explicitly approve it.
```

**The DEV CPU analysis** (§4) would need its own go-ahead, or could be included in this authorisation, with a cap of 4 threads and 3 CPU-h.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task (controlled extension)
[YES] Kept H1 primary and H2/H3 conditional
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently (no real probe training)
[YES] Preserved oracle isolation and train/val/test independence (DEV only; family-disjoint selection; no TEST)
[YES] Did not equate token availability with prospective path representation (bitwise-identical keys; controls 0 by construction)
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (GPU requested, not run)
[YES] Preserved negative results and raw logs (null-model draft error documented and corrected)
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decision on BTC-H1c-DEV.
```
