# BTC-H1c — single DEV extension: CPU preparation, stop criteria, revised budget, GPU request

**Date:** 2026-11-14 · **Owner decision:** "DÉCISION SCIENTIFIQUE — BTC-H1c" (single DEV extension; CPU preparation only) · **Status:** PREPARED, **NOT RUN.**

> **Not done:** no GPU, no TEST.
> **Limit in force:** the $5 cumulative limit applies until the owner explicitly approves a raise to $6.
> **Historical results:** H1c DEV, the diagnostic and everything earlier stay as historical, exploratory results.

## 1. What is prepared

| Item | File | Result |
|---|---|---|
| **DEV extension pool** (CPU, not a TEST) | `src/build_h1c_devx.py` → `data/h1c/devx/` | **600 new families × 5 variants = 3,000 worlds**, seed 2026111406, F51000–F51599. Labels, families and prompt hashes are disjoint from v1, v2, PROBE-DEV, LOCKED-TEST, the H1b design and the H1c DEV pool. Model inputs sha `ab4ee61e…` |
| **Frozen three-way family roles** (requirement 3) | `outputs/h1c_devx/plan/SPLITS.json` (sha `68ca7d14…`) | **TRAIN 700** = all 400 already-analysed H1c-DEV families + 300 new. **SELECT 100** new families. **EVAL 200** new families, never seen. Stratified by optimal action. **Analysed families are never used for selection or evaluation** |
| **Positions** | `src/h1c_devx_positions.py` → `outputs/h1c_devx/plan/` | the 30-slot layout verified on **3,000/3,000** worlds; binary keys before the first differing token with identical prefix in **600/600** late pairs; positions sha `c4c58f0b…` |
| **Capture** | `src/h1c_devx_capture.py` (sha `72f2a5f8…`) | **13 slots** (ANS + 12 arrival keys, same indices as the DEV capture) × **3 layers {32, 40, 48}**; frozen fidelity check, 20 replays, tokenisation sha; gates G1 (integrity), **G2 (blocking bitwise identity)**, G3 (competence) |
| **Analysis** | `src/h1c_devx_analysis.py` (sha `3b87545a…`) | combines the DEV and extension captures (layers 32 / 40 / 48, slots 0–12); procedure and criteria fixed (§2) |

**Tests:**

| Suite | Result |
|---|---|
| cpucheck | **62/62**, including `test_h1c_devx.py`: frozen roles, disjoint families, decision follows the criteria, single run |
| main | **99/100** (historical D3-A4 only), including the static checks of the extension capture |
| gemma | **14/14**, including `test_h1c_devx_capture.py`: 13 slots and bitwise identity of the binary keys on a real tiny-model forward |
| post-evaluation | 4/4 |

Hashes are in `outputs/h1c_devx/plan/PREPARED.json`.

## 2. Pre-declared procedure and stop criteria (requirements 2 and 4; fixed before any capture)

**Procedure:**

1. **Layer selection based on PC1 only** (requirement 2).
   - LAYER = the argmax over {32, 40, 48} of PC1 accuracy on **SELECT**. PC1 is the 12-way c1 pointer (ANS query × 12 arrival keys), fitted on TRAIN, with early stopping on SELECT; ties go to the lower layer.
   - **u2 is never used for selection.**
2. At LAYER, on **EVAL**:
   - PC1;
   - **PC2**, the binary *chosen* continuation, with a sign-flip test on family means;
   - **P**, the binary *unchosen* continuation, with the robust sign-flip rule (A1) on late pairs;
   - the controls (0.5 by construction on late pairs), P_perm, and early pairs.

**Decoders:** pointer, r = 8, λ = 0.1, seeds 0, 1, 2, early stopping on SELECT. **No hyper-parameter search.**

**Instrument adequacy, evaluated on EVAL only:**

| Criterion | Threshold | Rationale |
|---|---|---|
| **SC1** sensitivity | PC1 ≥ 0.80 | the original requirement |
| **SC2** generalisation | \|PC1(SELECT) − PC1(EVAL)\| ≤ 0.10 | guards against a lucky selection |
| **SC3** continuation decoding | PC2 > 0.5 with sign-flip p < 0.025 | the instrument must decode at least the *chosen* continuation (it reached 0.666 post hoc at layer 40 on the old DEV) |

**Decision rule:**
- **Any failure → INSTRUMENT INSUFFICIENT: close H1c, no new TEST** (owner rule).
- **All pass → INSTRUMENT ADEQUATE:** re-examine a frozen confirmatory protocol.
- The unchosen-continuation result is reported (exploratory). It does not enter SC1–SC3.

## 3. Budget (requirement 5)

| Step | Expected | Cap |
|---|---|---|
| **DEV extension capture** (3,000 worlds; load ≈ 250–290 s + ≈ 500 s of forwards; timeout 950 s, guard 900 s; ≈ 1.3 GB) | ≈ **$0.76** | ≈ **$0.99** |
| BTC cumulative after it | $3.76 → ≈ $4.52 | ≤ **$4.75**, within the **current $5 limit** |
| Eventual independent TEST capture (only if adequate and after a new frozen protocol; ≈ 600 families × 4 worlds, 13 slots × 1–3 layers) | ≈ $0.65 | ≈ $0.85 |
| **Proposed revised cumulative cap** | — | **$6** (covers DEV extension + TEST; worst case ≈ **$5.60**) |

The DEV extension fits under the current $5 limit even at its cap. **Only an eventual TEST needs the $6 raise.**

## 4. GPU request

```text
AUTORISATION DEMANDÉE — BTC-H1c-DEVX
Objectif scientifique : extension unique du DEV H1c (3 000 mondes, familles indépendantes) pour tester, selon des critères d'arrêt pré-déclarés (SC1 PC1 ≥ 0,80 ; SC2 généralisation ; SC3 décodage de la continuation choisie), si l'instrument devient suffisamment sensible ; sinon, clôture de H1c sans TEST.
Why needed / alternative : le diagnostic a montré un instrument limité par les données (courbe d'apprentissage encore croissante) ; aucune alternative CPU.
Exact change/command : `modal run src/h1c_devx_capture.py` (sha 72f2a5f8…) ; 3 000 mondes (inputs ab4ee61e…, positions c4c58f0b…), 13 positions (ANS + 12 clés d'arrivée) × couches {32,40,48}, même image et révision 842da379… ; contrôle de fidélité D3 gelé, 20 rejeux, empreinte de tokenisation, vérification des shards avant suppression du Volume ; portes G1, G2 (identité bit à bit, bloquante), G3 ; puis analyse CPU `python src/h1c_devx_analysis.py --run` (4 threads, plafond 2 h CPU).
Compute and cost : 1 × A100-80GB, cpu (4,4), mémoire 16–96 GiB, max_containers=1, retries=0, timeout 950 s, garde 900 s ; ≈ 13 min distantes ; ≈ 1,3 Go ; coût attendu ≈ 0,76 $, plafond ≈ 0,99 $ ; cumul BTC ≤ ≈ 4,75 $ (sous la limite actuelle de 5 $).
Risk and stop condition : arrêt sans relance sur tout échec ; échec de démarrage → arrêt immédiat (watchdog) ; G2 ou G3 en échec → analyse refusée, rapport négatif ; SC1/SC2/SC3 en échec → clôture de H1c sans TEST. Aucun TEST, aucun accès à LOCKED-TEST.
I will not execute this step until you explicitly approve it.
```

**Separate decision:** raising the cumulative cap to **$6** (only needed for an eventual TEST).

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently (no training in this step)
[YES] Preserved oracle isolation and train/val/test independence (frozen TRAIN/SELECT/EVAL family roles; analysed families TRAIN only; no TEST)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (GPU requested, not run; $5 limit in force)
[YES] Preserved negative results and raw logs (all earlier results historical)
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decisions on BTC-H1c-DEVX and on the $6 cap.
```
