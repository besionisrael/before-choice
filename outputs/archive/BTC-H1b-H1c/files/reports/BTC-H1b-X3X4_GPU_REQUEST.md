# BTC-H1b X3 + X4 — protocol, controls, cost and GPU request (multi-position capture on PROBE-DEV)

**Date:** 2026-11-14 · **Authorisation for this document:** owner, "DÉCISION SCIENTIFIQUE — BTC-H1b, PRÉPARATION X3 + X4" (CPU preparation only) · **Status:** PREPARED, **NOT RUN.**

> **Scope.** Everything here is EXPLORATORY: PROBE-DEV already served for H1 selection. LOCKED-TEST is consumed and inaccessible. No new TEST pool, no change to H1 v0.3 results, no GPU used.

## 1. Question

Is the label-decoding failure (H1, and the X1 positive controls c1 / A1 at chance) caused by:
- **the static candidate keys** (representation), or
- **the observation position** (`<channel|>`)?

Answering this decides whether the instrument can be made valid for u2 at all.

## 2. Design (one capture, one analysis; factorial on the same worlds)

**Worlds:**
- The **3,480 PROBE-DEV training-rule worlds**: 2,194 at depth 4 and 1,286 at depth 3.
- These are exactly the worlds of H1-DEV-TRAIN and H1b.
- Admissibility comes from the D3-DEV-R1 forced choice; as in H1, the oracle is used only for selection.

**Positions** (`src/d3x_positions.py`): 22 slots per world, computed on CPU from the prompt and the pinned tokenizer.

| Slot | Position | Purpose |
|---|---|---|
| 0 | `ANS` = `<channel\|>` (pre-answer, as in H1) | reference query position |
| 1–12 | each candidate's **arrival** (`… leads to X.` / `… next state is X.`) and **departure** (`From X, …`) occurrence: last token of the label | contextual candidate keys (X3) |
| 13, 14 | end of the action lines of A and B | query positions **before** the rewards are seen |
| 15–20 | end of each `From …` rule line | query position (last rule line) |
| 21 | end of the instruction line (`… Respond with A or B only.`) | query position after the rewards |

**Layers:** 9 layers, {0, 8, 16, 24, 32, 40, 48, 56, 61}.
- Step 8 covers the full depth.
- It contains ℓ\* = 32 of H1 and the X1 peaks (choice from 36–40, reward gap at 40–48).
- Layer 0 serves as the purely lexical control (token embedding of the label's last token).
- Layer 61 is the raw pre-norm output.
- A capture of all positions × all layers is not needed (§5).

**Analysis** (`src/h1b_x3x4_analysis.py`):
- Factor **representation** R ∈ {static e(c) as in H1, contextual arrival key, contextual departure key}, with the key layer equal to the query layer.
- Factor **query position** Q ∈ {ANS, Q_END, ACT_END_A, ACT_END_B, LAST_RULE_END}.
- Factor **layer** (9).
- Slots A1, c1, c2, u2.
- **The same H1 bilinear decoder** (r = 8, λ = 0.1, **3 seeds everywhere**, amended), **the same worlds, the same family splits** as H1b (inner-train / inner-val for every choice; DEV-val reported only), **the same metrics**: single-world accuracy, and paired accuracy for u2.
- **Representation effect** = contextual − static at fixed Q and layer. **Position effect** = Q vs ANS at fixed R.

## 3. Controls against shortcuts and leakage

| Control | What it excludes |
|---|---|
| **Key-only decoder** s(c) = wᵀk_c (no query), per contextual R and layer | the key alone revealing the slot from its context, e.g. "this label follows *action A*" |
| **Choice + contextual keys** (amended) s(c) = w_choiceᵀk_c, using the model's chosen letter and no query activation | c1 being rebuilt from the chosen letter plus the candidates' context alone. It accompanies the interpretation; the thresholds are unchanged |
| **Layer-0 keys** | purely lexical identity of the label tokens |
| **Pre-reward query positions** (amended) | ACT_END_A/B and LAST_RULE_END precede the reward lines. Their performance is **measured and reported as a control analysis**, with no automatic causal interpretation |
| **B3+ prompt baseline** | information readable from the prompt text with the model's choice |
| **No oracle input** | candidates are built from the prompt only (`candidates_from_prompt`, canonical order); positions use prompt and tokenizer only; the oracle is used only for labels |
| **Token-position alignment** | verified for all 3,480 worlds (`outputs/d3x/plan/audit.json`): every label occurrence ends on a token boundary and decodes to the label end; line ends decode to "."; the last token is `<channel\|>`. **The container asserts that its token ids equal the locally computed ones (sha256 per world)** |

## 4. Positive controls and sensitivity rule (fixed now, before any capture)

**Cell selection:** the cell (R, Q, layer) with the best **inner-val** c1 accuracy.

**On DEV-val, the instrument is SENSITIVE iff all three hold:**
- **S-A:** c1 accuracy ≥ 0.80. c1 is printed next to the chosen letter.
- **S-B:** c1 − key-only(c1; same R and layer) ≥ 0.25. The query must contribute the choice; key-only can reach at most about 0.5, between the two action-line labels.
- **S-C:** A1 accuracy ≥ 0.80 in the same cell.

**Consequences:**
- **If SENSITIVE:** u2 at that cell (all slots, 3 seeds) is reported against B3+ and key-only, single and paired. It is then **informative about E1**.
- **If NOT SENSITIVE:** u2 is reported but declared **uninformative**. E3 and E4 then remain open in the form "no valid label-identity read-out at the captured positions".

**Expected signatures:**

| Pattern | Reading |
|---|---|
| contextual ≫ static at ANS | E3: representation was the problem |
| c1 at Q_END / ANS vs pre-reward positions | reported as a control; no automatic causal reading (amended) |
| static fails everywhere, contextual succeeds only at the label's own positions | E4 / binding not carried to the answer position |

## 5. Volume, storage, memory, duration, cost

| Item | Estimate |
|---|---|
| Captured vectors | 68,844 valid positions × 9 layers (padded storage 3,480 × 22 × 9 × 5,376 bf16) |
| Storage / download | **7.41 GB** (9 shards of ≤ 0.85 GB); local disk 248 GB free |
| An exhaustive capture, for comparison | all ~300 tokens × 62 layers ≈ 700 GB: avoided |
| GPU memory | as in D3 (peak 63.4 GB on A100-80GB); the gathered positions add < 1 MB per world |
| Remote time | load 100–245 s + 3,500 forwards ≈ 460 s (0.131 s/world measured in D3) + writing 7.4 GB ≈ 60–120 s → **≈ 650–850 s** (guard 1,420 s, timeout 1,500 s) |
| Download (after the container stops) | ≈ 280 s at the ~27 MB/s measured |
| **Cost** | expected ≈ **$0.80–0.95** (GPU + CPU + memory + egress of about 7.4 GB). **Worst case $1.50** (1,560 s at the worst-case rate) **+ about $0.10 egress → cap $2.00** |
| Cumulative after the run | ≤ about $4.43 under the $5 limit. **Note:** this leaves little margin for further runs |
| CPU analysis afterwards | ≈ 1,080 bilinear + 216 key-only + B3+ fits; estimate 1–2 CPU-h; **cap 6 CPU-h**, 4 threads (a separate authorisation is needed to run it, or include it in this one) |

## 6. Integrity and stop conditions

**Fidelity:** same frozen D3 fidelity check at `ANS`.
- max\|Δ\| ≤ 0.25, pre- and post-norm;
- A/B decision preserved, with the original-tie and τ rules;
- non-finite values rejected;
- **20 bitwise replays**;
- the tiny-model test shows that the ANS slot is **identical** to the frozen `capture_world` vectors.

**Transfer:** shard hashes and shapes are verified **before** the ephemeral Volume is deleted.
- Afterwards, a **replication check** compares the ANS vectors with the D3-DEV-R1 capture at the same 9 layers. It reports bitwise-equal counts and max\|Δ\|.
- A large discrepancy would stop the analysis.

**Stop with no retry** on:
- any failure of preflight, tokenisation sha, fidelity, replay, transfer or integrity;
- a container startup failure: immediate `modal app stop`, with the watchdog as in R1.

**Single run:** the run refuses to start if `outputs/d3x/probe_dev/MANIFEST.json` exists. The analysis refuses to run a second time.

**Tests:**

| Suite | Result |
|---|---|
| Gemma (tiny random model, real tokenizer, template and prompts) | 9/9 |
| cpucheck (including synthetic X3/X4: the planted pointer signal is recovered only by contextual keys at the right position, 0.64 vs ≈ 0.17 for static, departure keys or the wrong position) | 50/50 |
| Main | 91/92 (the only failure is the historical D3-A4 test, as documented) |
| Post-evaluation seals | 4/4 |

- Hashes: `outputs/d3x/plan/PREPARED.json`.

**Code hashes:**
- capture script `ffdef1aaf2cb…`
- capture core `0284cd4233ad…`
- analysis `4d89d809d4f4…`
- positions `2663054996…`

## 7. Authorisation requested

```text
AUTORISATION DEMANDÉE — BTC-H1b-X3X4
Objectif scientifique : déterminer si l'échec du décodage des labels (H1 et contrôles c1/A1) vient des clés statiques ou de la position d'observation, en validant d'abord la sensibilité de l'instrument sur c1/A1 (exploratoire, PROBE-DEV uniquement).
Why needed / alternative : les activations PROBE-DEV existantes ne contiennent que la position pré-réponse ; les clés contextuelles et les autres positions exigent une nouvelle passe avant ; pas d'alternative CPU.
Exact change/command : `modal run src/h1b_x3x4_run_gemma.py` (sha ffdef1aa…) ; 3 480 mondes PROBE-DEV (règle d'entraînement H1), 22 positions × 9 couches {0,8,16,24,32,40,48,56,61}, même image et même révision 842da379… ; contrôle de fidélité D3 gelé, 20 rejeux, vérification des shards avant suppression du Volume, contrôle de réplication vs D3-DEV-R1 ; puis analyse CPU `python src/h1b_x3x4_analysis.py --run` (plafond 6 h CPU, 4 threads).
Compute and cost : 1 × A100-80GB, cpu (4,4), mémoire 16–96 GiB, max_containers=1, retries=0, timeout 1 500 s, garde 1 420 s ; durée distante attendue ≈ 11–14 min + téléchargement ≈ 5 min ; stockage 7,4 Go ; coût attendu ≈ 0,80–0,95 $, plafond 2,00 $ ; cumul ≤ ≈ 4,43 $ sous la limite de 5 $.
Risk and stop condition : arrêt sans relance sur tout échec (prévol, empreinte de tokenisation, fidélité, rejeu, transfert, intégrité) ; échec de démarrage → arrêt immédiat de l'application (watchdog) ; aucun accès à LOCKED-TEST ni nouveau pool ; résultats exploratoires, règle de sensibilité S-A/S-B/S-C fixée d'avance.
I will not execute this step until you explicitly approve it.
```

**Separate decision points:**
1. Authorise the capture alone, or the capture plus the CPU analysis (6 CPU-h cap).
2. Whether to accept the reduced margin under the $5 Modal limit, or raise the limit.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (no H2/H3 work)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (PROBE-DEV only; LOCKED-TEST inaccessible; oracle only for labels/selection)
[YES] Did not equate token availability with prospective path representation (key-only, layer-0 and causality controls)
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (GPU requested, not run)
[YES] Preserved negative results and raw logs
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decision on BTC-H1b-X3X4.
```

## 8. Amendments of 2026-11-14 (owner decision "BTC-H1b-X3X4", applied before the capture)

1. **Choice + contextual keys control** added for both contextual representations, every layer and every slot, and at the selected cell. It is reported next to the S-B key-only control. The thresholds are unchanged.
2. **Pre-reward positions:** the causal claim is withdrawn. They are a measured control with no automatic interpretation.
3. **Harmonisation:**
   - 3 seeds for every decoder.
   - Missing positions are declared: depth-3 masks (slots 9–12, 19–20); PAD gets probability 0; nothing is imputed; an unexpected gap raises an error.
   - c1 and A1 at the selected cell are reported separately for model choice A and choice B.
4. **Precondition of the analysis:** every ANS vector must be bitwise identical to D3-DEV-R1. If not, the analysis stops.
5. **Limits confirmed:** the cumulative cap of $5 is kept, not raised.

The frozen plan is in `outputs/d3x/plan/PREPARED.json` (hashes) and committed before the capture.
