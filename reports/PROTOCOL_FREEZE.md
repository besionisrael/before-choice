# PROTOCOL FREEZE — DRAFT v0.3 (pilot protocol accepted; confirmatory protocol NOT frozen)

**Status:**
- D-1′ is accepted **for the feasibility pilot** (owner, 2026-11-13).
- The confirmatory protocol is not frozen.
- No GPU run may start without explicit approval of BTC-D2-001 (§10).

### Changes since v0.2 (owner requests 2026-11-13)
- **§0 new:** explicit evidential ladder (decodable relational information ≠ prospective representation ≠ sequential forward sweep).
- **§4b new:** prompt-only reconstruction control P0, descriptive, **not a threshold**. Results included.
- **§6b new:** mandatory separate reporting of chosen and unchosen branches, with admissible pair counts.
- **§5:** the 4 depth-4 pilot test pairs are explicitly non-confirmatory.
- **§9:** D-7 approved in principle, gated on competence results. D-8 proposed.
- **§10:** executable BTC-D2-001.

## 0. Evidential ladder — never collapse these levels

| Level | Name | What establishes it in this study | What does NOT |
|---|---|---|---|
| L1 | **Decodable relational information** | D-1′ probe beats B1–B3/B6 on held-out families | label presence, answer or outcome decoding |
| L2 | **Prospective path representation** | L1 for the **unchosen** branch, in admissible pairs, generalising to alias/shuffle, with **ordered** content (`before` metric) | L1 on the chosen branch only; L1 explainable by a backward anchor strategy (see P0) |
| L3 | **Functional prospective computation** | L2 + selective intervention effect on the choice (H3) | probe accuracy alone |
| L4 | **Sequential forward sweep (strong)** | measured, ordered, within-decision exploration of S→future, alternating between branches | layerwise decoding profiles; any result of this study as currently designed |

Every report states the highest level reached and names the next level as `NOT ESTABLISHED`. **L4 is UNTESTED by design** in this protocol.
**Date:** 2026-11-13 · **Data:** `pilot_v1` (`f9073bdf…`, unchanged) + `pilot_v1_pathcf` (`571444a8…`, new)
**Model:** Qwen/Qwen3-4B-Instruct-2507 @ `cdbee75f17c01a7cc42f958dc650907174af0554` (pinned under D-3)

### Changes since v0.1
- **§3 primary endpoint replaced (D-1 → D-1′).** The v0.1 target, the unchosen terminal, equals the lowest visible reward in 300/300 worlds (`reports/01b_endpoint_revision_D1.md`). It is demoted to a control.
- **§4:** baselines B6 (anchor-and-eliminate) and B7 (chosen-path reference) added; choice-preserving controls C5–C7 added.
- **§5:** splits of the new file inherited by family.
- **§7:** stop conditions added for choice changes under `upath`.
- **§9:** decision status updated, D-7 added.
- The GPU request now includes `upath` worlds and the pinned revision.

## 1. Research question (unchanged)

Do pre-answer activations of an open-weight LLM encode **ordered, action-conditioned representations of both future paths, including the unchosen one**, beyond strong lexical/prompt and answer-only baselines? And do such representations matter to the choice?

## 2. Hypotheses (unchanged; fixed hierarchy)

- **H1 (primary):** ordered, branch-conditional path information for **both** actions is decodable from the last pre-answer position, beyond baselines.
- **H2 (conditional on H1):** decodability shows a directional organisation compatible with forward construction, versus backward-from-reward or a parallel bag of states. Layers ≠ time. The strong sequential sweep is reported **UNTESTED** unless a separately approved design measures it.
- **H3 (conditional on H1):** selective interventions on path-related directions shift the choice beyond matched controls. Whole-residual patches are non-specific positive controls only.

## 3. Primary H1 endpoint — D-1′ (PROPOSED, replaces v0.1)

**Unchosen intermediate-state identification under choice-preserving path counterfactuals.**

| Element | Specification |
|---|---|
| Units | `upath` pairs (source world ∈ {base, cf}, its `upath` world). Labels, rewards, line positions, prompt length, token length and optimal action are identical within a pair; only the order of the non-optimal branch differs. |
| Primary stratum | depth 4: the target is ≥2 hops from every lexical anchor. Pre-specified secondary: depth 3+4 pooled. |
| Branch | oracle non-optimal action. Primary population: pairs where the model chose optimal on **both** members (truly unchosen). Also reported: all pairs, conditioned on the action letter. |
| Target | label at route depth 2 of that branch |
| Candidates | all intermediate labels of both branches (6 at depth 4) |
| Metric | **paired accuracy**: correct in both members. Random probe ≈ 1/36; **any decoder of answer/rewards/label inventory/length = 0 exactly** (proved by test). |
| Probe | candidate-conditioned L2-logistic on last-pre-answer residual stream; layer and C chosen by grouped CV on train + val; locked test evaluated once |
| Inference | family-cluster bootstrap 95 % CI (10,000 resamples) |

**H1 rating rule:**
- `unsupported`: the CI includes the best of B1–B3 or B6.
- `weak`: beats B1–B3 but not B6.
- `moderate`: beats all baselines, generalises to alias, and the depth-4 result holds.

Unchosen and chosen (`cpath`) results are always reported side by side.

**Secondary (Holm-corrected, labelled):**
1. `xswap` membership paired accuracy (depth 2 flagged lexical).
2. Pairwise order `before(s_i, s_j)` on `upath`.
3. Per-depth reconstruction on both branches.
4. Former v0.1 endpoint (unchosen terminal, base/cf), kept as a **positive/manipulation control only, never H1 evidence**.

## 4. Baselines and controls

| ID | Baseline / control | Rules out |
|---|---|---|
| B1 | probe on embedding/layer-0 activations | lexical presence |
| B2 | shuffled-target probe | probe capacity |
| B3 | prompt-only text features (label tokens + line positions) | reading the prompt without the model |
| B4 | answer + visible rewards decoder (= 0 on D-1′ by construction; ceiling on the old endpoint) | outcome/answer knowledge |
| B5 | candidate-only (no context) | candidate encoder solving alone |
| **B6** | anchor-and-eliminate: probe trained only on anchor-adjacent targets (depth 1 and terminal), applied by elimination | ≤1-hop lookups from action/reward lines |
| **B7** | same metric on `cpath` (chosen branch) | reference, not a threshold |
| C1–C4 | swap, alias, shuffle, decoy (v1) | answer letter, label identity, order, max-reward anchor |
| **C5** | `upath` | outcome-invariant change of the unchosen order |
| **C6** | `cpath` | same on the chosen branch |
| **C7** | `xswap` | branch membership |

The model's output preference enters every probe analysis as a nuisance covariate.

## 4b. Prompt-only reconstruction control P0 (DESCRIPTIVE — not a threshold)

`src/prompt_only_control.py` → `outputs/metrics/prompt_only_control.json`

P0 parses **only the prompt text** and runs explicit symbolic strategies, each with a budget of k rule applications. Its purpose is to map which computational shortcuts the prompt affords for each target. It is **not** a bar the model must beat: any complete symbolic solver eventually solves every item. It is distinct from B3, a learned bag-of-features baseline without chaining, which stays a threshold.

| Strategy | Mechanism |
|---|---|
| FWD | forward chaining from S. Knowing *which* branch is unchosen requires reaching the terminal: `depth` hops. |
| BWD− | backward chaining from the lowest printed reward (non-optimal branch) |
| BWD+ | backward chaining from the highest printed reward (optimal branch; misled by decoys) |
| ELIM | 1 hop from each anchor + elimination (2 composed hops) |

Minimal budget at which a strategy recovers ≥ 99 % of targets (identical for base, cf, upath, xswap):

| Depth | Branch | d=1 | d=2 | d=3 | d=4 |
|---|---|---|---|---|---|
| 4 | non-optimal | 3 (BWD−) | **2 (BWD−, ELIM)** ← primary target | 1 (BWD−) | 1 (BWD−) |
| 4 | optimal | 3 (BWD+) | 2 (BWD+, ELIM) | 1 | 1 |
| 4 | optimal, **decoy** worlds | 4 (FWD only) | 4 | 4 | 4 |
| 3 | non-optimal | 2 | 1 (BWD−) | 1 | — |

**What P0 tells us:**
1. For the **unchosen** branch, backward chaining from the lowest reward is always the cheapest route. It reaches the primary target in 2 hops, while forward reasoning needs 4 hops just to know which branch is unchosen.
2. The decoy control blocks backward anchoring only for the **chosen** branch. The unchosen branch is never protected, because the decoy carries the maximum reward, not the minimum.
3. Therefore a positive D-1′ result supports L1/L2 (relational multi-hop content of the unchosen path) but is **fully compatible with backward-from-reward computation**. It is not evidence of forward direction (L4 or H2).

These are properties of the task, not of the model. In D5/H2, the order of layer emergence for d=2 versus d=3 on the unchosen branch is the pre-specified *descriptive* contrast: terminal-adjacent first suggests backward, S-adjacent first suggests forward. It remains descriptive (layers ≠ time).

## 5. Splits and locked test

- `pilot_v1`: family-level 60/20/20 stratified by depth; test hash `34ab3841…`.
- `pilot_v1_pathcf`: inherits each family's split; test hash `20e559e0…`.
- All selection is on train/val. The test split is evaluated once per pre-registered analysis.
- **The pilot test has 4 depth-4 `upath` pairs** (2 families × {base, cf}), fewer still after admissibility.
  - **Whatever the outcome, they are not confirmatory evidence for H1.**
  - Pilot probe results will be labelled `PILOT / FEASIBILITY — NOT CONFIRMATORY` and reported with counts, never with H1 ratings.
  - A confirmatory D-1′ test requires the main dataset (D-5, deferred).

## 6. Analysis plan

- **D2 behaviour:**
  - Accuracy vs 0.5, by depth.
  - Paired cf-flip rate.
  - A/B bias (swap).
  - Decoy accuracy.
  - **Choice invariance under `upath`/`cpath`/`xswap`** (expected: the choice does not change).
  - Family-cluster bootstrap CIs.
  - Feasibility checkpoint: depth-2 base+cf accuracy ≥ 0.70 on train+val.
  - First-answer token: `A` (id 32) / `B` (id 33). Both are single tokens directly after `<|im_start|>assistant\n`, verified under D-3.
- **D3:** residual stream at the last prompt position, all layers, fp16. Token lengths are logged. All 360 counterfactual pairs are token-length aligned (D-3).
- **D4:** as §3. **D5:** layerwise decodability per depth and branch, descriptive; backward-from-terminal kept as a live alternative. **D6:** only if H1 ≥ weak, with separate approval.

## 6b. Mandatory reporting format: chosen vs unchosen, admissibility

Every D2/D4 report includes this table. Counts come from `src/score_d2.py`; the probe columns are filled at D4.

| Branch | Pair type | Constructed pairs | Choice invariant | **Admissible** (optimal chosen on both members) | Probe paired acc. (admissible) | Probe paired acc. (all, letter-conditioned) |
|---|---|---|---|---|---|---|
| Unchosen | `upath` d3 / d4 | n | n | n | — | — |
| Chosen | `cpath` d3 / d4 | n | n | n | — | — |

Rules:
- Chosen and unchosen are **never pooled** into one H1 number.
- An admissible count below 1 family-cluster per split is reported as `INSUFFICIENT`, not as a rate.
- Exclusions are listed by `world_id`.

## 7. Failure criteria (STOP)

1. Depth-2 accuracy < 0.70 → stop D3+.
2. Primary CI overlaps best of B1–B3/B6 → H1 `unsupported` for this model/task; no metric changes.
3. Oracle leakage or token misalignment → stop the affected analysis.
4. Test split touched during development → that test is void.
5. **New:** if `upath` changes the model's choice in > 10 % of pilot pairs, report it as a behavioural finding and halt D-1′ until reviewed. The construct "unchosen" is then unstable.

## 8. Not measured

- Within-decision sequential/alternating sweep (strong H2), biological time.
- Consciousness, sentience, welfare, intentional agency, VTE.
- Non-linear topologies, other models.
- **Forward vs backward direction is not identified by D-1′.** It tests relational multi-hop path content of the unchosen branch, not its direction of construction.

## 9. Decisions

| ID | Decision | Status |
|---|---|---|
| D-1 | v0.1 endpoint | **Withdrawn** (reward shortcut) |
| D-1′ | §3 endpoint, `upath` pairs, depth-4 primary | **Accepted for feasibility pilot** 2026-11-13; confirmatory freeze pending |
| D-2 | v1 controls + family split | Approved 2026-11-13 |
| D-3 | tokenizer audit, revision pin | Approved and **done** (`outputs/metrics/tokenizer_audit.json`) |
| D-4 | reward gap / decoy range as-is | Accepted for pilot |
| D-5 | main dataset | Deferred |
| D-6 | GPU smoke BTC-D2-001 | **Executable, NOT authorised** — see §10 |
| D-7 | depth 5–6 families for the main dataset | **Approved in principle** 2026-11-13; **no depth change before competence results** |
| **D-8** | (proposal) "min-reward decoy": unreachable chain carrying the **lowest** reward, to block the BWD− shortcut on the unchosen branch (§4b). Same task family; would be generated only for the main dataset, after competence results. | **AWAITING DECISION** — not implemented |

## 10. BTC-D2-001 — executable smoke test (NOT RUN, awaiting explicit approval)

**Purpose:** pipeline feasibility only. This means token assumptions, determinism, activation shapes and the scorer arithmetic, plus first sight of behaviour on 20 train worlds. It is **not** the D2 competence checkpoint, which needs all train+val worlds and a separate approval.

**Files:** `src/d2_smoke.py` (Modal) and `src/score_d2.py` (local scoring). Tests: `tests/test_d2_pipeline.py`, all passing. Dry import is verified without any remote call. Local and container tokenisation paths match on all 20 worlds.

**Worlds (fixed, train only, 20):**
- Families `F0001` (d2), `F0013` (d3), `F0020` (d4), each with {base, cf, swap_base, swap_cf}.
- `F0013` and `F0020` also with {upath_base, upath_cf, cpath_base, cpath_cf}.
- Every world is run twice (determinism replay).

**Exact commands (owner's PowerShell, project root):**
```powershell
.\.venv\Scripts\Activate.ps1
modal setup                          # one-time login (owner)
modal token info
# Owner sets Workspace budget in Modal "Usage & Billing" (suggest US$5) BEFORE the next line
modal run src/d2_smoke.py            # the ONLY billable command
python src/score_d2.py outputs/d2_smoke   # local CPU, free
```

**Hard limits in code:**
- `gpu="L4"`, `max_containers=1`, `retries=0`, `timeout=900` s.
- Internal wall-clock guard at 780 s.
- No Volume, no persistent storage, no endpoint, no schedule.
- The container exits after returning.

**Cost:**
- L4 Modal list price is about US$0.80/h (re-verify on modal.com/pricing before running). Expected 5–8 min, mostly the 8.06 GB weight download, so about US$0.10.
- Worst case at the 900 s timeout is about US$0.25 including CPU/RAM, plus a few cents of image build.
- **Max authorised spend requested: US$1.00.**

**Automatic stop conditions (assertions abort the run):**
1. A prompt has an unexpected field (oracle isolation) or a prompt hash mismatch.
2. Token ids for `A`/`B` are not 32/33, or the chat template does not end with `<|im_start|>assistant\n`.
3. The container token count differs from the local count (template/tokenizer drift).
4. The wall-clock guard is reached, or the timeout hits.

**Manual stop after the run:**
- No second run and no other world set without new approval.
- If the A/B probability mass is < 0.5 on any world, or replay choices differ, report and stop.

**Outputs:** `outputs/d2_smoke/{behavior_raw.jsonl, acts_last_pos_fp16.npy (40×37×2560), manifest.json, scored.json}`. The actual cost is read from Modal billing and recorded in the manifest.

```text
AUTORISATION DEMANDÉE — BTC-D2-001
Objectif scientifique : vérifier la faisabilité du pipeline D2/D3 (tokens A/B, déterminisme, capture pré-réponse, invariance du choix sous upath/cpath) sur 20 mondes train — prérequis à H1, pas une mesure de compétence.
Why needed / alternative : aucune alternative CPU réaliste pour un modèle 4B ; tout le reste est testé localement.
Exact change/command : modal run src/d2_smoke.py — Qwen3-4B-Instruct-2507 @ cdbee75f, bf16, 20 mondes × 2 passes, hidden_states 37 × dernière position, retour local ~3,8 Mo.
Compute and cost : 1 × L4, max_containers=1, retries=0, timeout 900 s ; attendu 5–8 min, ~0,10 $ US ; plafond demandé 1,00 $ US ; budget workspace Modal à fixer par vous (suggéré 5 $).
Risk and stop condition : assertions d'isolation/tokenisation/longueur, garde 780 s ; aucun stockage persistant ; aucune exécution enchaînée.
I will not execute this step until you explicitly approve it.
```

## 11. BTC-D2-002 — pre-registered D2 competence checkpoint (RUN 2026-11-13 → **FAIL**, 0.5625 < 0.70; D3+ stopped per §7)

**Unchanged from v0.3:**
- Model and revision (`cdbee75f`), bf16, official chat template, prompts (352/352 prompt hashes identical to the frozen dataset), `A`/`B` = tokens 32/33, first-answer-token scoring.
- Criterion: **depth-2 {base, cf} accuracy ≥ 0.70 on train+val; otherwise STOP D3+** (§6, §7).

**Population:** all train+val worlds, **352** = pilot_v1 train/val 240 + pathcf train/val 112. They are read only from `data/splits/{train,val,pathcf_train,pathcf_val}.jsonl`; the locked test files are never opened.
- **Checkpoint population:** 16 worlds (8 families: 6 train, 2 val).
- **Power caveat:** one family moves the estimate by 0.125. The 95 % family-bootstrap CI is reported, descriptive only.

**Tie addendum** (procedural; it does not change the criterion; fixed before data):
- A tie is `log p(A) == log p(B)` exactly. These ties come from bf16 logit quantisation; 2/20 occurred in BTC-D2-001b.
- **Primary score:** tie = 0.5, the expected value of a fair coin.
- **Bounds, always reported:** strict (tie = 0) and lenient (tie = 1).
- The decision uses the primary score. A `TIE_SENSITIVE` flag is raised when strict and lenient fall on opposite sides of 0.70, and it is reported to the owner. Ties are never re-scored post hoc, and no higher-precision rerun is made without approval.
- The raw bf16 logits for A and B are saved for audit.

**Reported (descriptive, not decision-relevant):**
- Accuracy by depth, by family, by split and by variant.
- Counterfactual sensitivity: choice change within base/cf, alias, shuffle, decoy and swap pairs; consistency under renaming, reordering and decoy.
- `upath`/`cpath`/`xswap` invariance and admissible counts, chosen and unchosen reported separately.
- A/B bias, A+B mass, replay determinism (first 20 worlds) and confident errors (|margin| ≥ 3).

**No activations are captured** (that is D3).

**Files:** `src/d2_competence.py` (`5ff8ee23…`), `src/score_d2_competence.py` (`f03e66a7…`), `tests/test_d2_competence.py` (38/38 project tests pass).
