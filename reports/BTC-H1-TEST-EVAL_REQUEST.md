# BTC-H1-TEST-EVAL — request for the single confirmatory H1 evaluation on LOCKED-TEST

**Date:** 2026-11-14 · **Status:** PREPARED, **NOT RUN.**

> **Current state:** no analytical unlock exists. The oracle (`worlds.jsonl`) and the sealed TEST activations have never been read.

## 1. What would run

| Item | Value |
|---|---|
| Command | `.\.venv-cpucheck\Scripts\python src\sealed_stage\h1_eval_test.py --run` (CPU, 4 threads) |
| Evaluator | sha `a967c42e…`; checks its own hash against `BTC-H1_FREEZE_MANIFEST.json → prepared.BTC-H1-TEST-EVAL` |
| Inputs, all hash-verified first | 16 frozen predictors (`artifacts.H1-DEV-TRAIN`); `STANDARDISATION.npz` `9cd6b229…` (DEV-train only); TEST activations seal `LOCKED-TEST-D3-ACT` (`fe7c5756…`); LOCKED-TEST pool seal |
| Reads | only through `staged_read(stage="H1-TEST-EVAL")`, each logged: `worlds.jsonl` and `model_inputs.jsonl` (pool), plus every file of the activation seal |
| One attempt only | `EVAL_STARTED.json` is written before the first sealed read. Any second start is refused |
| Outputs | `outputs/h1_test/EVAL_RESULT.json`, `outputs/h1_test/per_world.jsonl` |
| Expected cost | CPU only. A few minutes: 3.4 GB read and hashed, plus 10,000-resample bootstraps. Cloud cost $0 |

## 2. Frozen decision procedure (H1 v0.3 + owner decisions of 2026-11-14)

1. **Admissibility:**
   - The model's forced choice (logp_A vs logp_B, taken from the capture) must equal the oracle optimum.
   - An exact tie is not admissible. The capture logged 9 original exact ties.
2. **Primary population:** admissible depth-4 `upath` pairs.
3. **INSUFFICIENT** if there are fewer than 180 pairs or fewer than 100 contributing families. In that case results are descriptive only.
4. **H1 MET** iff, for every k ∈ {B1, B2, B3, B6}, the lower bound of probe − B_k (95 % family-cluster percentile bootstrap, 10,000 resamples, seed 0, the same resamples for every k) is > 0.
   - **B2 (C-1):** predictors trained on permuted targets and **scored against the true u2 target** on the same pairs; max(B2₀, B2₁, B2₂) is taken inside each replicate.
   - **B6:** joint distinct-anchor assignment, with expectation p₁·p₂.
5. **L2, only if MET** (all three needed):
   - **(a)** same-layer order test, `order_u1` + `primary_u2` + `order_u3` at ℓ\* = 32, on admissible depth-4 `base`/`cf` worlds (C-2), vs max(1/6, B3-order);
   - **(b)** alias generalisation vs max(1/6, B3);
   - **(c)** shuffle generalisation vs max(1/6, B3).
6. **Secondary analyses (O-2):** every pre-registered secondary is listed with its status in `plan_status`.

| ID | Analysis | Status |
|---|---|---|
| S1 | depth 3 + 4 pooled `upath` | INFERENTIAL, probe − B3; **Holm** |
| S2 | `xswap`, depth 4 | INFERENTIAL, probe − B3; **Holm** |
| S3 | per-depth reconstruction (u1/u2/u3 heads, anchors) | DESCRIPTIVE |
| S4 | decoy control | **NOT COMPUTABLE**: 10 or 7 candidates vs 6-way frozen decoders; restricting the candidates would need the oracle |
| H2 | layer profiles | **NOT RUN**: one vector index per frozen head; would need new training |
| B7 | `cpath` chosen-branch reference | DESCRIPTIVE |
| B4 | answer-only | NOT COMPUTED: 0 by construction |

- The Holm family {S1, S2} is labelled in the output as **a subset of the pre-registered plan, not the whole plan**.

**What each outcome permits (v0.3 §7):**

| Outcome | Strongest statement allowed |
|---|---|
| INSUFFICIENT | descriptive only |
| NOT MET | H1 not supported beyond the baselines |
| MET, L2 not attributed | L1 |
| MET + L2 | ordered, label- and order-general unchosen-path information at the pre-answer position. Not forward construction (P0), not sequential sweep, not causal use |

**Prior expectation (DEV, exploratory):** the selected probe was at chance on DEV-val (0.030 vs 0.028) and below B3 (0.156). NOT MET is the most likely outcome. It would be reported as such.

## 3. Verification status

- **Synthetic tests** (cpucheck 44/44, `outputs/logs/d3test_r1_post_tests_cpucheck.log`):
  - planted signal → MET;
  - null → NOT MET;
  - frozen thresholds on a small sample → INSUFFICIENT;
  - C-1 scoring against the true u2;
  - O-2 status of every secondary;
  - single-evaluation guard;
  - stage scopes.
- **Main suite:** 89/89, including the amended seal test D3-A4.
- **Hashes:** recorded in `prepared.BTC-H1-TEST-EVAL`.
- **Code change since your last review:** apart from C-1 and O-2, one line in `run()`. The evaluator now checks its own hash against `prepared.BTC-H1-TEST-EVAL` instead of the stale `prepared.BTC-D3-TEST` entry.

## 4. Authorisation requested

```text
AUTORISATION DEMANDÉE — BTC-H1-TEST-EVAL
Objectif scientifique : décider H1 (primaire) une seule fois sur LOCKED-TEST selon les règles gelées v0.3 et les décisions C-1, C-2, O-2 ; L2 seulement si H1 est atteint.
Why needed / alternative : c'est l'unique étape confirmatoire prévue ; les prédicteurs, la normalisation, les seuils et le code sont gelés et hachés avant tout accès.
Exact change/command : écriture du déverrouillage analytique (portée H1-TEST-EVAL : worlds.jsonl + model_inputs.jsonl du pool, et tous les fichiers du sceau LOCKED-TEST-D3-ACT), puis `python src/sealed_stage/h1_eval_test.py --run` (sha a967c42e…) ; sorties EVAL_RESULT.json + per_world.jsonl.
Compute and cost : CPU local, 4 threads, quelques minutes ; aucun coût cloud.
Risk and stop condition : une seule tentative (marqueur écrit avant la première lecture) ; arrêt sans relance si un pré-contrôle ou une empreinte échoue ; un échec technique après le marqueur exige une nouvelle décision ; aucun ajustement après lecture ; tout résultat, y compris INSUFFICIENT ou NOT MET, est rapporté tel quel.
I will not execute this step until you explicitly approve it.
```

**Unlock files I would write on approval** (each with your approval id):

- `outputs/approvals/unlock/H1-TEST-EVAL/UNLOCK_LOCKED-TEST-D5.json`, files: [`worlds.jsonl`, `model_inputs.jsonl`]
- `outputs/approvals/unlock/H1-TEST-EVAL/UNLOCK_LOCKED-TEST-D3-ACT.json`, files: `"*"`

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (oracle and TEST activations unread)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (evaluation requested, not run)
[YES] Preserved negative results and raw logs
[YES] Labeled unrun stages NOT RUN and limitations honestly (S4 not computable, H2 not run)
Any deviation: none.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner decision on BTC-H1-TEST-EVAL.
```
