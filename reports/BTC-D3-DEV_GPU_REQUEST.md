# BTC-D3-DEV — GPU request: pre-answer activation capture on PROBE-DEV (NOT RUN)

**Date:** 2026-11-13 · **Basis:** BTC-H1 protocol v0.3 **FROZEN** (`reports/BTC-H1_FREEZE_MANIFEST.json`), BTC-D3_ACTIVATION_DESIGN v0.2 (D3-R1 approved).
**This document requests one GPU run. Nothing has been executed.** It does not request probe training, LOCKED-TEST capture or unlock. Each of those needs its own authorisation.

## 1. What the run does

| Item | Value |
|---|---|
| Command | `modal run src/d3_run_gemma.py --pool PROBE-DEV` (script sha `e056f9fe…`, core `d3_capture_core.py` hash in the freeze manifest) |
| Model | `google/gemma-4-31B-it` @ `842da3794eaa0b77d5f08bae87a17459d91ff475`, bf16, `enable_thinking=False`; torch 2.6.0, transformers 5.23.0 (separate image, as used in S3/C1) |
| Data | **PROBE-DEV only**: 4,800 worlds (300 families), model-input file sha `a3c81436…` (checked locally before sending). Payload is oracle-free: `world_id`, prompt, prompt hash, expected token count. LOCKED-TEST is not supported by the script (tested) |
| Captured per world | 62 × 5,376 bf16 vectors at `<channel\|>` (61 HF hidden states + raw block-60 pre-norm), `mean_emb` (5,376 fp32), forced A/B log-probs, full-vocabulary top-1/top-5 |
| Captured once | frozen candidate embeddings for all labels parsed from the prompts (fp32), with their token ids |
| Outputs (local) | `outputs/d3/probe_dev/`: 12 shards (`shard_k_acts.npy`, `shard_k_mean_emb.npy`, `shard_k_index.jsonl`), `candidate_embeddings.npy`, `candidate_labels.json`, `MANIFEST.json` (all sha256). ≈ 3.5 GB; git-ignored |

## 2. Resources, time, cost

| Item | Value |
|---|---|
| GPU / CPU / RAM | 1 × A100-80GB; cpu = (4, 4); memory = (16, 96) GiB; `retries=0`; `max_containers=1` |
| Timeout / internal guard | **1,500 s / 1,420 s** |
| Expected remote time | ≈ 850–960 s: load ~170 s (measured in S3/C1) + 4,820 forwards × ~0.13–0.15 s + shard writes |
| Expected cost | ≈ $0.75–0.85 compute + ≈ $0.13 egress (~3.5 GB download) → **≈ $0.9–1.0** |
| Worst case | (1,500 + 60) s × $0.00096/s + $0.04 + $0.15 egress ≈ **$1.69** |
| Spending objective | **$2.00** (15 % margin). An objective, not a technical guarantee: it relies on Modal honouring the timeout and published rates |
| Storage | ephemeral Volume, **deleted automatically** when the local context exits (also on error); no persistent Volume |
| Credits | ≈ $12 of this month's Starter credits remain; workspace spend limit $5 (applies to charges after credits) |

## 3. Integrity controls (abort on failure, no retry)

**Before sending (local, CPU):**
- PROBE-DEV model-input sha256 equals the manifest.
- 4,800 rows.
- Expected token counts computed with the container-equivalent rendering. Verified today: jinja2 and transformers 5.23.0 renderings are **identical on 4,800/4,800 PROBE-DEV prompts**, with identical token counts (max 287 tokens).

**In the container, per world:**
- Payload fields restricted; prompt hash, prompt digest and token count all match.
- A/B ids are 236776/236799.
- Template tail `<channel|>`.
- **Logit fidelity from the post-norm and from the pre-norm vector separately:** same top-1, same A/B decision, max |Δ| ≤ 0.25.

**In the container, run-level:**
- Bitwise replay of the first 20 worlds.
- Wall guard at 1,420 s.

**After download (local):**
- `verify_shards` re-hashes every shard and candidate file against `MANIFEST.json`, and checks shapes and dtypes.
- 4,800 worlds.

**Cost:** read immediately via `modal billing report --show-resources`. If it is above the $1.69 worst case, all further runs stop pending owner review.

**Stop after the run:** no probe training, no D3-TEST, no other run without a new authorisation.

## 4. Tests supporting this request

- Main suite **83/83**, including the static D3 checks: resources, timeout, ephemeral Volume, no sealed path, worst case ≤ 85 % of objective.
- `tests_gemma/test_d3_capture.py` **5/5**: tiny-model end-to-end with the real config structure and real PROBE-DEV prompts; both logit reconstructions; lossless bf16; `mean_emb`; frozen candidate embeddings; shard hashes and tamper detection; replay; payload rejection.
- `tests_cpucheck` **15/15**.

```text
AUTORISATION DEMANDÉE — BTC-D3-DEV
Objectif scientifique : capturer les activations pré-réponse de Gemma-4-31B-it sur PROBE-DEV (4 800 mondes), données de développement nécessaires à l'entraînement et à la sélection des sondes H1 selon le protocole v0.3 gelé.
Why needed / alternative : aucune alternative CPU (modèle 31B) ; LOCKED-TEST reste scellé et non capturé.
Exact change/command : modal run src/d3_run_gemma.py --pool PROBE-DEV — 62 vecteurs × 5 376 (bf16) + mean_emb + embeddings des candidats ; Volume éphémère supprimé automatiquement ; sortie outputs/d3/probe_dev/ (~3,5 Go).
Compute and cost : 1 × A100-80GB, CPU 4/4, RAM 16/96 Gio, retries=0, max_containers=1, timeout 1 500 s, garde 1 420 s ; attendu ≈ 0,9–1,0 $ ; pire cas ≈ 1,69 $ ; objectif 2,00 $ (non garanti techniquement).
Risk and stop condition : arrêt sur toute erreur d'intégrité (empreintes, tokens, template, fidélité des logits pré/post-norm, répétition bit à bit, vérification des shards) ou garde 1 420 s ; aucune relance ; coût relevé aussitôt ; arrêt après la capture, sans entraînement de sonde ni D3-TEST.
I will not execute this step until you explicitly approve it.
```

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (protocol frozen, unchanged)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (DEV only; LOCKED-TEST sealed, unread)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (this run requested, not executed)
[YES] Preserved negative results and raw logs
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none.
Next step: owner decision on BTC-D3-DEV.
```
