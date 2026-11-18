# BTC-V2 — run preparation (CPU) and authorisation request for BTC-V2-S1

**Date:** 2026-11-13 · **Basis:** BTC-V2-PROTOCOL_FREEZE v1.0 (F-1, F-2, F-3, F-5 validated; F-4 in principle).
**No GPU has been used and no model weights have been downloaded in this phase.** Only tokenizer, config and template files were downloaded.

## 1. What was prepared

| File | Role | sha256 |
|---|---|---|
| `src/v2_common.py` | frozen constants: models and revisions, template tails, A/B ids, pool hashes and prompt digests, stages, timeouts, CPU/RAM limits, rates, worst-case cost | `8f6d6a53…` |
| `src/v2_remote.py` | in-container inference: assertions, single forward pass, full-vocabulary top-1/top-5, forced A/B log-probs, replay of 20 worlds, JSON-only output | `fef0dcda…` |
| `src/v2_local.py` | local driver: reads only the oracle-free pool file (both hashes verified), computes expected token counts, calls Modal once, refuses a second execution of the same run id | `555eff1f…` |
| `src/v2_run_std.py` | Modal app for Qwen3-32B / Olmo (image: torch 2.5.1, transformers 4.57.1, hub 0.36.2, accelerate 1.10.1, numpy 2.1.3, jinja2 3.1.6) | `19bc60d9…` |
| `src/v2_run_gemma.py` | **separate** Modal app for Gemma 4 (torch 2.6.0, transformers 5.23.0, hub 2.4.0, tokenizers 0.23.3, safetensors 0.8.0, accelerate 1.10.1) | `5b25f5ee…` |
| `src/score_v2.py` | local scoring of S1–S4 / C1–C5 exactly as frozen, plus the pre-registered selection rule | `8dde6799…` |
| `src/v2_cpu_checks.py` | CPU pre-flight per model in the environment that mirrors its image | `d8024a82…` |
| `tests/test_v2_run.py` | 14 tests: budget, decorators, image pins, isolation, cross-environment token counts, scorer | — |

Reused from validated D2 components:
- JSON-only return, so no torch object can cross the wire.
- Exceptions converted to strings.
- `retries=0`, wall guard, prompt-hash and token-count assertions.
- Tie rule and family bootstrap (`score_d2_competence.py`).

**Unchanged:** criteria S1–S4 / C1–C5, the SCREEN/COMPETENCE datasets (hashes re-verified), the prompts, the task and the metrics. PROBE-DEV and LOCKED-TEST were not generated, and no activations are captured.

## 2. Checks performed (all PASS)

**Local test suite:** **67/67 OK** (`outputs/logs/unittest_v2_prep.log`).

**CPU pre-flight per model** (`outputs/v2_audit/cpu_check_<model>.json`, logs `outputs/logs/v2_cpu_checks_*.log`):

| Check | Qwen3-32B (torch 2.5.1 / tf 4.57.1, py 3.11) | Olmo-3.1-32B (same env) | Gemma-4-31B (torch 2.6.0 / tf 5.23.0, py 3.11) |
|---|---|---|---|
| A/B ids, single tokens | 32 / 33 ✓ | 32 / 33 ✓ | 236776 / 236799 ✓ |
| Official template + frozen tail on all V2 prompts | 1,560/1,560 ✓ | 1,560/1,560 ✓ | 1,560/1,560 ✓ |
| Pre-answer position (last template token) | `ĊĊ` after `</think>` | `Ċ` after `assistant` | `<channel\|>` |
| Local rendering = container rendering | ✓ (same library version) | ✓ | **jinja2 (local) = transformers 5.23.0 (container): 1,560/1,560 identical strings** |
| Local token counts = container-environment counts | 1,560/1,560 ✓ | 1,560/1,560 ✓ | 1,560/1,560 ✓ |
| Real config on meta device: layers / hidden states | 64 / 65 ✓ | 64 / 65 ✓ | 60 / 61 ✓ |
| Parameters (meta) | 32.76 B | 32.23 B | 31.27 B (incl. vision tower) |
| Logits vocabulary (top-1 computed over it) | 151,936 | 100,278 | 262,144 |
| `infer()` end to end on CPU with a tiny random model of the same architecture + real tokenizer | ✓ 45 results, replay identical, JSON-serialisable | ✓ | ✓ |
| Tampered payloads rejected (extra field, wrong digest, wrong token count) | 3/3 ✓ | 3/3 ✓ | 3/3 ✓ |

**Full-vocabulary top-1 (requirement 3):**
- `top1_id` is the argmax of the log-softmax over the **whole** output vocabulary (`vocab_size` is recorded per world and checked against the meta model).
- `top1_in_AB` is computed from it, independently of `logp_A`/`logp_B`.
- `tests/test_v2_run.py::test_top1_conformity_independent_of_forced_choice` shows that a model correct on 100 % of forced A/B choices but with top-1 = "Let" on 10 % of worlds **fails S4**.

**Budget (requirement 6):**
- Modal accepts the resource objects exactly as specified (`milli_cpu=4000/4000`, `memory_mb=16384/98304`, `A100-80GB`); verified with the client's own converter.
- Worst-case rate $0.00095952/s gives:
  - **$0.846** per screening run (objective $1.00).
  - **$1.287** for the competence run (objective $1.50).
  - **$3.825** for the programme (objective $4.50).
  These match BTC-V2-PROTOCOL_FREEZE §7 and are asserted in tests.
- Disk: Modal's default per-container quota is 512 GiB ≥ ~65 GB of weights, so no extra disk request (which would raise memory billing).

**Bugs found and fixed during preparation:**
1. `render()` treated the tokenizer as a rendering function, because tokenizers are callable. This would have aborted S1 inside the container at the first prompt. It was caught by the CPU pre-flight; the fix checks `apply_chat_template` first.
2. My first Gemma image pin of `huggingface_hub` (1.15.0) did not match the verified environment (2.4.0). Corrected, and a test now enforces equality with `outputs/v2_audit/gemma4_env_freeze.txt`.

## 3. Known limits of this preparation

- The real forward pass of each 32B model, GPU memory use, download speed and the actual allocation wait **cannot be verified without a GPU run**.
- The CPU end-to-end test uses a tiny random model with the real tokenizer and architecture class, not the real weights.
- Olmo's config emits a `rope_scaling` type warning (integer beta values) under transformers 4.57.1. It is harmless for config parsing but unverified at runtime.
- Gemma 4 runs on a different torch version (2.6.0) than M1/M2 (2.5.1), as required by transformers 5.23.0.

---

## 4. AUTHORISATION REQUEST — BTC-V2-S1 (Qwen3-32B screening)

```text
AUTORISATION DEMANDÉE — BTC-V2-S1
Objectif scientifique : criblage préenregistré S1–S4 de Qwen3-32B (prérequis de compétence pour H1), sur le réservoir SCREEN (240 mondes neufs), sans raisonnement textuel.
Why needed / alternative : seul moyen de savoir si un modèle plus capable satisfait la condition préalable ; aucune alternative CPU.
Exact change/command : modal run src/v2_run_std.py --run-id BTC-V2-S1
  — modèle Qwen/Qwen3-32B @ 9216db5781bf21249d130ec9da846c4624c16137, bf16, enable_thinking=False (bloc <think> vide vérifié), premier token ;
  — SCREEN : 240 mondes (60 familles) + 20 rejoués = 260 passes avant ; empreintes data cd3499d6… / inputs 42cc0517… / digest 6e52d9dc… ;
  — sorties : outputs/v2/BTC-V2-S1/{behavior_raw.jsonl, manifest.json}, puis notation locale `python src/score_v2.py BTC-V2-S1`.
Compute and cost : 1 × A100-80GB ; cpu=(4,4) ; memory=(16 GiB, 96 GiB) ; retries=0 ; max_containers=1 ; timeout 780 s ; garde interne 720 s ; aucun Volume.
  Coût attendu ≈ 0,40 $ ; pire cas calculé 0,85 $ ; objectif maximal 1,00 $ (marge 15 %, non garanti techniquement). Crédits restants ≈ 12,9 $ ; aucune facturation attendue.
Risk and stop condition : arrêt automatique si template/queue, identifiants A/B, nombre de tokens, empreintes ou champs d'entrée divergent, ou si la garde de 720 s est atteinte (résultat FAILED, aucune relance).
  Coût réel relevé immédiatement via `modal billing report --show-resources` ; si > 0,85 $, arrêt de toute la suite V2.
  Après S1 : notation locale, rapport, ARRÊT. S2, S3 et C1 ne seront pas lancés sans nouvelle autorisation, même si S1 réussit.
I will not execute this step until you explicitly approve it.
```

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (unchanged; D-1′ unchanged)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently (pinned revisions)
[YES] Preserved oracle isolation and train/val/test independence (oracle-free payload, tamper tests, sealed pools)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (no GPU run; S1 requested)
[YES] Preserved negative results and raw logs
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none. Two separate local CPU venvs created (.venv-cpucheck, .venv-gemma4) to mirror the container images; no weights downloaded.
Next step: owner decision on BTC-V2-S1.
```
