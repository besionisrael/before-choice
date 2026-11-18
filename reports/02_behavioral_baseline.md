# D2 — Behavioural baseline: BTC-D2-002 competence checkpoint **FAIL** (+ BTC-D2-001/001b smoke)

> **PRE-REGISTERED DECISION: FAIL.** Depth-2 {base, cf} accuracy on train+val = **9/16 = 0.5625 < 0.70**. There are 0 ties, so the result is not tie-sensitive. 95 % family-bootstrap CI [0.50, 0.69].
> **Per PROTOCOL_FREEZE §7, D3 and all later stages (D4, D5, D6) are STOPPED.** No model, prompt or task change has been made.
> H1, H2, H3: **NOT TESTED**. This result is not evidence for or against them; the precondition for testing them is not met with this model and task.


> **Cost correction (2026-11-13, from `modal billing report --show-resources`, the authoritative per-app source):**
>
> | App | L4 | CPU | Memory | Egress | **Total** |
> |---|---|---|---|---|---|
> | BTC-D2-001 attempt 1 `ap-GV4MRnyC…` | 0.01416 | 0.00567 | 0.00028 | 0.00746 | **$0.0276** |
> | BTC-D2-001b `ap-w05OyvI2…` | 0.01244 | 0.00254 | 0.00019 | 0.00684 | **$0.0220** |
> | BTC-D2-002 `ap-YkjX7dMp…` | 0.01733 | 0.00292 | 0.00025 | 0.00672 | **$0.0272** |
>
> Plus image builds ≈ $0.01. **Cumulative D2 ≈ $0.087 in credits; $0 charged.** The earlier "≈ $0.01–0.02" figures read from the dashboard total were slightly low. Raw CSV: `outputs/v2_audit/modal_billing_2026-11-13_resources.csv`.

## A. BTC-D2-002 — competence checkpoint (2026-11-13)

**Run:**
- Command: `modal run src/d2_competence.py` (sha `5ff8ee23…`), app `ap-YkjX7dMpMtv3riNTTQBV1G`, exit 0. Single attempt, 1 × NVIDIA L4, `retries=0`, `max_containers=1`, `timeout=900`.
- Data: **352 worlds** (train 264 + val 88) read only from `data/splits/{train,val,pathcf_train,pathcf_val}.jsonl`. The locked test split was not opened; `test.jsonl` hash is unchanged at `34ab3841…`.
- Model `cdbee75f`, bf16, official chat template. Prompts are byte-identical to the frozen dataset (352/352 hashes). First-answer-token scoring of `A`(32) vs `B`(33). Scoring is local (`src/score_d2_competence.py`, sha `f03e66a7…`).

**Timing:**

| Phase | Value |
|---|---|
| Allocation wait (local call → remote start) | 1,213 s |
| Weights download + CPU load | 39.6 s |
| Move to GPU | 3.5 s |
| 372 forward passes | 24.9 s (≈ 67 ms each) |
| Transfer | 0.02 s |
| Remote total | ≈ 70 s |
| Local wall time | 1,283 s |

GPU memory peak: 8.32 GB.

**Cost:**
- Total usage went from $17.11 to $17.12, i.e. ≈ **$0.01–0.02 in credits**, **$0 charged**, with the spend limit at $5.
- The cumulative D2 campaign (001, 001b, 002) cost ≈ $0.05 in credits.

### A1. Pre-registered checkpoint

| Population | n | Families | Ties | Primary | Strict | Lenient | 95 % CI (family bootstrap) | Decision |
|---|---|---|---|---|---|---|---|---|
| depth-2 {base, cf}, train+val | 16 | 8 | 0 | **0.5625** | 0.5625 | 0.5625 | [0.500, 0.688] | **FAIL** |

### A2. Descriptive performance (no decision attached)

**{base, cf} by depth and split:**

| Subset | n | Ties | Primary | Strict–lenient | 95 % CI |
|---|---|---|---|---|---|
| depth 2 | 16 | 0 | 0.562 | 0.562–0.562 | [0.50, 0.69] |
| depth 3 | 16 | 2 | 0.500 | 0.438–0.562 | [0.41, 0.59] |
| depth 4 | 16 | 1 | 0.469 | 0.438–0.500 | [0.41, 0.50] |
| all depths | 48 | 3 | 0.510 | 0.479–0.542 | [0.46, 0.56] |
| train | 36 | 2 | 0.500 | 0.472–0.528 | [0.44, 0.57] |
| val | 12 | 1 | 0.542 | 0.500–0.583 | [0.50, 0.63] |

**By variant** (primary accuracy; n = 24 per v1 variant, 16 per upath/cpath):

| | base | cf | alias_b | alias_cf | swap_b | swap_cf | shuffle_b | shuffle_cf | decoy_b | decoy_cf | upath_b | upath_cf | cpath_b | cpath_cf | xswap_b | xswap_cf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| acc | 0.458 | 0.562 | 0.479 | 0.625 | 0.375 | 0.562 | 0.521 | 0.583 | 0.583 | 0.396 | 0.469 | 0.438 | 0.562 | 0.438 | 0.479 | 0.542 |

All variant CIs include 0.5.

**By family** (choice on base / cf; optimal on base is shown, and cf's optimal is the opposite):

| Family | Depth | Split | Acc. (all variants) | base | cf | Optimal (base) |
|---|---|---|---|---|---|---|
| F0000 | 2 | val | 0.50 | A | A | B |
| F0001 | 2 | train | 0.67 | B | B | A |
| F0003 | 2 | train | 0.42 | A | A | B |
| F0004 | 2 | train | 0.58 | A | A | A |
| F0005 | 2 | val | 0.50 | A | A | B |
| F0007 | 2 | train | 0.58 | A | A | B |
| F0008 | 2 | train | 0.67 | **A** | **B** | A |
| F0009 | 2 | train | 0.58 | A | A | A |
| F0010 | 3 | val | 0.53 | tie | A | B |
| F0012 | 3 | val | 0.50 | B | B | B |
| F0013 | 3 | train | 0.50 | A | A | A |
| F0014 | 3 | train | 0.50 | A | A | A |
| F0015 | 3 | train | 0.50 | tie | A | A |
| F0016 | 3 | train | 0.50 | A | A | B |
| F0017 | 3 | train | 0.53 | B | B | A |
| F0019 | 3 | train | 0.50 | B | B | B |
| F0020 | 4 | train | 0.44 | B | tie | A |
| F0021 | 4 | train | 0.44 | B | B | B |
| F0022 | 4 | train | 0.50 | A | A | A |
| F0024 | 4 | train | 0.47 | B | B | A |
| F0025 | 4 | train | 0.50 | A | A | B |
| F0026 | 4 | train | 0.50 | A | A | B |
| F0027 | 4 | val | 0.50 | A | A | B |
| F0028 | 4 | val | 0.38 | B | B | B |

**Only 1 of 24 families (F0008) answers base and cf correctly.** Per-family accuracy over all variants ranges 0.38–0.67.

### A3. Counterfactual sensitivity

| Pair | Optimal changes? | Pairs | Choice changed | Same choice | Involves tie | Both correct |
|---|---|---|---|---|---|---|
| base vs cf (deep swap) | yes | 24 | **1** | **20** | 3 | 1 |
| alias_base vs alias_cf | yes | 24 | 2 | 20 | 2 | 2 |
| shuffle_base vs shuffle_cf | yes | 24 | 4 | 18 | 2 | 3 |
| decoy_base vs decoy_cf | yes | 24 | 1 | 22 | 1 | 0 |
| swap_base vs swap_cf | yes | 24 | 1 | 22 | 1 | 0 |
| base vs swap_base (A↔B letters) | yes | 24 | 16 | 6 | 2 | 6 |
| base vs alias_base (renamed labels) | no | 24 | 5 | 16 | 3 | 7 |
| base vs shuffle_base (lines reordered) | no | 24 | **11** | 10 | 3 | 5 |
| base vs decoy_base (decoy added) | no | 24 | 2 | 20 | 2 | 10 |

**Reading (descriptive):**
- The decision is almost **insensitive** to the transition that determines the optimum: 1/24 changes on base/cf, and ≤ 4/24 for every other minimal pair.
- It is **sensitive to task-irrelevant surface changes**: exchanging the action letters changes the choice in 16/24 pairs, and reordering the rule lines in 11/24.
- This is the opposite of the profile a model that simulates the paths would show.

**Choice-preserving counterfactuals (§6b, chosen and unchosen reported separately):**

| Branch | Pair type | Depth | Constructed | Choice invariant | Admissible (optimal on both) |
|---|---|---|---|---|---|
| Unchosen | upath_base | 3 / 4 | 8 / 8 | 6 / 7 | 4 / 2 |
| Unchosen | upath_cf | 3 / 4 | 8 / 8 | 6 / 8 | 2 / 4 |
| Chosen | cpath_base | 3 / 4 | 8 / 8 | 7 / 6 | 4 / 2 |
| Chosen | cpath_cf | 3 / 4 | 8 / 8 | 8 / 7 | 3 / 4 |
| Both (membership) | xswap_base | 2 / 3 / 4 | 8 / 8 / 8 | 7 / 7 / 7 | 3 / 4 / 2 |
| Both (membership) | xswap_cf | 2 / 3 / 4 | 8 / 8 / 8 | 7 / 8 / 7 | 5 / 3 / 4 |

High choice invariance here is **not informative**: the model's choice does not track even the decisive edge. "Admissible" pairs are correct largely by chance (≈ 25 % expected under random choice), so they are not admissible in the scientific sense intended by D-1′.

### A4. Response properties and negative findings

- **Choices:** A 189, B 146, tie 17 (of 352). There is an A preference; A is chosen in 56.4 % of non-tie worlds.
- **Format:** in 9 worlds, all from family **F0021** (depth 4), the most likely first token is `Let`, i.e. the model starts writing a reasoning sentence despite "Respond with A or B only". A+B mass there is 0.23–0.45. Scoring still uses argmax(A, B) as pre-registered. Everywhere else, top-1 is A or B.
- **Confident errors:** 44 worlds are wrong with |log p(A) − log p(B)| ≥ 3.
- **Determinism:** the 20-world replay has identical choices and max |Δ log p| = 0.
- **POST HOC / EXPLORATORY (not pre-registered):** the model's choice agrees with simple surface rules at chance level only. The rules tested were first-listed action line 0.49, last-listed 0.51, alphabetically first root label 0.42, and action line nearest the max-reward line 0.52, against 0.56 for always-A. **No simple shortcut explains the behaviour.** The choices look family-specific and driven by surface form rather than by graph structure. This is unexplained.

### A5. Conclusion and what is NOT concluded

- **Justified:** with this prompt format and first-token scoring (no reasoning), Qwen3-4B-Instruct-2507 @ `cdbee75f` does not solve the depth-2 task above the pre-registered threshold, and its choices are almost insensitive to the decisive transition. The precondition for H1 probing is **not met**.
- **Not justified:**
  - That the model "cannot plan" in general.
  - That no prospective representation exists in its activations. That was not measured, and probing a model that does not solve the task would be uninterpretable for H1 as defined.
  - Any statement about H2 or H3.
- **Possible directions, each requiring an owner scientific decision (none taken):**
  - A different model.
  - Allowing reasoning tokens before the answer (changes the "pre-answer" construct).
  - A prompt-format change.
  - Simpler graphs.
  - Accept the negative result and stop.
- **Limitations:** the checkpoint rests on 16 worlds and 8 families; one prompt template; bf16 only; train+val only.

```text
SCOPE AUDIT (BTC-D2-002)
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence (test split never opened)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (BTC-D2-002)
[YES] Preserved negative results and raw logs
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none. Post-hoc heuristic check labelled POST HOC / EXPLORATORY.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — owner scientific decision after pre-registered FAIL.
```

---

## B. Earlier: BTC-D2-001 / 001b feasibility smoke

**Date:** 2026-11-13 · **Model:** Qwen/Qwen3-4B-Instruct-2507 @ `cdbee75f17c01a7cc42f958dc650907174af0554`, bf16 · **GPU:** 1 × NVIDIA L4 (Modal)

**Status:**
- **Pipeline: PASS** (BTC-D2-001b).
- **Behaviour on the smoke set: AT CHANCE, and insensitive to the counterfactual.** This is a 20-world smoke set, not the competence checkpoint.
- No further run. D2 competence, D3, D4 and D6 are **NOT RUN**.

> This smoke test is **not** a validation or refutation of H1, H2 or H3. With n = 20 worlds (3 families) it is also **not** the pre-registered D2 competence checkpoint, which needs all train+val worlds and separate approval. The behavioural numbers below are pipeline-level observations.

## 1. Runs

| Run | UTC | Modal app | Outcome |
|---|---|---|---|
| BTC-D2-001 attempt 0 | 12:15:44–12:15:46 | `ap-INHLdYFz…` | Local Modal client 1.2.1 incompatible with image builder 2026.10; **0 tasks, no compute**. Client upgraded to 1.6.4. |
| BTC-D2-001 attempt 1 | 12:16:31–12:31:32 | `ap-GV4MRnyC…` | Model loaded, but the result was undeserialisable locally (`torch.__version__` is a `TorchVersion` object); **outputs lost**. |
| **BTC-D2-001b** (owner-authorised rerun) | 12:39:02–13:00:36 | `ap-w05OyvI2…` | **Completed, exit 0.** All outputs saved. |

**Code differences between attempt 1 and 001b:**
- Serialisation fix: JSON/bytes-only return, remote exceptions converted to strings. Validated by 33 tests.
- Timing-only instrumentation (`time.time()` stamps and `torch.cuda.synchronize()` around the forward loops). This was added because the owner asked for timing breakdowns; it does not change any computation.
- Script sha256 as run: `9cf84868…`. The 33-test validated version is archived at `outputs/logs/d2_smoke_validated_33tests.py`, and the attempt-1 version at `outputs/logs/d2_smoke_as_run_attempt1.py`.

Unchanged throughout: model, revision, 20 worlds, limits (`gpu="L4"`, `max_containers=1`, `retries=0`, `timeout=900`, guard 780 s) and protocol.

## 2. Timing (BTC-D2-001b)

| Phase | Seconds | Source |
|---|---|---|
| Wait from local call to remote function start | **1,237.5** (20.6 min) | remote start epoch − local call epoch. The image was cached (no rebuild line in the log), so this is container/GPU allocation wait. Assumes synchronised clocks. |
| Tokenizer load | 1.64 | remote |
| Weights download (8.06 GB) + CPU load | 38.29 | remote |
| Move to GPU | 2.87 | remote, synchronised |
| **Model ready (total load)** | **42.8** | remote log |
| Forward passes, replay 0 (20 worlds) | **2.26** | remote, synchronised (includes first-call warm-up) |
| Forward passes, replay 1 (20 worlds) | **1.30** | remote, synchronised |
| Remote serialisation of activations (np.save) | 0.013 | remote |
| **Remote function total** | **46.4** | remote |
| Transfer, remote end → local receipt (~7.6 MB) | **1.34** | cross-machine |
| Local save of outputs | 0.003 | local |
| Whole command, local wall time | 1,294 (21.6 min) | local |

The ~20 min allocation wait is also the most plausible explanation for the previously unexplained ~12 min of attempt 1. That attribution is an **inference, not verified**. GPU memory peak: **8.34 GB**.

## 3. Reproducibility

| Check | Result |
|---|---|
| Decisions, replay 0 vs 1 | **identical, 20/20** |
| Max \|Δ log p\| (A and B), replay 0 vs 1 | **0.0** |
| Activations, replay 0 vs 1 | **bitwise identical** (max \|diff\| = 0.0) |
| Token counts vs local pre-computation | all equal (assertion did not fire) |
| `A` / `B` token ids | 32 / 33 confirmed in-container |

## 4. Activation capture

| Item | Value |
|---|---|
| File | `outputs/d2_smoke/acts_last_pos_fp16.npy`, 7,577,728 bytes, sha256 `cfe54b2d…` |
| Shape / dtype | **(40, 37, 2560)**, float16 = 20 worlds × 2 replays × (embeddings + 36 blocks) × d_model |
| Position | last prompt token (after `<\|im_start\|>assistant\n`), i.e. before the first answer token |
| Finite | all values finite |
| Mean L2 norm by layer (replay 0) | L0 1.1 · L1 5.6 · L9 32.6 · L18 45.0 · L27 135.4 · L35 531.2 · L36 163.9 |
| Distinct across worlds | min pairwise distance at layer 18 = 0.52 (no duplicated rows) |

Capture works as designed. These activations were **not** used for any probe (D4 not authorised).

## 5. Behavioural performance (smoke set, n = 20, 3 train families)

| Metric | Value |
|---|---|
| Accuracy (argmax of log p(A) vs log p(B)) | **10/20 = 0.50** |
| Exact ties (log p(A) = log p(B) in bf16) | 2 (`F0020-cf`, `F0020-upath_cf`). The scorer resolves ties as B, which counts both as correct. **Excluding ties: 8/18 = 0.44.** |
| Choice A / B | 10 / 10 |
| A + B probability mass | min 0.861, all top-1 tokens are A or B (format followed) |
| **base/cf pairs both correct** | **0/3** |
| swap pairs both correct / letter flipped | 2/6 / 4/6 |
| By depth (base+cf) | d2: 1/2 · d3: 1/2 · d4: 1/2 (counting the tie as correct) |

**Per-world decisions** (margin = log p(A) − log p(B)):

| World | Optimal | Choice | Margin | | World | Optimal | Choice | Margin |
|---|---|---|---|---|---|---|---|---|
| F0001-base | A | B ✗ | −1.25 | | F0013-upath_base | A | A ✓ | +3.75 |
| F0001-cf | B | B ✓ | −1.75 | | F0013-upath_cf | B | A ✗ | +5.00 |
| F0001-swap_base | B | A ✗ | +9.00 | | F0013-cpath_base | A | A ✓ | +4.00 |
| F0001-swap_cf | A | A ✓ | +10.50 | | F0013-cpath_cf | B | A ✗ | +3.75 |
| F0013-base | A | A ✓ | +4.50 | | F0020-upath_base | A | B ✗ | −0.25 |
| F0013-cf | B | A ✗ | +3.50 | | F0020-upath_cf | B | tie | 0.00 |
| F0013-swap_base | B | B ✓ | −4.50 | | F0020-cpath_base | A | A ✓ | +0.25 |
| F0013-swap_cf | A | B ✗ | −3.00 | | F0020-cpath_cf | B | A ✗ | +0.25 |
| F0020-base | A | B ✗ | −0.25 | | | | | |
| F0020-cf | B | tie | 0.00 | | | | | |
| F0020-swap_base | B | B ✓ | −1.75 | | | | | |
| F0020-swap_cf | A | B ✗ | −2.25 | | | | | |

**Observed pattern (descriptive, n = 3 families):**
- Within every base/cf pair the model gives the **same** choice: B/B, A/A, B/tie.
- Within every `swap` pair (letters exchanged, graph identical) it changes letter in 4/6 pairs.
- So on this set, the decision follows **surface/lexical features of the prompt** and does not track the deep transition that determines the optimum.
- `F0001-swap_*` margins of +9 to +10.5 show **high-confidence wrong answers** are possible.

**Choice-preserving counterfactuals (PROTOCOL_FREEZE §6b):**

| Branch | Pair type | Constructed | Choice invariant | Admissible (optimal on both) |
|---|---|---|---|---|
| Unchosen | `upath` (d3, d4) | 4 | 4 | **2** (`F0013-upath_base`, `F0020-upath_cf`; the latter rests on a tie) |
| Chosen | `cpath` (d3, d4) | 4 | 2 | **1** (`F0020-cpath_base`, margin +0.25) |

Choice invariance under `upath` (4/4) is **not informative** here: the model's choice does not track the deep structure even in base/cf. Admissible counts are far below the `INSUFFICIENT` threshold; no probe analysis would be meaningful.

## 6. Interpretation and limits

- **Pipeline:** feasible and deterministic. Tokenisation, capture position, shapes, transfer, oracle isolation (only prompts sent, train split only) and local scoring are all verified end to end.
- **Behaviour:** this smoke sample gives no sign of task competence. Accuracy is at chance, base/cf flips 0/3, and the choice follows the letter/surface form.
  - It is *suggestive* that the pre-registered feasibility checkpoint (depth-2 accuracy ≥ 0.70 on train+val) may fail.
  - It **does not test** that checkpoint: there are 3 families, 1 depth-2 pair, and no confidence intervals.
  - Under PROTOCOL_FREEZE §7, if the checkpoint fails, D3+ stops. No change of model, prompt or task is made without owner approval.
- **Not measured here:** competence at scale, CIs, A/B bias at scale, decoy/alias/shuffle variants, and anything about H1/H2/H3.

## 7. Cost (observed on Modal Usage & billing)

| Item | Value |
|---|---|
| "Ephemeral apps" before BTC-D2-001 | $17.08 |
| After attempt 1 | $17.10 |
| Before 001b | $17.10 |
| After 001b | $17.12 (cycle total usage shown $17.11; display lags/rounds) |
| **BTC-D2-001b** | **≈ $0.01–0.02 in credits** |
| **BTC-D2-001 + 001b cumulative** | **≈ $0.04 in credits** |
| Charged (payment method) | **$0.00** |
| Spend limit | $5 (set 2026-11-13 on owner instruction), untouched |

Granularity is $0.01 and the page may lag. The figures were read about 1 min after the run ended; a later re-check is advisable.

## 8. Artifacts

- `outputs/d2_smoke/behavior_raw.jsonl` (`cf7b8494…`), `acts_last_pos_fp16.npy` (`cfe54b2d…`), `manifest.json` (includes timing and observed cost), `scored.json`.
- Logs: `outputs/logs/d2b_run.log`, `d2b_{start,end}_utc.txt`, `d2_smoke_attempt0_client_version_error.log`, `d2_smoke_run.log` (attempt 1), `d2_smoke_remote_app_logs.txt`.
- Scripts: `src/d2_smoke.py` (as run for 001b), `outputs/logs/d2_smoke_as_run_attempt1.py`, `outputs/logs/d2_smoke_validated_33tests.py`.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently (pinned cdbee75f)
[YES] Preserved oracle isolation and train/val/test independence (prompts only, train split only)
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (BTC-D2-001, BTC-D2-001b, spend limit)
[YES] Preserved negative results and raw logs
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: timing-only instrumentation added to the validated script for 001b (requested measurements; no computational change; tests re-run, 33/33).
Next step: superseded by section A (BTC-D2-002 FAIL).
```
