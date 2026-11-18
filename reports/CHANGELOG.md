# Change log

## 2026-11-13 — BTC-D0D1-001 (CPU)
- Starter files (`generate_worlds.py`, `oracle.py`, `tests/test_oracle.py`, `modal_smoke.py`, `data/worlds.jsonl`) **unchanged**; shipped data copied to `reports/_orig/`.
- Added `src/independent_check.py` (second oracle, DFS, does not import `oracle.py`) and leakage audit.
- Added `src/build_dataset.py` (`pilot_v1`, reuses v0 prompt template) + `tests/test_dataset.py` (16 tests).
- Regenerated `pilot_v0` to `data/generated/pilot_v0.jsonl` (seed 42, 25 pairs).
- Added `src/modal_d2_smoke_DRAFT.py` — **not executed**.
- Added `requirements.txt` (stdlib only), manifest and test log in `outputs/logs/`, reports 00, 01, PROTOCOL_FREEZE (draft), results handoff.
- Fix during development: decoy reward/positions now drawn once per family.

## 2026-11-13 — D1b endpoint revision (CPU; owner request after review)
- Verified: non-optimal terminal == lowest visible reward in 300/300 pilot_v1 worlds → v0.1 D-1 withdrawn.
- Added `src/build_path_cf.py` → `data/generated/pilot_v1_pathcf.jsonl` (140 worlds: upath/cpath/xswap; choice-preserving). pilot_v1 untouched (hash re-checked `f9073bdf…`).
- Added `tests/test_path_cf.py` (10 tests; 28/28 total pass).
- D-3 (owner-approved): installed transformers 4.57.1 (no torch); downloaded Qwen3-4B-Instruct-2507 tokenizer/config files only; pinned revision `cdbee75f…`; `src/tokenizer_audit.py` → `outputs/metrics/tokenizer_audit.json` (360/360 pairs aligned; A/B = tokens 32/33).
- `src/modal_d2_smoke_DRAFT.py`: revision pinned, transformers version updated — still NOT executed.
- PROTOCOL_FREEZE → v0.2 (D-1′ proposed, D-7 added).

## 2026-11-13 — v0.3: prompt-only control + executable D2 smoke (CPU; no GPU run)
- Added `src/prompt_only_control.py` (P0, descriptive) → `outputs/metrics/prompt_only_control.json`.
- Added `src/d2_smoke.py` (Modal, NOT executed) and `src/score_d2.py` (local scorer); `tests/test_d2_pipeline.py` (4 tests, synthetic outputs in-memory only). 32/32 tests pass.
- Installed Modal client 1.2.1 locally (no login, no remote call); dry import verified.
- Removed superseded `src/modal_d2_smoke_DRAFT.py` (never executed).
- PROTOCOL_FREEZE → v0.3 (§0 evidential ladder, §4b P0, §6b reporting, §10 BTC-D2-001; D-7 approved in principle; D-8 proposed).

## 2026-11-13 — BTC-D2-001 executed (owner-authorised) — FAILED
- Spend limit set to $5 on Modal workspace on owner instruction.
- Attempt 0: Modal client 1.2.1 incompatible with image builder 2026.10 (no compute); local client upgraded to 1.6.4.
- Attempt 1: ran on L4; result undeserialisable locally (TorchVersion/torch type); no outputs. Cost ≈ $0.02 credits.
- As-run script archived: `outputs/logs/d2_smoke_as_run_attempt1.py`. Fix in `src/d2_smoke.py` (JSON/bytes-only return, exceptions → strings, progress prints) + regression test; NOT executed.
- Report: `reports/02_behavioral_baseline.md`.

## 2026-11-13 — BTC-D2-001b executed (owner-authorised) — pipeline PASS
- Timing-only instrumentation added to `src/d2_smoke.py` (sha `9cf84868…`); validated version archived `outputs/logs/d2_smoke_validated_33tests.py`; 33/33 tests.
- Run `ap-w05OyvI29OYbXTgxrn0hLn`: exit 0; outputs in `outputs/d2_smoke/`; scored locally.
- Cost ≈ $0.01–0.02 credits (cumulative ≈ $0.04), $0 charged.
- Reports updated: `02_behavioral_baseline.md`.

## 2026-11-13 — BTC-D2-002 prepared (NOT RUN)
- Added `src/d2_competence.py`, `src/score_d2_competence.py`, `tests/test_d2_competence.py` (38/38 tests).
- PROTOCOL_FREEZE §11: population, tie addendum, reporting; criterion unchanged.

## 2026-11-13 — BTC-D2-002 executed (owner-authorised) — pre-registered FAIL
- App `ap-YkjX7dMpMtv3riNTTQBV1G`; 352 train+val worlds; checkpoint 0.5625 < 0.70; D3+ stopped per §7.
- Cost ≈ $0.01–0.02 credits; $0 charged.
- Reports: `02_behavioral_baseline.md` (section A), PROTOCOL_FREEZE §11 status.

## 2026-11-13 — BTC-V2_DESIGN_REVIEW (proposal only; no GPU, no data generated)
- `reports/BTC-V2_DESIGN_REVIEW.md`: candidates Qwen3-32B / Olmo-3.1-32B-Instruct / Gemma-4-31B-it (revisions pinned), costs from modal.com/pricing, fresh-data pools, screening S1–S3 and competence C1–C5.
- CPU-only checks: HF metadata/configs, tokenizer + chat templates (tokenizer/config files only).

## 2026-11-13 — V2 data + BTC-V2-PROTOCOL_FREEZE draft (CPU only)
- `src/build_v2.py` → `data/v2/{screen,competence}{,_model_inputs}.jsonl`, `data/v2/manifest.json` (seed 2026111302). PROBE-DEV / LOCKED-TEST not generated.
- `tests/test_v2_dataset.py` (15 tests); `src/v2_token_audit.py` → `outputs/v2_audit/token_audit.json`; Qwen3-32B template saved `outputs/v2_audit/qwen3_32b_chat_template.jinja`.
- Billing calibration from `modal billing report --show-resources` (`outputs/v2_audit/*.csv`); D2 cost figures corrected (≈ $0.087 total).
- `reports/BTC-V2-PROTOCOL_FREEZE.md` drafted (S4/C3b, §5.2 template rule, §7 corrected caps and timeouts).

## 2026-11-13 — BTC-V2 run preparation (CPU only; no GPU, no weights)
- Added `src/v2_common.py`, `src/v2_remote.py`, `src/v2_local.py`, `src/v2_run_std.py`, `src/v2_run_gemma.py`, `src/score_v2.py`, `src/v2_cpu_checks.py`, `tests/test_v2_run.py` (67/67 tests total).
- CPU mirror envs: `.venv-cpucheck` (py3.11, torch 2.5.1, transformers 4.57.1, accelerate 1.10.1) and `.venv-gemma4` (py3.11, torch 2.6.0, transformers 5.23.0, hub 2.4.0); freezes in `outputs/v2_audit/`.
- Fixed: `render()` tokenizer-callable bug (caught by CPU pre-flight); Gemma hub pin corrected.
- Report + S1 request: `reports/BTC-V2-S1_READINESS.md`.

## 2026-11-13 — BTC-V2-S1 executed (owner-authorised) — SCREEN FAIL
- App `ap-JfQvAzr6gTXsVrri1n7c98`; Qwen3-32B; SCREEN 240 + 20 replays; S1 0.6125 ✗, S2 0.25 ✗, S3 ✓, S4 ✓.
- Actual cost $0.158 (billing report); no overrun. Outputs `outputs/v2/BTC-V2-S1/`; report `reports/BTC-V2-S1_RESULTS.md`. Stopped.

## 2026-11-13 — BTC-V2-S2 executed (owner-authorised) — SCREEN FAIL
- App `ap-krDl9AnSNOqObuyIA5IAUO`; Olmo-3.1-32B-Instruct; template verified pre-run (sha `49a944c9…`); S1 0.550 ✗, S2 0.10 ✗, S3 ✓, S4 ✓.
- Actual cost $0.245; no overrun. Outputs `outputs/v2/BTC-V2-S2/`; report `reports/BTC-V2-S2_RESULTS.md`. Stopped.

## 2026-11-13 — BTC-V2-S3 executed (owner-authorised) — SCREEN PASS; selection applied
- App `ap-yjwZVM5mErrWE8YuBbn29W`; Gemma-4-31B-it (tf 5.23.0 / torch 2.6.0); S1 1.000, S2 1.00, S3 1.00, S4 0.996 → PASS.
- Actual cost $0.169; no overrun. Selection rule → `gemma4_31b` (`outputs/v2/selection.json`). Report `reports/BTC-V2-S3_RESULTS.md`. Stopped; C1 not authorised.

## 2026-11-13 — BTC-V2-C1 executed (owner-authorised) — COMPETENCE PASS, D-1′ FEASIBLE
- App `ap-Fl6k865yhqxoWcvFmdD6Bs`; Gemma-4-31B-it; COMPETENCE 1,320 + 20 replays; C1–C5 all pass (C3a/C3b by one world; failures all decoy).
- Actual cost $0.284; V2 total ≈ $0.86. Report `reports/BTC-V2-C1_RESULTS.md`. Stopped; D3/D5 not launched.

## 2026-11-13 — Post-C1 design deliverables (CPU only; nothing generated or run)
- `reports/BTC-D5_DATA_DESIGN.md` (+ `src/power_d1prime.py` → `outputs/metrics/power_d1prime.json`).
- `reports/BTC-D8_CONTROL_DESIGN.md` (+ `src/d8_prototype.py` → `outputs/v2_audit/d8_prototype_audit.json`; design sample only).
- `reports/BTC-D3_ACTIVATION_DESIGN.md` (+ `src/d3_capture_check.py` → `outputs/v2_audit/d3_capture_check.json`; tiny model).
- `reports/BTC-H1_CONFIRMATORY_FREEZE_PROPOSAL.md`.

## 2026-11-13 — D5/D3/H1 revisions after owner review (CPU only)
- Extended power analysis (`outputs/metrics/power_d1prime_extended.json`); insufficiency rule (≥180 admissible pairs, ≥100 families).
- `src/sealed.py` (seal, hash-only, unlock-gated read with access log); `src/build_d5.py`; **PROBE-DEV generated** (`data/d5/probe_dev/`, sha `d26b5127…`); LOCKED-TEST generation refused (tested). `tests/test_d5_dev.py` (10).
- Candidate tokenization check: 75,400/75,400 identical (`outputs/v2_audit/d3_candidate_tokenization_check.json`); a first buggy check (72 %) was corrected.
- `src/d3_capture_core.py`, `src/d3_run_gemma.py` (NOT run; PROBE-DEV only; ephemeral Volume); `tests_gemma/test_d3_capture.py` (5, Gemma env), `tests/test_d3_run.py` (3). Main suite 80/80.
- Reports: BTC-D5_DATA_DESIGN (v0.2 section), BTC-D3_ACTIVATION_DESIGN v0.2, BTC-H1_CONFIRMATORY_FREEZE_PROPOSAL v0.2.

## 2026-11-13 — H1 v0.3 (CPU only; synthetic data)
- `src/h1_probe.py` (bilinear probe, joint assignment, B6 expectation, B2 paired permutations, family bootstrap, BudgetMeter, same-layer order decoder); `tests_cpucheck/test_h1_probe.py` (10/10, synthetic).
- `src/bench_h1_grid.py` → `outputs/metrics/bench_h1_grid.json`: v0.2 grid ≈ 194–332 CPU-h worst case (infeasible); proposed v0.3 grid 2,130 fits ≈ 9.9 CPU-h worst at 4 threads.
- `reports/BTC-H1_CONFIRMATORY_FREEZE_PROPOSAL.md` → v0.3. LOCKED-TEST not generated; no GPU; no real-activation training.

## 2026-11-13 — H1 v0.3 FROZEN; LOCKED-TEST generated and sealed (CPU only)
- Canonical candidate order (label-sorted) + canonical tie-breaking in `src/h1_probe.py`; oracle-free input builders `src/h1_features.py`; B3 conditional logit counted in global budget. `tests_cpucheck` 15/15 (permutation invariance, prompt-only candidates = oracle intermediates 4,200/4,200).
- Protocol marked FROZEN; `reports/BTC-H1_FREEZE_MANIFEST.json` (code/test/data hashes).
- LOCKED-TEST generated with approval file, sealed read-only (`SEAL_MANIFEST.json` sha `f1d541e2…`); access log absent. Independence tests: 0 overlaps. Main suite 83/83.
- `reports/BTC-D3-DEV_GPU_REQUEST.md` (not run).

## 2026-11-13 — BTC-D3-DEV executed (owner-authorised) — FAILED (fidelity check)
- Pre-launch: amendment of `src/d3_run_gemma.py` per run conditions (verify before Volume deletion, commit wait, scaledown_window=2, env/timing) recorded in freeze manifest; final H1 doc sha archived; LOCKED-TEST intact/unread.
- App `ap-xkgw5RYfrUkLZ9iWXZNCh1`: stopped on post-norm top-1 mismatch (|Δ| 0.14 ≤ 0.25, A/B decision same) — bf16 near-tie; no data retained; cost $0.170. No retry.
- Incident report `reports/BTC-D3-DEV_RESULTS.md`; correction proposed, not applied.

## 2026-11-13 — D3 technical amendment A2 (CPU only; owner-authorised)
- `check_fidelity` (max|Δ| ≤ 0.25 full vocab; identical A/B decision; top-1 equal or margin ≤ 2·max|Δ| + τ, τ = 0.125; original-tie extension pending validation); world_id in assertions; divergence log; progress every 400.
- Tests: `tests_cpucheck/test_d3_fidelity.py` (9 new; cpucheck 24/24), gemma 5/5, main 84/84. Amendment D3-A2 recorded in freeze manifest; H1 files/document unchanged; LOCKED-TEST sealed/unread.
- `reports/BTC-D3-DEV-R1_REQUEST.md` (not run).

## 2026-11-13 — BTC-D3-DEV-R1 executed (owner-authorised) — SUCCESS
- D3-A3 (non-finite rejection, owner condition 4) added pre-launch; tests 84/25/5.
- App `ap-wPTKSUy2kXQTtegmMVWRma`: 4,800 worlds + 20 bitwise replays; max |Δ| 0.195 / 0.205 (≤ 0.25); 0 non-finite; 11 original A/B ties + 4 near-tie top-1 logged; shards verified before Volume deletion and locally; 3.43 GB in `outputs/d3/probe_dev/`; GPU peak 63.35 GB; cost $0.579.
- Report `reports/BTC-D3-DEV-R1_RESULTS.md`. Stopped; no probe training.

## 2026-11-13 — BTC-H1-DEV-TRAIN prepared (CPU; no training on real activations)
- `src/h1_dev_precheck.py` → `outputs/h1_dev/precheck.json` (integrity; 75 non-decoy format violations; training-rule counts train 2,625 / val 855; 77 DEV-val admissible pairs; families disjoint; representation descriptives).
- `src/h1_train_dev.py` runner (pre-declared order, ledger, no selection on incomplete grids, budget stop, no restart without approval, final refits + hashes); B2 depth-3 fallback bug caught and fixed before use. `tests_cpucheck/test_h1_train_dev.py`; cpucheck 31/31, main 84/84; conformance 15/15.
- `reports/BTC-H1-DEV-TRAIN_REQUEST.md`.

## 2026-11-14 — BTC-H1-DEV-TRAIN executed (owner-authorised, CPU) — DEV selection complete
- Pre-launch: B2 predictors saved (9 frozen fits on all DEV), 16 final predictors asserted, DEV-train metrics + epoch-cap flag logged.
- Run: 1.475 CPU-h / 48; 2,130 bilinear + 20 logit fits; 0 fits at 60-epoch cap.
- DEV-val (exploratory): primary probe ℓ*=32, r*=8, λ*=0.1, paired acc 0.030 (chance 0.028); B3 0.156; B1/B5 0.000; anchors 0.24–0.31 with strong overfitting.
- `outputs/h1_dev/FINAL.json` verified; hashes archived; report `reports/BTC-H1-DEV-TRAIN_RESULTS.md`. LOCKED-TEST sealed, unread.

## 2026-11-14 — BTC-D3-TEST prepared (owner-authorised CPU preparation; no GPU, no unlock, no evaluation)
- `src/h1_frozen.py` + `src/h1_verify_final.py` → `outputs/h1_dev/FINAL_VERIFICATION.json`: 16 predictors hash- and structure-verified; primary u2 and B3 refits reproduce bitwise; DEV-train standardisation frozen once in `outputs/h1_dev/final/STANDARDISATION.npz` (identical transform). Predictors unchanged.
- `src/sealed_stage/sealed_access.py`: stage-scoped access on top of the unchanged `sealed.sealed_read` (D3-TEST-CAPTURE: prompts only; H1-TEST-EVAL: oracle + activations).
- `src/sealed_stage/d3_run_gemma_test.py`: LOCKED-TEST capture with the frozen capture core and D3-DEV-R1 limits; immediate sealing of activations (NOT RUN).
- `src/sealed_stage/h1_eval_test.py`: single pre-registered evaluation (NOT RUN).
- Tests: `tests_cpucheck/test_d3_test_prep.py` (10), `tests/test_d3_test_run.py` (4); cpucheck 42/42, main 88/88. Hashes in `BTC-H1_FREEZE_MANIFEST.json → prepared.BTC-D3-TEST`.
- Request `reports/BTC-D3-TEST_REQUEST.md` (open points C-1, C-2, O-2, T-1; decoy control and H2 profiles not computable with frozen predictors).

## 2026-11-14 — BTC-D3-TEST-CAPTURE attempt 1 (owner-authorised) — FAILED at container startup; no relaunch
- Pre-launch checks all PASS (`outputs/d3/d3_test_prelaunch.json`). Amendment D3-A4 (owner T-1) applied to `tests/test_d5_dev.py`: only capture-scope prompt reads allowed in the access log; log never reset.
- Capture-stage unlock written; one logged read of `model_inputs.jsonl`. Container crashed at import (`IndexError`: module-level `parents[2]` under `/root`); platform restarted startup 12×; app stopped manually after ~2 min. No activations, no seal, oracle not read. Cost $0.00 (billing report, provisional).
- CPU fix prepared (ROOT resolved locally only; preflight accepts the empty unsealed directory left by attempt 1) + container-mount import regression test. Evaluator amended for owner decisions C-1 (B2 scored vs true u2) and O-2 (status of every pre-registered secondary; Holm subset flagged). cpucheck 44/44, main 89/89.
- Report `reports/BTC-D3-TEST_CAPTURE_RESULTS.md`; request BTC-D3-TEST-R1.

## 2026-11-14 — BTC-D3-TEST-R1 executed (owner-authorised, single attempt) — SUCCESS, activations sealed
- Pre-launch PASS (`outputs/d3/d3_test_r1_prelaunch.json`): fix d7936624, frozen core, model pin, mount-path import, seal and attempt-1 traces intact. Attempt-1 unlock archived read-only; R1 unlock embeds previous approval. Startup watchdog armed (not triggered).
- Capture: 4,800/4,800 worlds, shards verified before Volume deletion, 20 bitwise replays, max|Δ| 0.1952 (pre/post), 32 divergence entries (16 worlds: 9 original A/B ties, 7 top-1 near-ties), peak 63.35 GB; 17.5 min wall. Sealed `LOCKED-TEST-D3-ACT` (40 files, `fe7c5756…`). Pool access log: 2 capture-scope prompt reads only. Cost $0.746 (attempt 1 reconfirmed $0.00).
- Evaluator self-hash now points to `prepared.BTC-H1-TEST-EVAL`. Reports `reports/BTC-D3-TEST-R1_CAPTURE_RESULTS.md`, request `reports/BTC-H1-TEST-EVAL_REQUEST.md`.

## 2026-11-14 — BTC-H1-TEST-EVAL executed once (owner-authorised, CPU) — H1 NOT MET
- Preconditions PASS (`outputs/h1_test_preconditions.json`); H1-TEST-EVAL unlocks written; single run 14:06:58–14:07:28Z.
- 4,800 worlds; 9 exact ties; 3,791 admissible; 359 decoys not evaluable; 293 depth-4 upath pairs / 200 families (sufficient).
- Paired accuracy: probe 0.020, B1 0.000, B2max 0.020, B3 0.195, B6 0.020. IUT: fails vs B2, B3 (Δ −0.174 [−0.220, −0.129]) and B6 → NOT MET; L2 not evaluated. S1/S2 Holm p = 1.0; S4 not computable; H2 not run.
- Seal test D3-A4 now fails by design (authorised analytical reads); not modified. Report `reports/BTC-H1-TEST-EVAL_RESULTS.md`.

## 2026-11-14 — BTC-H1 v0.3 CLOSED (owner decision; CPU documentation only)
- Final report `reports/BTC-H1_FINAL_REPORT.md` (methods, confirmatory NOT MET, secondaries with statuses, limits, seal status, spend ≈ $2.43).
- Archive `outputs/archive/BTC-H1-v0.3/` (66 files copied read-only + access-log snapshots; sealed activations referenced by hash). `ARCHIVE_MANIFEST.json` sha 13f453aa…; originals of the 16 predictors and TEST results set read-only.
- Historical seal test (D3-A4) kept unchanged; fails by design after the authorised analytical reads (main 88/89; cpucheck OK). Proposed replacement `tests_proposed/test_post_eval_seal.py` (authorised-reads only, append-only logs vs closure snapshot, single evaluation) — passes 4/4, not adopted.
- Follow-up design note `reports/BTC-V2-STUDY_DESIGN_NOTE.md` (explanations E1–E7, experiments X1–X7; LOCKED-TEST consumed). No GPU, no new computation on LOCKED-TEST.

## 2026-11-14 — BTC-H1b exploratory block X1 + X2 + X7 (owner-authorised, CPU, PROBE-DEV only)
- Post-evaluation seal check adopted as separate read-only suite `tests_post_eval/` (4/4); D3-A4 kept unchanged.
- `src/h1b_explore.py` (plan written before fits; inner family split for all choices; DEV-val reported only); 645 fits, 0.20 / 5 CPU-h.
- X1: scalar task variables linearly decodable (choice ≈ 0.97–1.0 from vector 40; depth 1.0; reward gap ≈ 0.93), but the H1 candidate decoder is at chance even for labels printed in the prompt (c1, A1: ≈ 0.2 vs prompt baseline 1.0) → no demonstrated sensitivity to label identity. X2: capacity/regularisation/data change train, not val (u2 ≤ 0.30 vs B3+ 0.327; MLP head degenerate). X7: confidence predicts admissibility (0.61→1.00), not probe accuracy.
- Report `reports/BTC-H1b-EXPLORATORY_RESULTS.md`; recommendation: combined X3 + X4 DEV capture (contextual-label pointer decoder + positions), GPU, needs a decision.

## 2026-11-14 — BTC-H1b X3 + X4 prepared (owner-authorised CPU preparation; no GPU)
- `src/d3x_positions.py`, `src/d3x_build_positions.py`: 22 prompt-derived positions × 3,480 PROBE-DEV training-rule worlds; alignment audit 3,480/3,480 (labels, line ends, `<channel|>`).
- `src/d3x_capture_core.py` (frozen D3 fidelity reused; tokenisation sha asserted in container), `src/h1b_x3x4_run_gemma.py` (NOT RUN), `src/h1b_x3x4_analysis.py` (representation × query position × layer factorial; key-only, layer-0, causality and B3+ controls; pre-declared sensitivity rule S-A/S-B/S-C on c1/A1).
- Tests: gemma 9/9, cpucheck 50/50, main 91/92 (historical D3-A4 only), post-eval 4/4. Hashes `outputs/d3x/plan/PREPARED.json`.
- Request `reports/BTC-H1b-X3X4_GPU_REQUEST.md` (7.4 GB, ≈ $0.8–0.95, cap $2.00).

## 2026-11-14 — BTC-H1b X3 + X4 executed (owner-authorised capture + single CPU analysis; EXPLORATORY)
- Capture: 3,480 PROBE-DEV worlds × 22 positions × 9 layers (7.41 GB); fidelity max|Δ| 0.195/0.205; 20 bitwise replays; ANS 3,480/3,480 bitwise identical to D3-DEV-R1; $0.751 (BTC cumulative ≈ $3.24; earlier total corrected to $2.49).
- Analysis (1.43 CPU-h): instrument SENSITIVE by the pre-declared rule (c1 0.958, A1 0.943, c1 − key-only 0.42), but choice + contextual keys alone gives c1 = A1 = 1.0. Static keys at chance everywhere; contextual arrival keys recover all slots (u2 up to 0.82 single / 0.69 paired vs B3+ 0.327 / 0.156). Choice + keys reaches u2 paired 0.52–0.58; the query's increment beyond the choice is not established.
- Report `reports/BTC-H1b-X3X4_RESULTS.md`. H1 v0.3 NOT MET unchanged.

## 2026-11-14 — BTC-H1b confirmatory protocol PROPOSED (CPU design only; nothing frozen, generated or run)
- `src/power_h1b.py` → `outputs/metrics/power_h1b.json` (family-level simulation; PROBE-DEV pair-per-family distribution).
- `reports/BTC-H1b_CONFIRMATORY_PROTOCOL.md`: pointer vs capacity-matched choice + keys control and within-choice query permutation (IUT), c1 instrument gate, 800 depth-4 families (power ≥ 0.86 at δ = 0.05), INSUFFICIENT < 700 families, capture ≈ $0.80 (cap $1.50; BTC cumulative ≤ $4.74), optimism risk documented. Decisions D-1 … D-7 requested.

## 2026-11-14 — BTC-H1b protocol revised to v0.2 (owner conditional validation; CPU only; nothing frozen)
- `src/h1b_control_c_dev.py` → `outputs/h1b_control_c/RESULT.json` (PROBE-DEV, family splits, 0.013 CPU-h): P u2 0.815 / paired 0.688; capacity-matched C_bil 0.502 / 0.403 (weak: rank-constrained, 2 query values); P_perm (between-family, within stratum) 0.298 / 0.075; c1: P 0.930, C_bil 0.961.
- `src/power_h1b_joint.py` → `outputs/metrics/power_h1b_joint.json`: joint IUT power ≥ 0.78 (800 fam) / ≥ 0.86 (1,000 fam) when both contrasts ≥ 0.05; ≈ 0.3–0.4 at Δ_perm = 0.025; nominal 2.5 % at Δ_perm = 0.
- Protocol v0.2: control C_max = max(C_bil, C_lin) per replicate (D-2′), partner counts (D-3), size options (D-5′), elimination control (D-8), corrected MET/NOT MET interpretations.

## 2026-11-14 — BTC-H1b v1.0 freeze prepared (owner-authorised CPU only; no pool, no GPU, no evaluation)
- `src/h1b_freeze_dev.py` (single run, 0.064 CPU-h): C_lin AdamW 0.717/0.584 vs L-BFGS 0.723/0.571 on DEV-val → L-BFGS frozen (inner-val); elimination via c2 0.883/0.818 > P 0.815/0.688 (DEV, exploratory); 11 decoders frozen (9 used), DEV-train standardisation.
- `src/power_h1b_cmax.py`: exact C_max-in-replicate rule; joint power 0.82–0.96 at both contrasts ≥ 0.05 (1,000 families, adverse scenarios incl.); 0.32–0.48 at Δ_perm = 0.025; INSUFFICIENT risk under low admissibility.
- Scripts prepared, not run: `sealed_stage/{sealed_access_h1b,build_h1b_pool,h1b_capture_test,h1b_eval_test}.py`. Tests cpucheck 57/57, main 93/94 (historical D3-A4), gemma 9/9, post-eval 4/4.
- `outputs/h1b_freeze/V1_MANIFEST.json` (sha b4248392…); protocol v1.0; `reports/BTC-H1b_V1_FREEZE_VERIFICATION.md`.

## 2026-11-14 — BTC-H1b methodological decision note (CPU reasoning only)
- Owner suspended LOCKED-TEST-H1b generation (identifiability: u2 = f(prompt structure, c2)); v1.0 freeze kept as historical prepared version.
- `reports/BTC-H1b_METHOD_DECISION_NOTE.md`: options O1 (P − elimination contrast: weak), O2 (within-upath analysis: insufficient), O3 (dead-end distractors + dead-end-swap pairs: strong; chosen branch, c2 and depth-2 set fixed within pairs), O4 (causal, out of scope); budget conflict with the $5 limit stated.

## 2026-11-14 — BTC-H1c design proposal (owner decision O3; CPU only; no pool, no GPU, no training)
- `src/build_h1c.py` (dead ends + dead-end-swap pairs; length-2 dead-end chains after an occurrence-count cue was found in the first draft), `src/h1c_design_checks.py` → `outputs/h1c_design/checks.json` (all pair invariants 400/400; deterministic 1-hop/count controls paired = 0; full-structure prompt control 1.0), `tests/test_h1c_design.py` 3/3, token alignment 12,000/12,000.
- `src/power_h1c.py` → `outputs/metrics/power_h1c.json`: 600 families × 2 late pairs → power ≥ 0.99 at δ ≥ 0.10; 0.59–0.92 at δ = 0.05 depending on admissibility.
- `reports/BTC-H1c_DESIGN_PROPOSAL.md`: hypothesis, controls (keys vs query), chance levels, competence gate, budget ≈ $4.54 expected / ≤ $4.88 worst under $5. BTC-H1b v1.0 untouched.

## 2026-11-14 — BTC-H1c protocol draft (C-2 revised; CPU only; no GPU, no pool, no training)
- Task revision 2: dead-end chain of route length (12 candidates); reward-free graph symmetric under the swap (revision 1 left a depth-3 count cue).
- `src/h1c_token_checks.py` → `outputs/h1c_design/token_checks.json`: binary candidates' arrival keys before the first differing token, same index, identical token prefix (200/200); reward-free symmetry 200/200; prompt ladder inputs identical 200/200; full-structure control exact.
- `src/power_h1c_binary.py` → `outputs/metrics/power_h1c_binary.json`: exact binary rule; type I ≤ 0.033; 600 families → ≥ 0.985 at δ = 0.10, 0.58–0.88 at δ = 0.05.
- Tests: `tests/test_h1c_design.py` 4/4, `tests_gemma/test_h1c_tokens.py` 1/1. Draft `reports/BTC-H1c_PROTOCOL_DRAFT.md`.

## 2026-11-14 — BTC-H1c DEV capture prepared (owner-authorised CPU preparation; no GPU, no TEST, no real training)
- Null calibration (`src/power_h1c_null.py`, 5,000 datasets/scenario): N1 boundary 0.0200 ± 0.0020, N4 floor-binding 0.0228 ± 0.0021, correlated/deterministic nulls 0 → calibrated. First null draft (correctness-correlated) stopped and corrected.
- H1c DEV pool (`src/build_h1c_dev.py`, 2,000 worlds, seed 2026111404, family split 300/100); positions (`src/h1c_positions.py`, 30 slots; audit 2,000/2,000; binary-key prefix identity 400/400).
- `src/h1c_dev_capture.py` (not run), `src/h1c_gates.py` (G2 blocking bitwise identity, G3 competence), `src/h1c_dev_analysis.py` (family-disjoint selection, positive controls PC1–PC3).
- Tests: cpucheck 60/60, main 98/99 (historical D3-A4), gemma 12/12 (bitwise identity of binary keys on a real tiny-model forward), post-eval 4/4.
- Request `reports/BTC-H1c_DEV_CAPTURE_REQUEST.md` (≈ $0.68, cap ≈ $0.90; BTC cumulative ≤ ≈ $4.14).

## 2026-11-14 — BTC-H1c DEV executed (owner-authorised; one capture + one CPU analysis; EXPLORATORY)
- Pre-GPU: anti-correlated null shows the old paired rule rejects 100 %; robust sign-flip rule (A1) calibrated (0.022–0.025) and frozen; orientation audit balanced.
- Capture `ap-TNhxsd…`: G1 OK 2,000/2,000; fidelity max|Δ| 0.195; 20 bitwise replays; G2 binary keys bitwise identical 400/400; G3 competence 0.723 / 0.728 → PASS. Cost $0.519 (BTC cumulative ≈ $3.76).
- Analysis (0.015 CPU-h): selection all at chance (0.46–0.56); P robust primary 0.479 (sign-flip p 0.75); controls exactly 0.5; P_perm 0.488; PC1 c1 12-way 0.399 < 0.80 → instrument insufficient; PC2 0.543; PC3 choice 0.971. Stop; no TEST. Report `reports/BTC-H1c_DEV_RESULTS.md`.

## 2026-11-14 — BTC-H1c instrument diagnostic (CPU, POST HOC; existing DEV activations; 0.027 / 1 CPU-h)
- PC1 by layer: 0.34 / 0.50 / 0.75 / 0.60 / 0.40 (layers 24–56); layer-56 selection explains most of the PC1 failure, but the best (40) is still < 0.80.
- Strong overfitting (train 0.64–1.00 vs val 0.34–0.75); PC1 learning curve still rising (218 → 0.39, 421 → 0.62, 852 → 0.75): data-limited. 2/3 of PC1 errors confuse c1 with u1 (choice binding). c1 trivially reconstructible without the query (choice × keys 0.995; text 1.000).
- Post hoc: chosen continuation decodable at layer 40 (0.666), unchosen at chance at every layer. Report `reports/BTC-H1c_INSTRUMENT_DIAGNOSTIC.md` (recommendation: close H1c under $5, or one larger DEV if the budget may be raised).

## 2026-11-14 — BTC-H1c DEV extension prepared (owner decision: single DEV extension; CPU only; no GPU, no TEST)
- `src/build_h1c_devx.py`: 600 new families × 5 = 3,000 worlds (seed 2026111406); frozen roles `outputs/h1c_devx/plan/SPLITS.json` (TRAIN 700 incl. all 400 analysed families, SELECT 100, EVAL 200 never seen).
- Positions audit 3,000/3,000, late-pair prefix identity 600/600; capture `src/h1c_devx_capture.py` (13 slots × {32,40,48}; gates G1–G3); analysis `src/h1c_devx_analysis.py` (PC1-only layer selection; stop criteria SC1 PC1 ≥ 0.80, SC2 |SELECT−EVAL| ≤ 0.10, SC3 chosen continuation p < 0.025).
- Tests: cpucheck 62/62, main 99/100 (historical D3-A4), gemma 14/14, post-eval 4/4. Request `reports/BTC-H1c_DEVX_CAPTURE_REQUEST.md` (≈ $0.76, cap ≈ $0.99, cumulative ≤ $4.75; $6 cap proposed only for an eventual TEST).

## 2026-11-14 — BTC-H1c-DEVX executed (owner-authorised; one capture + one CPU analysis; EXPLORATORY)
- Capture `ap-2tmsd0…`: G1 3,000/3,000; fidelity max|Δ| 0.195; 20 bitwise replays; G2 600/600; G3 0.727 / 0.749 → PASS. Cost $0.474 (BTC cumulative ≈ $4.24; cap raised to $6 by the owner, not a TEST authorisation).
- Analysis (0.043 CPU-h): layer 40 chosen from PC1 on SELECT (0.909); EVAL PC1 0.886 [0.855, 0.915], SC2 gap 0.023, PC2 chosen continuation 0.756 [0.720, 0.795] p < 1e-4 → INSTRUMENT ADEQUATE. Unchosen continuation: late pairs 0.521 (sign-flip p 0.18; CI [0.480, 0.557]); early pairs 0.703 (p < 1e-4; keys can carry it there; controls not pre-declared).
- Report + confirmatory proposal (separate decision) `reports/BTC-H1c_DEVX_RESULTS.md`.

## 2026-11-14 — BTC-H1c final diagnostic (CPU, POST HOC; existing DEV + DEVX; 0.016 / 1 CPU-h)
- Key identity: late unchosen 141/141, early unchosen 0/118, cswap chosen 26/141.
- Early pairs: P 0.703 [0.648, 0.758] vs key-only 0.445, choice × keys 0.487, μ_choice 0.517 → P − best control +0.186 [0.110, 0.263] (query–key interaction; not query-alone).
- cswap chosen: PC2 0.770 (all pairs); identical-key subset (26 families) 0.712 [0.596, 0.827] with controls exactly 0.5. Chosen − unchosen (late-comparable) +0.26 [0.12, 0.40] on 25 families: not a demonstrated asymmetry (populations, size, post hoc).
- Recommendation: close H1c on its original question; a future asymmetry study would be a new, post hoc-generated hypothesis. Report `reports/BTC-H1c_FINAL_DIAGNOSTIC.md`.

## 2026-11-14 — BTC-H1c CLOSED; H1 line of work consolidated (owner decision; CPU documentation only)
- H1c status: EXPLORATORY COMPLETE — UNCHOSEN CONTINUATION NOT DETECTED (not refuted). H1 v0.3 NOT MET and all H1b/H1c results kept unchanged.
- Archive `outputs/archive/BTC-H1b-H1c/ARCHIVE_MANIFEST.json` (165 files read-only; tensors/pools by hash; sha 4724d316…).
- Consolidated report `reports/BTC_H1_RESEARCH_CONSOLIDATION.md` (chronology, results, corrections, limits, contributions, costs ≈ $4.235). Chosen/unchosen asymmetry recorded as a new post hoc hypothesis, not designed.
