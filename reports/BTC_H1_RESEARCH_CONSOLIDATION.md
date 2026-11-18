# Before the Choice — consolidated research report: H1 → H1b → H1c

**Date:** 2026-11-14 · **Owner decision:** "CLÔTURE BTC-H1c" (CPU-only documentation) · **Status of the H1 line of work:** CLOSED.

| Study | Final status (owner) |
|---|---|
| **H1 v0.3** | **NOT MET** (confirmatory, LOCKED-TEST; historical, unchanged) |
| **H1b** | exploratory; v1.0 confirmatory freeze prepared, then **suspended** for an identifiability problem; historical |
| **H1c** | **EXPLORATORY COMPLETE — UNCHOSEN CONTINUATION NOT DETECTED.** *Not refuted.* |

No result below was rewritten; each sub-report remains the reference for its stage. No forward-sweep, causal (H3) or dynamics (H2) claim is made anywhere.

---

## 1. Question

- **Primary question (H1):** do pre-answer activations of an LLM encode information about the path of the action it will **not** choose, beyond prompt-text and answer-based baselines?
- **Framing:** a candidate functional analogue of hippocampal forward sweep.
- **Model:** Gemma-4-31B-it @ `842da379…`. It was chosen after pre-registered competence screens; Qwen3-4B, Qwen3-32B and Olmo-3.1-32B failed them.

## 2. Chronology and results

### 2.1 H1 v0.3 — confirmatory (frozen; `BTC-H1_FINAL_REPORT.md`)

**Design:**
- Static candidate embeddings, low-rank bilinear probe at `<channel|>`, layer selected on PROBE-DEV.
- Intersection-union test against B1 / B2 / B3 / B6 on LOCKED-TEST: 293 admissible depth-4 pairs, 200 families.

**Result: NOT MET.**

| Decoder | Paired accuracy |
|---|---|
| probe | 0.020 [0.007, 0.038] |
| B3 (prompt text) | 0.195 |
| B2 max | 0.020 |
| B6 | 0.020 |

- The probe minus B3 was −0.174 [−0.220, −0.129].
- LOCKED-TEST is consumed.

### 2.2 H1b — understanding the negative result (exploratory; PROBE-DEV)

**X1, X2, X7** (`BTC-H1b-EXPLORATORY_RESULTS.md`):
- The H1 decoder could not read even labels printed verbatim in the prompt (c1, A1 ≈ 0.2 vs prompt 1.0).
- Yet linear probes read scalar task variables (choice ≈ 1.0, reward gap ≈ 0.93).
- So the H1 instrument had **no demonstrated sensitivity to label identity**.

**X3, X4** (`BTC-H1b-X3X4_RESULTS.md`): contextual label keys (the label's own arrival state) fix the instrument.

| Measure | Value |
|---|---|
| c1 | 0.96 |
| u2 | up to 0.82 single / 0.69 paired |
| choice + keys control | u2 paired 0.52–0.58 |

The label representations built while reading already encode branch and depth.

**Controls and elimination** (`BTC-H1b_CONFIRMATORY_PROTOCOL.md`, `BTC-H1b_V1_FREEZE_VERIFICATION.md`):
- The capacity-matched control was weak (0.40 paired). The c2-elimination control (0.818 paired) **beat the pointer** (0.688).

**Identifiability** (`BTC-H1b_METHOD_DECISION_NOTE.md`):
- With two depth-2 candidates, u2 = f(prompt structure, c2).
- No accuracy contrast on that task can separate unchosen-branch from chosen-branch information.
- The v1.0 freeze was kept as a historical archive; generation of LOCKED-TEST-H1b was **suspended**.

### 2.3 H1c — dead ends and dead-end-swap pairs (exploratory)

**Design** (`BTC-H1c_DESIGN_PROPOSAL.md`, `BTC-H1c_PROTOCOL_DRAFT.md`):
- Each depth-1 state has a continuing successor and a dead-end chain of the same length.
- Each pair swaps which unchosen depth-2 state continues. Within a pair, the chosen branch, c2, the candidate set, the label counts and the reward-free graph are identical.
- In **late** pairs, the two binary candidates' arrival keys are computed **before** the modified lines. They are therefore bitwise identical within the pair: verified at token level, and on the captured vectors 400/400 (DEV) and 600/600 (DEVX).

**Primary rule (A1):** single-world binary accuracy on late-pair worlds vs 0.5, with a sign-flip randomisation test on family means. It replaced a paired threshold of 0.25, which proved invalid (§4).

**Results:**

| Stage | Key numbers | Report |
|---|---|---|
| **DEV** (2,000 worlds, $0.519) | gates passed (competence 0.723); unchosen late **0.479** (p = 0.75); PC1 **0.399 < 0.80**, so the instrument was **insufficient** | `BTC-H1c_DEV_RESULTS.md` |
| **Instrument diagnostic** (post hoc) | PC1 0.75 at layer 40 vs 0.40 at the selected layer 56; still-rising learning curve (data-limited); 2/3 of errors c1 → u1 | `BTC-H1c_INSTRUMENT_DIAGNOSTIC.md` |
| **DEVX** (+3,000 worlds, $0.474; frozen TRAIN / SELECT / EVAL families; layer chosen from PC1 only) | **INSTRUMENT ADEQUATE**: PC1 **0.886** [0.855, 0.915]; chosen continuation **0.756** [0.720, 0.795]. **Unchosen late 0.521** (p = 0.18, CI [0.480, 0.557]); unchosen early 0.703 | `BTC-H1c_DEVX_RESULTS.md` |
| **Final diagnostic** (post hoc) | early-pair signal needs the query (query-free controls at chance; P − best +0.19 [0.11, 0.26]); **chosen continuation from the query alone** on identical-key cswap pairs 0.712 [0.60, 0.83] (26 families); chosen − unchosen +0.26 [0.12, 0.40] on 25 families (**not a demonstrated asymmetry**) | `BTC-H1c_FINAL_DIAGNOSTIC.md` |

## 3. Positive, negative and post hoc results

| Type | Result |
|---|---|
| **Confirmatory, negative** | H1 v0.3 NOT MET (static-key probe at the pre-answer position) |
| **Exploratory, positive** | contextual label keys make label identity and branch structure decodable (H1b X3/X4); the H1c instrument is adequate on never-seen families (PC1 0.89, chosen continuation 0.76) |
| **Exploratory, negative** | H1c: **unchosen continuation not detected** from the pre-answer query when the keys are blind (0.521, CI upper bound 0.557) |
| **Post hoc** | early-pair unchosen signal requires the query–key interaction; chosen continuation is decodable from the query alone (small subset); the chosen/unchosen asymmetry is suggestive |
| **New hypothesis (post hoc, not tested confirmatorily; no design started)** | *The pre-answer state carries usable information about the chosen continuation but not about the unchosen continuation, under strictly comparable conditions.* Testing it would need late order on both branches and a new sealed TEST |

## 4. Methodological corrections made along the way (all documented, none hidden)

| Issue | Correction |
|---|---|
| D3-DEV strict top-1 fidelity aborted on a bf16 near-tie | amendment D3-A2/A3 (tolerance τ; non-finite rejection) |
| D3-TEST attempt 1: container start-up crash (path computed at module import) | fixed; mount-path regression test; watchdog; $0 |
| H1 decoder used static keys; no sensitivity check | H1b positive controls (c1, A1) → contextual keys |
| B2 scoring ambiguity in H1b | owner C-1: scored against the true u2 |
| H1b v1.0 identifiability (elimination beats pointer) | suspension; H1c design |
| H1c first designs: occurrence-count cue at depth 2, then at depth 3 | dead-end chain lengthened to the route length; reward-free symmetry verified |
| Null model of the power simulation correlated *correctness* instead of *preference* | stopped and rewritten before use |
| **Paired 0.25 threshold invalid under anti-correlated predictions** (100 % false rejections in simulation) | amendment A1: sign-flip rule (calibrated 0.022–0.025); orientation audit |
| PC1 pre-declared at the binary-selected layer (noise-driven) | DEVX: layer chosen from PC1 on SELECT; frozen three-way family roles |
| Orientation-audit artefact (token-length ties) | corrected definition (ties excluded) |
| Spend accounting (an unrelated app in the workspace; $2.43 vs $2.49) | billing-based figures per app |

## 5. Limits

1. **One model, one family of tasks, one decoder family** (low-rank bilinear pointer), and positions at or around `<channel|>`.
2. **Backward inference from rewards (P0) is never excluded.** Decodability says nothing about the direction or sequence of construction.
3. The H1c template ("you can move to", dead ends) differs from H1 / H1b, so the studies are not directly comparable.
4. **Competence** with dead ends is moderate (≈ 0.72–0.73); about 27 % of worlds are not admissible.
5. The H1c conclusions are **DEV-based** (EVAL families never seen, but no sealed TEST).
6. The chosen-side comparison rests on 26 families selected by an uncontrolled line-order property.
7. **Not tested:** H2 (dynamics), H3 (causal use), decoy control (H1), layer profiles.

## 6. Methodological contributions (what this line of work established as practice)

- **Sealed, stage-scoped access** (prompts-only capture vs analytical unlock), single-evaluation markers, append-only access logs, and a post-evaluation seal check.
- **Instrument sensitivity before inference.** Positive controls (c1, A1, chosen continuation) are pre-declared with thresholds. A negative result counts only if the instrument passes them.
- **Separating decodability, key-borne information and query-specific information.** Contextual keys, key-only and choice + keys controls, and capacity-matched controls.
- **Minimal-pair designs with bitwise-identical inputs.** Controls are zero *by construction*, verified at token level and on the captured activations (gate G2).
- **Robust inference under dependent predictions.** Orientation randomisation plus a sign-flip test on family means. We showed by simulation that the naïve paired threshold fails.
- **Reproducibility:** frozen hashes for every protocol, decoder and dataset; byte-exact git history with EOL conversion disabled; bitwise replays; identity of ANS vectors across captures (3,480/3,480).

## 7. Costs

| App (Modal billing) | $ |
|---|---|
| V2 screens and competence (4 apps) | 0.920 |
| D2 smoke and competence (Qwen, L4; 3 apps) | 0.077 |
| D3-DEV (failed attempt + R1) | 0.170 + 0.579 |
| D3-TEST (attempt 1 + R1) | 0.000 + 0.746 |
| H1b X3/X4 capture | 0.751 |
| H1c DEV capture | 0.519 |
| H1c DEVX capture | 0.474 |
| **Total BTC** | **≈ $4.235** (cap $6; earlier cap $5) |

CPU: every analysis stayed far under its cap. The largest was H1-DEV-TRAIN at 1.5 CPU-h.

## 8. Archives

| Archive | Contents |
|---|---|
| `outputs/archive/BTC-H1-v0.3/ARCHIVE_MANIFEST.json` | H1 v0.3 |
| `outputs/archive/BTC-H1b-H1c/ARCHIVE_MANIFEST.json` (sha `4724d316…`) | 165 files copied read-only: H1b and H1c reports, protocols, plans, gates, run manifests, analyses, power files, logs, scripts, tests. Activation tensors and pools referenced by hash |
| `reports/BTC-H1_FREEZE_MANIFEST.json`, `outputs/h1b_freeze/V1_MANIFEST.json`, `outputs/h1c_dev/plan/PREPARED.json`, `outputs/h1c_devx/plan/PREPARED.json` | the frozen hashes of each stage |

## 9. Bottom line

- Under the frozen H1 protocol, unchosen-path information was **not** found (NOT MET).
- The H1b diagnostics showed that this was largely an **instrument** failure.
- With a validated instrument and a design that removes chosen-branch and reading-time explanations (H1c), the pre-answer query **did not show detectable information about the unchosen continuation**, while it did for the chosen one in a small post hoc subset.
- The original hypothesis is therefore **not supported and not refuted**.
- The chosen/unchosen asymmetry is recorded as a **new, post hoc hypothesis** for possible future work.

```text
SCOPE AUDIT
[YES] Stayed within symbolic two-branch prospective-trajectory task
[YES] Kept H1 primary and H2/H3 conditional (H2/H3 never tested)
[YES] Did not study J-space, sentience, AI welfare, VTE as new targets
[YES] Did not train/fine-tune the base LLM or change checkpoints silently
[YES] Preserved oracle isolation and train/val/test independence throughout
[YES] Did not equate token availability with prospective path representation
[YES] Did not equate transformer layers with biological time
[YES] Obtained explicit owner approval for each billable/major change (consolidation CPU only)
[YES] Preserved negative results and raw logs (all historical results unchanged; archives read-only)
[YES] Labeled unrun stages NOT RUN and limitations honestly
Any deviation: none.
Next step: HORS PÉRIMÈTRE OU AUTORISATION REQUISE — none pending; H1 line of work closed.
```
