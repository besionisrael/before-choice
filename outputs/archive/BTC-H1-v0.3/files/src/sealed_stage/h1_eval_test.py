"""BTC-H1-TEST-EVAL — the SINGLE pre-registered H1 evaluation on LOCKED-TEST (H1 v0.3 §2, §5, §6, §8). PREPARED, NOT RUN.
Requires (i) a completed, sealed BTC-D3-TEST capture and (ii) the owner's analytical unlock for stage H1-TEST-EVAL.

    python src/sealed_stage/h1_eval_test.py --run        (CPU; cpucheck environment)

Order of operations (nothing is tuned, refit or selected here):
  1. refuse if an evaluation was ever started (EVAL_STARTED.json is written BEFORE any sealed read; a crash therefore
     cannot be followed by a silent second attempt);
  2. verify the 16 frozen predictors (FINAL.json + freeze manifest), the frozen standardisation file, the frozen code,
     and both seals (pool + activations) with hash_only;
  3. read activations, prompts and oracle ONLY through sealed_access.staged_read(stage="H1-TEST-EVAL");
  4. admissibility: the model's forced A/B choice (logp_A vs logp_B; exact tie = not admissible) equals the oracle optimum;
  5. primary: admissible depth-4 upath pairs (base->upath_base, cf->upath_cf); INSUFFICIENT if < 180 pairs or < 100
     contributing families (descriptive only);
  6. H1 MET iff the 95 % family-cluster percentile bootstrap (10,000 resamples, seed 0) lower bound of
     probe - B_k > 0 for every k in {B1, B2 (max over the 3 permutations INSIDE each replicate), B3, B6};
     B2 predictors were trained on permuted targets and are SCORED against the TRUE u2 target (owner decision C-1);
  7. L2 attribution only if MET: (a) same-layer order test vs max(1/6, B3-order), (b) alias, (c) shuffle generalisation;
  8. reported, not thresholds: B5, B7 (cpath pairs), B4 (= 0 by construction), per-head accuracies, secondaries.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "sealed_stage"))
import h1_features as F  # noqa: E402
import h1_frozen as Z  # noqa: E402
import h1_probe as H  # noqa: E402
import sealed  # noqa: E402
import sealed_access as SA  # noqa: E402
from d3_capture_core import u16_to_bf16_f32  # noqa: E402

STAGE = "H1-TEST-EVAL"
MIN_PAIRS, MIN_FAMILIES = 180, 100     # H1 v0.3 §6.1
B_RESAMPLES, BOOT_SEED = 10_000, 0     # H1 v0.3 §6.2
CHANCE_ORDER = 1 / 6                   # §6.3
FROZEN_CODE = ("src/h1_probe.py", "src/h1_features.py", "src/sealed.py", "src/d3_capture_core.py")


# ------------------------------------------------------------------ sealed inputs
def _read(stage, d, rel, purpose, unlock_root):
    return SA.staged_read(stage, d, rel, purpose, unlock_root)


def load_test(pool_dir: Path, act_dir: Path, unlock_root: Path, layers: list[int]):
    rd = lambda d, rel, why: _read(STAGE, d, rel, why, unlock_root)
    man = json.loads(rd(act_dir, "MANIFEST.json", "activation manifest"))
    labels = json.loads(rd(act_dir, "candidate_labels.json", "candidate labels"))["labels"]
    T_raw = np.load(io.BytesIO(rd(act_dir, "candidate_embeddings.npy", "candidate embeddings")))
    idx, Zl, mean_emb = [], {ell: [] for ell in layers}, []
    for s in man["shards"]:
        idx += [json.loads(x) for x in rd(act_dir, f"{s['name']}_index.jsonl", "index").decode("utf-8").splitlines()]
        a = np.load(io.BytesIO(rd(act_dir, f"{s['name']}_acts.npy", "activations")))
        for ell in layers:
            Zl[ell].append(u16_to_bf16_f32(a[:, ell]))
        mean_emb.append(np.load(io.BytesIO(rd(act_dir, f"{s['name']}_mean_emb.npy", "mean_emb"))))
        del a
    Zl = {ell: np.concatenate(v).astype(np.float32) for ell, v in Zl.items()}
    inputs = {r["world_id"]: r for r in map(json.loads, rd(pool_dir, "model_inputs.jsonl", "prompts").decode("utf-8").splitlines())}
    worlds = {r["world_id"]: r for r in map(json.loads, rd(pool_dir, "worlds.jsonl", "oracle labels").decode("utf-8").splitlines())}
    return idx, Zl, np.concatenate(mean_emb).astype(np.float32), labels, T_raw, inputs, worlds


# ------------------------------------------------------------------ evaluation core (also used by synthetic tests)
def build_rows(idx, inputs, worlds):
    rows, audit = [], defaultdict(int)
    for g, r in enumerate(idx):
        w = worlds[r["world_id"]]
        ch = "A" if r["logp_A"] > r["logp_B"] else "B" if r["logp_A"] < r["logp_B"] else "TIE"
        adm = ch != "TIE" and ch == w["oracle"]["optimal_action"]
        audit["n_worlds"] += 1
        audit["n_ties"] += ch == "TIE"
        audit["n_admissible"] += adm
        if not adm:
            continue
        prompt = inputs[r["world_id"]]["prompt"]
        cands = F.canonical_candidates(F.candidates_from_prompt(prompt))
        sup = F.supervision_targets(w, ch)
        if len(cands) > H.K or not all(l in cands for l in sup.values()):
            audit[f"not_evaluable_K{len(cands)}_{w['variant'].split('_')[0]}"] += 1   # decoys: 10 / 7 candidates
            continue
        rows.append({"g": g, "world_id": r["world_id"], "family_id": w["family_id"], "variant": w["variant"],
                     "depth": w["depth"], "choice": ch, "prompt": prompt, "cands": cands,
                     "tgt": {s: cands.index(l) for s, l in sup.items()}})
    return rows, dict(audit)


def score_all(rows, P, S, Zl, mean_emb, labels, T_raw):
    """Per-world probability tables of every frozen predictor (canonical candidate order)."""
    li = {l: i for i, l in enumerate(labels)}
    cand = torch.full((len(rows), H.K), H.PAD, dtype=torch.long)
    for k, r in enumerate(rows):
        for j, l in enumerate(r["cands"]):
            cand[k, j] = li[l]
    T = Z.standardise_with(T_raw, S["T_mu"], S["T_sd"])
    g = [r["g"] for r in rows]
    X = torch.zeros(len(rows), H.K, len(F.B3_FEATURES))
    M = torch.ones(len(rows), H.K, dtype=torch.bool)
    for k, r in enumerate(rows):
        X[k, :len(r["cands"])] = torch.from_numpy(F.b3_features(r["prompt"], r["cands"], r["choice"]))
        M[k, :len(r["cands"])] = False
    probs = {}
    for name, p in P.items():
        if p["kind"] == "logit":
            probs[name] = Z.apply_logit(p, X, M)
            continue
        if p["input"] == "mean_emb":
            z = Z.standardise_with(mean_emb[g], S["mean_emb_mu"], S["mean_emb_sd"])
        elif p["input"] == "none":
            z = torch.ones(len(rows), 1)
        else:
            z = Z.standardise_with(Zl[p["input"]][g], S[f"z{p['input']}_mu"], S[f"z{p['input']}_sd"])
        probs[name] = Z.apply_bilinear(p, z, cand, T)
    return probs, cand


def world_scores(rows, probs, cand):
    """Per-world success (0/1, or B6 expectation) for the u2 target, plus order / B3-order exact matches."""
    am = {n: H.argmax_canonical(p).numpy() for n, p in probs.items()}
    nv = (cand != H.PAD).sum(-1).numpy()
    out = []
    for k, r in enumerate(rows):
        t = r["tgt"]
        rec = {}
        if "u2" in t:
            for n in ("primary_u2", "B1", "B5", "B3"):
                rec[n] = float(am[n][k] == t["u2"])
            for kp in range(3):
                # Owner decision C-1 (2026-11-14): B2 is TRAINED on permuted targets but SCORED against the TRUE u2
                # target, on the same admissible pairs as H1; max over k is taken inside each bootstrap replicate.
                rec[f"B2_perm{kp}"] = float(am[f"B2_perm{kp}"][k] == t["u2"])
        if all(s in t for s in H.SLOTS):          # depth 4: 6 candidates
            alp = np.stack([torch.log(probs[f"anchor_{h}"][k]).numpy() for h in Z.ANCHORS])
            rec["B6"] = H.b6_expectation(alp, t["u2"], int(nv[k]))
            olp = np.stack([torch.log(probs[n][k]).numpy() for n in ("order_u1", "primary_u2", "order_u3")])
            blp = np.stack([torch.log(probs[n][k]).numpy() for n in ("B3o_u1", "B3o_u2", "B3o_u3")])
            true3 = (t["u1"], t["u2"], t["u3"])
            rec["order"] = H.order_correct(olp, true3, int(nv[k]))
            rec["B3o"] = H.order_correct(blp, true3, int(nv[k]))
            for h, n in (("u1", "order_u1"), ("u3", "order_u3"), ("c1", "anchor_c1"), ("c3", "anchor_c3"),
                         ("u1", "anchor_u1"), ("u3", "anchor_u3")):
                rec[f"head_{n}"] = float(am[n][k] == t[h])
        out.append(rec)
    return out


def pairs_of(rows, depth, kind):
    """Admissible pairs (both members evaluable) of a variant kind ('upath', 'cpath', 'xswap'), per family."""
    pos = {r["world_id"]: k for k, r in enumerate(rows)}
    by_f = defaultdict(list)
    for f in sorted({r["family_id"] for r in rows if r["depth"] in depth}):
        for src in ("base", "cf"):
            a, b = f"{f}-{src}", f"{f}-{kind}_{src}"
            if a in pos and b in pos and rows[pos[a]]["depth"] in depth:
                by_f[f].append((pos[a], pos[b]))
    return dict(by_f)


def _boot(by_f: dict, stat):
    return H.family_bootstrap(by_f, B=B_RESAMPLES, seed=BOOT_SEED, stat=stat)


def paired_values(ws, by_f, name):
    return {f: [ws[a][name] * ws[b][name] for a, b in ps] for f, ps in by_f.items()}


def mean_over(fams, vals):
    xs = [v for f in fams for v in vals[f]]
    return float(np.mean(xs)) if xs else float("nan")


def primary(rows, ws, min_pairs=MIN_PAIRS, min_fams=MIN_FAMILIES):
    by_f = pairs_of(rows, {4}, "upath")
    n_pairs, n_fams = sum(len(v) for v in by_f.values()), len(by_f)
    names = ["primary_u2", "B1", "B3", "B5", "B6", "B2_perm0", "B2_perm1", "B2_perm2"]
    V = {n: paired_values(ws, by_f, n) for n in names}
    fams = sorted(by_f)
    point = {n: mean_over(fams, V[n]) for n in names}
    point["B2_max"] = max(point[f"B2_perm{k}"] for k in range(3))
    res = {"population": "admissible depth-4 upath pairs", "n_pairs": n_pairs, "n_families": n_fams,
           "chance": 1 / 36, "point": point, "B4": {"value": 0.0, "note": "0 by construction on D-1' (v0.3 §5); not computed"}}
    if n_pairs < min_pairs or n_fams < min_fams:
        res["outcome"] = "INSUFFICIENT"
        res["rule"] = f"n_pairs {n_pairs} < {min_pairs} or n_families {n_fams} < {min_fams} (v0.3 §6.1): descriptive only"
        return res, by_f
    ci = {"primary_u2": _boot(by_f, lambda pick: mean_over(pick, V["primary_u2"]))}
    deltas = {}
    for k in ("B1", "B3", "B6"):
        deltas[k] = _boot(by_f, lambda pick, k=k: mean_over(pick, V["primary_u2"]) - mean_over(pick, V[k]))
    deltas["B2"] = _boot(by_f, lambda pick: mean_over(pick, V["primary_u2"])
                         - max(mean_over(pick, V[f"B2_perm{k}"]) for k in range(3)))
    res["ci95"] = ci
    res["delta_ci95"] = {k: {"point": point["primary_u2"] - (point["B2_max"] if k == "B2" else point[k]),
                             "lower": lo, "upper": hi, "lower_gt_0": lo > 0} for k, (lo, hi) in deltas.items()}
    res["outcome"] = "MET" if all(v["lower_gt_0"] for v in res["delta_ci95"].values()) else "NOT MET"
    return res, by_f


def single_world_test(rows, ws, select, name, ref, floor):
    """Single-world statistic over admissible worlds chosen by `select`: mean(name) - max(floor, mean(ref)), families resampled."""
    by_f = defaultdict(list)
    for k, r in enumerate(rows):
        if select(r) and name in ws[k]:
            by_f[r["family_id"]].append(k)
    by_f = dict(by_f)
    vals = lambda n: {f: [ws[k][n] for k in ks] for f, ks in by_f.items()}
    Vn, Vr = vals(name), vals(ref)
    fams = sorted(by_f)
    point = mean_over(fams, Vn) - max(floor, mean_over(fams, Vr))
    lo, hi = _boot(by_f, lambda pick: mean_over(pick, Vn) - max(floor, mean_over(pick, Vr))) if by_f else (float("nan"),) * 2
    return {"n_worlds": sum(len(v) for v in by_f.values()), "n_families": len(by_f), name: mean_over(fams, Vn),
            ref: mean_over(fams, Vr), "floor": floor, "point": point, "lower": lo, "upper": hi, "lower_gt_0": bool(lo > 0)}


def l2(rows, ws):
    d4 = lambda r, kinds: r["depth"] == 4 and r["variant"] in kinds
    a = single_world_test(rows, ws, lambda r: d4(r, ("base", "cf")), "order", "B3o", CHANCE_ORDER)
    b = single_world_test(rows, ws, lambda r: d4(r, ("alias_base", "alias_cf")), "primary_u2", "B3", 1 / 6)
    c = single_world_test(rows, ws, lambda r: d4(r, ("shuffle_base", "shuffle_cf")), "primary_u2", "B3", 1 / 6)
    out = {"a_order_same_layer": a, "b_alias": b, "c_shuffle": c}
    failing = [k for k, v in out.items() if not v["lower_gt_0"]]
    out["outcome"] = "MET + L2" if not failing else "MET, L2 not attributed"
    out["failing"] = failing
    return out


def secondaries(rows, ws):
    """Reported; Holm over the tested secondaries (operationalisation submitted with BTC-D3-TEST, item O-2)."""
    out = {}
    tests = {}
    for name, depth, kind in (("S1_pooled_depth3_4_upath", {3, 4}, "upath"), ("S2_xswap_depth4", {4}, "xswap")):
        by_f = pairs_of(rows, depth, kind)
        Vp, Vb = paired_values(ws, by_f, "primary_u2"), paired_values(ws, by_f, "B3")
        fams = sorted(by_f)
        reps = []
        if by_f:
            rng = np.random.default_rng(BOOT_SEED)
            for _ in range(B_RESAMPLES):
                pick = [fams[i] for i in rng.integers(0, len(fams), len(fams))]
                reps.append(mean_over(pick, Vp) - mean_over(pick, Vb))
        p = float(np.mean(np.array(reps) <= 0)) if reps else float("nan")
        out[name] = {"n_pairs": sum(len(v) for v in by_f.values()), "n_families": len(by_f), "probe": mean_over(fams, Vp),
                     "B3": mean_over(fams, Vb), "delta": mean_over(fams, Vp) - mean_over(fams, Vb), "p_one_sided_boot": p}
        tests[name] = p
    order = sorted((v, k) for k, v in tests.items() if v == v)
    m = len(order)
    running = 0.0
    for i, (p, k) in enumerate(order):
        running = max(running, min(1.0, (m - i) * p))
        out[k]["p_holm"] = running
    by_f = pairs_of(rows, {4}, "cpath")
    out["B7_cpath_depth4"] = {"n_pairs": sum(len(v) for v in by_f.values()),
                              "probe_paired_acc": mean_over(sorted(by_f), paired_values(ws, by_f, "primary_u2")),
                              "note": "reference, not a threshold (v0.3 §2)"}
    d4 = [k for k, r in enumerate(rows) if r["depth"] == 4 and r["variant"] in ("base", "cf")]
    out["S3_per_depth_heads_depth4_base_cf"] = {n: float(np.mean([ws[k][n] for k in d4])) if d4 else float("nan")
                                                 for n in ("head_order_u1", "primary_u2", "head_order_u3", "head_anchor_c1",
                                                           "head_anchor_c3", "head_anchor_u1", "head_anchor_u3")}
    out["S3_per_depth_heads_depth4_base_cf"]["chance"] = 1 / 6
    out["S4_decoy"] = {"status": "NOT COMPUTABLE with the frozen predictors",
                       "reason": "decoy worlds expose 10 (depth 4) / 7 (depth 3) prompt candidates; the frozen decoders are "
                                 "6-candidate softmaxes and restricting candidates would need the oracle"}
    out["H2_layer_profiles"] = {"status": "NOT RUN", "reason": "frozen predictors exist at one vector index per head; a layer "
                                "profile would need new per-layer DEV training (not authorised)"}
    out["plan_status"] = PLAN_STATUS
    out["holm_family"] = {"members": [k for _, k in order], "m": m,
                          "note": "Holm covers the computable INFERENTIAL secondaries only; it is a subset of the "
                                  "pre-registered secondary plan (v0.3 §8), not the whole plan: see plan_status"}
    return out


# Owner decision O-2 (2026-11-14): status of EVERY pre-registered secondary analysis (H1 v0.3 §8 and §2), kept visible.
PLAN_STATUS = [
    {"analysis": "depth 3+4 pooled (upath pairs)", "id": "S1", "status": "INFERENTIAL (Holm)",
     "form": "probe - B3 paired accuracy; one-sided family bootstrap p (10,000, seed 0)"},
    {"analysis": "xswap (depth 4)", "id": "S2", "status": "INFERENTIAL (Holm)",
     "form": "probe - B3 paired accuracy on (base,xswap_base),(cf,xswap_cf); one-sided family bootstrap p"},
    {"analysis": "per-depth reconstruction", "id": "S3", "status": "DESCRIPTIVE",
     "form": "single-world accuracy of the frozen u1 / u2 / u3 (and anchor) heads on admissible depth-4 base/cf worlds vs 1/6",
     "reason": "kept descriptive as approved in O-2; no new test added after the freeze"},
    {"analysis": "decoy control", "id": "S4", "status": "NOT COMPUTABLE",
     "reason": "decoy worlds expose 10 (depth 4) / 7 (depth 3) prompt candidates; the frozen decoders are 6-way softmaxes; "
               "restricting candidates would require the oracle"},
    {"analysis": "H2 descriptive layer profiles", "id": "H2", "status": "NOT RUN",
     "reason": "each frozen head exists at a single vector index; a profile needs new per-layer DEV training (not authorised)"},
    {"analysis": "chosen-branch reference B7 (cpath pairs)", "id": "B7", "status": "DESCRIPTIVE (reference, v0.3 §2)"},
    {"analysis": "answer-only B4", "id": "B4", "status": "NOT COMPUTED",
     "reason": "0 by construction on D-1' (the answer and visible rewards do not determine the unchosen depth-2 label), v0.3 §5"},
]


def evaluate(rows, audit, P, S, Zl, mean_emb, labels, T_raw, min_pairs=MIN_PAIRS, min_fams=MIN_FAMILIES):
    probs, cand = score_all(rows, P, S, Zl, mean_emb, labels, T_raw)
    ws = world_scores(rows, probs, cand)
    res, _ = primary(rows, ws, min_pairs, min_fams)
    out = {"admissibility": audit, "primary": res}
    out["L2"] = l2(rows, ws) if res["outcome"] == "MET" else {"status": "NOT EVALUATED (H1 not MET; v0.3 §6.3)"}
    out["secondary"] = secondaries(rows, ws)
    out["H1_outcome"] = out["L2"]["outcome"] if res["outcome"] == "MET" else res["outcome"]
    return out, ws


# ------------------------------------------------------------------ real run
def run(out_dir: Path, unlock_root: Path = SA.UNLOCK_ROOT) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    started = out_dir / "EVAL_STARTED.json"
    if started.exists():
        raise SystemExit("an evaluation was already started: a second evaluation requires a new owner decision")
    frz = json.loads((ROOT / "reports" / "BTC-H1_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
    prep = frz["prepared"]["BTC-H1-TEST-EVAL"]   # evaluator + standardisation hashes as submitted for BTC-H1-TEST-EVAL
    pre = {"frozen_code": all(sha(p) == frz["code_sha256"][p] for p in FROZEN_CODE),
           "evaluator_frozen": sha("src/sealed_stage/h1_eval_test.py") == prep["code_sha256"]["src/sealed_stage/h1_eval_test.py"],
           "pool_seal_intact": all(sealed.hash_only(SA.POOL_DIR).values()),
           "activation_seal_intact": all(sealed.hash_only(SA.ACT_DIR).values()),
           "oracle_never_read_before": SA.reads_outside_stage(SA.POOL_DIR, "D3-TEST-CAPTURE") == [],
           "activations_never_read_before": SA.access_log(SA.ACT_DIR) == []}
    if not all(pre.values()):
        raise SystemExit(f"PRECHECK FAILED (nothing read): {pre}")
    L = Z.load_final(ROOT / "outputs" / "h1_dev", frz["artifacts"]["H1-DEV-TRAIN"]["predictors"])
    S = Z.load_standardisation(ROOT / "outputs" / "h1_dev" / "final" / Z.STD_FILE, prep["standardisation_sha256"])
    started.write_text(json.dumps({"utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"), "prechecks": pre}) + "\n",
                       encoding="utf-8")
    P = L["predictors"]
    layers = sorted({p["input"] for p in P.values() if isinstance(p["input"], int)})
    idx, Zl, mean_emb, labels, T_raw, inputs, worlds = load_test(SA.POOL_DIR, SA.ACT_DIR, unlock_root, layers)
    rows, audit = build_rows(idx, inputs, worlds)
    res, ws = evaluate(rows, audit, P, S, Zl, mean_emb, labels, T_raw)
    res["prechecks"] = pre
    res["predictor_sha256"] = {n: p["sha256"] for n, p in P.items()}
    (out_dir / "per_world.jsonl").write_text("".join(json.dumps({"world_id": r["world_id"], "variant": r["variant"],
                                                                 "depth": r["depth"], **w}) + "\n" for r, w in zip(rows, ws)),
                                             encoding="utf-8")
    (out_dir / "EVAL_RESULT.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if not a.run:
        raise SystemExit("nothing to do (the single evaluation needs the owner's H1-TEST-EVAL unlock)")
    H.set_threads(4)
    res = run(ROOT / "outputs" / "h1_test")
    print(json.dumps({k: res[k] for k in ("H1_outcome", "admissibility")}, indent=1))


if __name__ == "__main__":
    main()
