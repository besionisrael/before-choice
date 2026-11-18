"""BTC-H1b v1.0 — the SINGLE confirmatory evaluation on LOCKED-TEST-H1b. PREPARED, NOT RUN.
Requires a sealed, verified capture and the owner's H1B-EVAL unlock.   python src/sealed_stage/h1b_eval_test.py --run

Frozen decision rule (protocol v1.0):
  population      admissible depth-4 upath pairs (base->upath_base, cf->upath_cf); admissible = forced choice from the
                  capture (logp_A vs logp_B; exact tie excluded) equals the oracle optimum
  decoders        frozen on PROBE-DEV (outputs/h1b_freeze): P (pointer, layer 40), C_bil, C_lin, P_perm (P with the query of
                  a world from a DIFFERENT family, same model choice and variant; K = 50 seeded draws), P_c1, P_c2, K, B3+
  sensitivity     P_c1 single-world accuracy on admissible TEST worlds >= 0.80, else SENSITIVITY CHECK FAILED (descriptive)
  insufficient    < 875 contributing families -> INSUFFICIENT (descriptive)
  primary         family-cluster percentile bootstrap, 10,000 resamples, seed 0, shared by both contrasts;
                  in every resample Delta_C = P - max(C_bil, C_lin) and Delta_perm = P - P_perm (paired accuracy);
                  H1b MET iff both 2.5 % percentiles > 0
  secondary       S1 P - B3+, S2 P - K, S3 Delta_C on single worlds, S4 P - elimination(c2); one-sided bootstrap p, Holm
  descriptive     layers 32 / 48, by model choice, C_bil and C_lin separately
Interpretation (owner): MET = additional decodable u2 information in the pre-answer query beyond choice + label keys,
NOT necessarily an independent representation of the unchosen branch (it may still come from the chosen branch).
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
import h1_probe as H  # noqa: E402
import h1b_explore as X  # noqa: E402
from h1b_freeze_dev import depth2_from_prompt, lin_features  # noqa: E402

STAGE = "H1B-EVAL"
LAYERS, PRIMARY = (32, 40, 48), 40
B_RESAMPLES, BOOT_SEED, K_PERM, PERM_SEED = 10_000, 0, 50, 2026111502
MIN_FAMILIES, C1_MIN = 875, 0.80


def u16f(a):
    return (a.astype(np.uint32) << 16).view(np.float32)


# ------------------------------------------------------------------ core (also used by synthetic tests)
def build_rows(idx, inputs, worlds):
    rows, audit = [], defaultdict(int)
    for g, r in enumerate(idx):
        w = worlds[r["world_id"]]
        ch = "A" if r["logp_A"] > r["logp_B"] else "B" if r["logp_A"] < r["logp_B"] else "TIE"
        audit["n_worlds"] += 1
        audit["n_ties"] += ch == "TIE"
        if ch == "TIE" or ch != w["oracle"]["optimal_action"]:
            audit["n_not_admissible"] += 1
            continue
        prompt = inputs[r["world_id"]]["prompt"]
        cands = F.canonical_candidates(F.candidates_from_prompt(prompt))
        sup = F.supervision_targets(w, ch)
        assert len(cands) == 6 and all(l in cands for l in sup.values()), f"{r['world_id']}: unexpected candidates"
        rows.append({"g": g, "world_id": r["world_id"], "family_id": w["family_id"], "variant": w["variant"], "depth": w["depth"],
                     "choice": ch, "prompt": prompt, "cands": cands, "tgt": {s: cands.index(l) for s, l in sup.items()}})
    audit["n_admissible"] = len(rows)
    return rows, dict(audit)


def inputs_for(rows, acts_layer, stats, prefix):
    """acts_layer: (N_all, 7, d) float32 for one layer; slot 0 query, slots 1..6 arrival keys (canonical order)."""
    g = [r["g"] for r in rows]
    z = acts_layer[g, 0]
    keys = acts_layer[g, 1:7].reshape(-1, acts_layer.shape[-1])
    Z = torch.from_numpy(((z - stats[f"{prefix}_zmu"]) / stats[f"{prefix}_zsd"]).astype(np.float32))
    Kt = torch.from_numpy(((keys - stats[f"{prefix}_kmu"]) / stats[f"{prefix}_ksd"]).astype(np.float32))
    cand = torch.arange(len(rows) * 6).view(len(rows), 6)
    Q = torch.from_numpy(np.stack([(stats[f"{prefix}_mu_{r['choice']}"] - stats[f"{prefix}_zmu"]) / stats[f"{prefix}_zsd"] for r in rows]).astype(np.float32))
    chA = torch.tensor([float(r["choice"] == "A") for r in rows])
    return Z, Kt, cand, Q, chA


def bil(dec, q, cand, Kt):
    fits = [H.Fit(U, V, dec["E"], float("nan")) for U, V in zip(dec["U"], dec["V"])]
    return H.predict_proba(fits, q, cand, Kt)


def lin(dec, Xf, M):
    with torch.no_grad():
        return torch.stack([torch.softmax((Xf @ w).masked_fill(M, float("-inf")), -1) for w in dec["w"]]).mean(0)


def perm_partners(rows, k, seed):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for j, r in enumerate(rows):
        by[(r["choice"], r["variant"])].append(j)
    out, n_av = [], []
    for j, r in enumerate(rows):
        cand = [i for i in by[(r["choice"], r["variant"])] if rows[i]["family_id"] != r["family_id"]]
        assert cand, f"no between-family partner for {r['world_id']}"
        n_av.append(len(cand))
        out.append(rng.choice(cand, size=k, replace=True))
    return np.stack(out), n_av


def evaluate(rows, acts, dec, stats, min_families=MIN_FAMILIES, n_boot=B_RESAMPLES, k_perm=K_PERM):
    am = lambda p: H.argmax_canonical(p).numpy()
    ok = lambda pred, slot: np.array([float(pred[k] == r["tgt"][slot]) for k, r in enumerate(rows)])
    W = {}
    Z, Kt, cand, Q, chA = inputs_for(rows, acts[PRIMARY], stats, "L40_u2")
    W["P"] = ok(am(bil(dec["P_u2_L40"], Z, cand, Kt)), "u2")
    W["C_bil"] = ok(am(bil(dec["C_bil_u2_L40"], Q, cand, Kt)), "u2")
    Xc, Mc = lin_features(Kt, cand, chA)
    W["C_lin"] = ok(am(lin(dec["C_lin_u2_L40"], Xc, Mc)), "u2")
    Xk, Mk = lin_features(Kt, cand, None)
    W["K"] = ok(am(lin(dec["K_u2_L40"], Xk, Mk)), "u2")
    Xb, Mb = X.b3plus(type("D", (), {"dev": type("R", (), {"rows": rows})()})(), list(range(len(rows))))
    W["B3plus"] = ok(am(lin(dec["B3plus_u2"], Xb, Mb)), "u2")
    partners, n_av = perm_partners(rows, k_perm, PERM_SEED)
    perm_ok = np.zeros((len(rows), k_perm))
    for kk in range(k_perm):
        perm_ok[:, kk] = ok(am(bil(dec["P_u2_L40"], Z[partners[:, kk]], cand, Kt)), "u2")
    Z1, K1, c1c, _, _ = inputs_for(rows, acts[PRIMARY], stats, "L40_c1")
    c1_acc = float(ok(am(bil(dec["P_c1_L40"], Z1, c1c, K1)), "c1").mean())
    Z2, K2, c2c, _, _ = inputs_for(rows, acts[PRIMARY], stats, "L40_c2")
    p_c2 = bil(dec["P_c2_L40"], Z2, c2c, K2)
    el = []
    for k, r in enumerate(rows):
        d2 = [j for j, l in enumerate(r["cands"]) if l in depth2_from_prompt(r["prompt"])]
        c2_hat = d2[int(torch.argmax(p_c2[k, d2]))] if p_c2[k, d2[0]] != p_c2[k, d2[1]] else d2[0]
        el.append([j for j in d2 if j != c2_hat][0])
    W["elim"] = ok(np.array(el), "u2")
    desc = {}
    for L in (32, 48):
        ZL, KL, cL, _, _ = inputs_for(rows, acts[L], stats, f"L{L}_u2")
        desc[f"P_L{L}"] = ok(am(bil(dec[f"P_u2_L{L}"], ZL, cL, KL)), "u2")
    # ---- pairs
    pos = {r["world_id"]: k for k, r in enumerate(rows)}
    pairs = defaultdict(list)
    for r in rows:
        if r["variant"] in ("base", "cf"):
            b = f"{r['family_id']}-upath_{r['variant']}"
            if b in pos:
                pairs[r["family_id"]].append((pos[r["world_id"]], pos[b]))
    fams = sorted(pairs)
    pair_vals = {n: np.array([[W[n][a] * W[n][b] for a, b in pairs[f]] for f in fams], dtype=object) for n in W}
    perm_pair = np.array([[float(np.mean(perm_ok[a] * perm_ok[b])) for a, b in pairs[f]] for f in fams], dtype=object)
    S = {n: np.array([sum(v) for v in pair_vals[n]]) for n in W}
    S["P_perm"] = np.array([sum(v) for v in perm_pair])
    for L in (32, 48):
        S[f"P_L{L}"] = np.array([sum(desc[f"P_L{L}"][a] * desc[f"P_L{L}"][b] for a, b in pairs[f]) for f in fams])
    cnt = np.array([len(pairs[f]) for f in fams], dtype=float)
    point = {n: float(S[n].sum() / cnt.sum()) for n in S}
    single = {n: float(W[n].mean()) for n in W}
    res = {"n_pairs": int(cnt.sum()), "n_contributing_families": len(fams), "point_paired": point, "single_world": single,
           "P_c1_single": c1_acc, "perm_partners": {"min": int(min(n_av)), "median": float(np.median(n_av))}}
    if c1_acc < C1_MIN:
        res["outcome"] = "SENSITIVITY CHECK FAILED"
        return res
    if len(fams) < min_families:
        res["outcome"] = "INSUFFICIENT"
        return res
    rng = np.random.default_rng(BOOT_SEED)
    Wt = rng.multinomial(len(fams), np.full(len(fams), 1 / len(fams)), size=n_boot).astype(float)
    den = Wt @ cnt
    M = {n: (Wt @ S[n]) / den for n in S}
    dC = M["P"] - np.maximum(M["C_bil"], M["C_lin"])
    dQ = M["P"] - M["P_perm"]
    lo = lambda a: float(np.quantile(a, 0.025))
    res["Delta_C"] = {"point": point["P"] - max(point["C_bil"], point["C_lin"]), "lower": lo(dC), "upper": float(np.quantile(dC, 0.975)),
                      "C_max_point": "C_bil" if point["C_bil"] > point["C_lin"] else "C_lin"}
    res["Delta_perm"] = {"point": point["P"] - point["P_perm"], "lower": lo(dQ), "upper": float(np.quantile(dQ, 0.975))}
    met = res["Delta_C"]["lower"] > 0 and res["Delta_perm"]["lower"] > 0
    res["outcome"] = "H1b MET" if met else "H1b NOT MET"
    res["failing_contrasts"] = [k for k in ("Delta_C", "Delta_perm") if not res[k]["lower"] > 0]
    # ---- secondaries (one-sided bootstrap p, Holm)
    fam_single = {n: np.array([sum(W[n][i] for i in {a for a, _ in pairs[f]} | {b for _, b in pairs[f]}) for f in fams]) for n in ("P", "C_bil", "C_lin")}
    n_single = np.array([len({a for a, _ in pairs[f]} | {b for _, b in pairs[f]}) for f in fams], dtype=float)
    Ms = {n: (Wt @ fam_single[n]) / (Wt @ n_single) for n in fam_single}
    sec = {"S1_P_minus_B3plus": M["P"] - M["B3plus"], "S2_P_minus_K": M["P"] - M["K"],
           "S3_DeltaC_single": Ms["P"] - np.maximum(Ms["C_bil"], Ms["C_lin"]), "S4_P_minus_elim": M["P"] - M["elim"]}
    ps = {k: float(np.mean(v <= 0)) for k, v in sec.items()}
    order, run_max, holm = sorted(ps, key=ps.get), 0.0, {}
    for i, k in enumerate(order):
        run_max = max(run_max, min(1.0, (len(order) - i) * ps[k]))
        holm[k] = run_max
    res["secondary"] = {k: {"point": float(np.mean(v)), "lower": lo(v), "p_one_sided": ps[k], "p_holm": holm[k]} for k, v in sec.items()}
    res["descriptive"] = {"P_paired_L32": point["P_L32"], "P_paired_L48": point["P_L48"],
                          "P_single_by_choice": {c: float(np.mean([W["P"][k] for k, r in enumerate(rows) if r["choice"] == c])) for c in ("A", "B")}}
    res["interpretation"] = ("MET = additional decodable u2 information in the pre-answer query beyond choice + label keys; not necessarily an "
                             "independent representation of the unchosen branch (it may come from the chosen branch). NOT MET = no "
                             "demonstration of the advantage. Neither establishes nor refutes a forward sweep.")
    return res


# ------------------------------------------------------------------ real run
def run(out_dir: Path) -> dict:
    import sealed
    import sealed_access_h1b as SA
    out_dir.mkdir(parents=True, exist_ok=True)
    started = out_dir / "EVAL_STARTED.json"
    if started.exists():
        raise SystemExit("an evaluation was already started: a second evaluation requires a new owner decision")
    v1 = json.loads((ROOT / "outputs" / "h1b_freeze" / "V1_MANIFEST.json").read_text(encoding="utf-8"))
    sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
    pre = {"frozen_files": all(sha(p) == h for p, h in {**v1["code_sha256"], **v1["artifact_sha256"]}.items()),
           "pool_seal": all(sealed.hash_only(SA.POOL_DIR).values()), "act_seal": all(sealed.hash_only(SA.ACT_DIR).values()),
           "no_oracle_read_before": SA.reads_outside_stage(SA.POOL_DIR, "H1B-CAPTURE") == [], "act_never_read": SA.access_log(SA.ACT_DIR) == []}
    if not all(pre.values()):
        raise SystemExit(f"PRECHECK FAILED (nothing read): {pre}")
    started.write_text(json.dumps({"utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"), "prechecks": pre}) + "\n", encoding="utf-8")
    rd = lambda d, rel, why: SA.staged_read(STAGE, d, rel, why)
    man = json.loads(rd(SA.ACT_DIR, "MANIFEST.json", "activation manifest"))
    idx, chunks = [], []
    for s in man["shards"]:
        idx += [json.loads(x) for x in rd(SA.ACT_DIR, f"{s['name']}_index.jsonl", "index").decode("utf-8").splitlines()]
        chunks.append(np.load(io.BytesIO(rd(SA.ACT_DIR, f"{s['name']}_acts.npy", "activations"))))
    a = np.concatenate(chunks)                                  # (N, 7, 3, d) uint16
    acts = {L: u16f(a[:, :, j]) for j, L in enumerate(man["layers"])}
    inputs = {r["world_id"]: r for r in map(json.loads, rd(SA.POOL_DIR, "model_inputs.jsonl", "prompts").decode("utf-8").splitlines())}
    worlds = {r["world_id"]: r for r in map(json.loads, rd(SA.POOL_DIR, "worlds.jsonl", "oracle labels").decode("utf-8").splitlines())}
    rows, audit = build_rows(idx, inputs, worlds)
    dec = {n: torch.load(ROOT / "outputs" / "h1b_freeze" / "decoders" / f"{n}.pt", weights_only=False) for n in v1["decoders"]}
    with np.load(ROOT / "outputs" / "h1b_freeze" / "STANDARDISATION.npz") as f:
        stats = {k: f[k] for k in f.files}
    res = evaluate(rows, acts, dec, stats)
    res["admissibility"], res["prechecks"] = audit, pre
    (out_dir / "EVAL_RESULT.json").write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    if not ap.parse_args().run:
        raise SystemExit("nothing to do (requires the owner's H1B-EVAL unlock)")
    H.set_threads(4)
    print(json.dumps(run(ROOT / "outputs" / "h1b_test"), indent=1, default=float))


if __name__ == "__main__":
    main()
