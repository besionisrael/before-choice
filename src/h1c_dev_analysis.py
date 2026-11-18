"""BTC-H1c DEV analysis (CPU; EXPLORATORY selection on H1c DEV). PREPARED, NOT RUN on real data.
Refuses to run unless outputs/h1c_dev/capture/GATES.json says analysis_allowed (G2 bitwise identity, G3 competence).

    python src/h1c_dev_analysis.py --run
    python src/h1c_dev_analysis.py --synthetic-test <dir>     # tests only

Splits (family-disjoint): DEV-train families (300) -> inner-train / inner-val (80/20, seed 2026111405) for fitting and
EVERY choice; DEV-val families (100) reported only.
Primary-type decoder (binary, C-2'): candidates = the two successors of the unchosen depth-1 label (prompt + model
choice), keys = their ARRIVAL states; query = ANS; s(c) = z^T U V^T k_c, softmax over 2; target = the continuing one.
Selection grid (inner-val single-world binary accuracy): layer in {24, 32, 40, 48, 56} x rank in {2, 8}; lambda 0.1;
3 seeds; early stopping on inner-val.
Pre-declared positive controls (DEV-val; reported with the selection):
  PC1  12-way c1 pointer (ANS x all arrival keys) at the selected layer: sensitivity requirement >= 0.80
  PC2  binary CHOSEN continuation (successors of the chosen depth-1 label) on all worlds
  PC3  model's chosen letter, linear logistic on ANS
ROBUST PRIMARY STATISTIC (owner decision "BTC-H1c-DEV"): single-world binary accuracy over the worlds of admissible late
pairs vs 0.5, one-sided SIGN-FLIP randomisation test on family means (valid under dependent / anti-correlated
predictions because the generator randomises orientation; audited in outputs/h1c_design/orientation_checks.json).
Paired accuracy (vs 0.25) is reported descriptively only.
Controls on DEV-val late pairs (binary): C_bil (query = mu_choice), C_lin (choice x keys, L-BFGS), key-only (all
identical across members -> family statistic exactly 0), P_perm (between-family query, same choice and variant; K = 50)
and the contrast P - P_perm (sign-flip on family means). Secondary: early pairs, cswap reference,
query-position sweep (last rule-line end, reward-line ends, Q_END, ANS).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import h1_probe as H  # noqa: E402
import h1c_positions as HP  # noqa: E402
from h1b_x3x4_analysis import _train_two_tables  # noqa: E402

INNER_SEED, PERM_SEED, K_PERM = 2026111405, 2026111502, 50
GRID_R, WD, SEEDS = (2, 8), 0.1, (0, 1, 2)
PC1_MIN = 0.80


def lin_features(K, cand, choice_A):
    X_ = K[cand]                                           # (N, 2, d): the two binary candidates' keys
    if choice_A is not None:
        a = choice_A.view(-1, 1, 1)
        X_ = torch.cat([X_ * a, X_ * (1 - a)], -1)
    return X_, torch.zeros(cand.shape, dtype=torch.bool)


def u16f(a):
    return (a.astype(np.uint32) << 16).view(np.float32)


class Cap:
    def __init__(self, cap_dir: Path, dev_dir: Path, plan: Path):
        man = json.loads((cap_dir / "MANIFEST.json").read_text(encoding="utf-8"))
        self.layers = list(man["layers"])
        self.W = {r["world_id"]: r for r in map(json.loads, (dev_dir / "worlds.jsonl").read_text(encoding="utf-8").splitlines())}
        self.idx, self.files = {}, []
        for s in man["shards"]:
            ids = []
            for x in (cap_dir / f"{s['name']}_index.jsonl").read_text(encoding="utf-8").splitlines():
                r = json.loads(x)
                self.idx[r["world_id"]] = r
                ids.append(r["world_id"])
            self.files.append((cap_dir / f"{s['name']}_acts.npy", ids))
        self.rows = []
        for wid, r in self.idx.items():
            w = self.W[wid]
            ch = "A" if r["logp_A"] > r["logp_B"] else "B" if r["logp_A"] < r["logp_B"] else "TIE"
            if ch == "TIE" or ch != w["oracle"]["optimal_action"]:
                continue
            non = "B" if ch == "A" else "A"
            bs = HP.binary_slots(w["prompt"], ch)
            cands = sorted({e["target"] for e in w["transitions"]} - set(w["terminal_rewards"]))
            cont_u, cont_c = w["oracle"]["paths"][non][2], w["oracle"]["paths"][ch][2]
            bsc = HP.binary_slots(w["prompt"], non)               # successors of the CHOSEN depth-1 label
            self.rows.append({"world_id": wid, "family_id": w["family_id"], "variant": w["variant"], "split": w["split"], "choice": ch,
                              "bs": bs, "y": [cands[s - 1] for s in bs].index(cont_u), "bsc": bsc, "yc": [cands[s - 1] for s in bsc].index(cont_c),
                              "c1_slot": 1 + cands.index(w["oracle"]["paths"][ch][1]) - 1})
        fams = sorted({r["family_id"] for r in self.rows if r["split"] == "train"})
        perm = np.random.default_rng(INNER_SEED).permutation(len(fams))
        iv = {fams[j] for j in perm[: round(0.2 * len(fams))]}
        self.itr = [k for k, r in enumerate(self.rows) if r["split"] == "train" and r["family_id"] not in iv]
        self.iva = [k for k, r in enumerate(self.rows) if r["split"] == "train" and r["family_id"] in iv]
        self.va = [k for k, r in enumerate(self.rows) if r["split"] == "val"]
        self._cache = {}

    def layer(self, ell):
        if ell not in self._cache:
            self._cache.clear()
            j = self.layers.index(ell)
            out = {}
            for p, ids in self.files:
                a = np.load(p, mmap_mode="r")
                for k, w in enumerate(ids):
                    out[w] = u16f(np.asarray(a[k, :, j]))
            self._cache[ell] = out
        return self._cache[ell]


def binary_inputs(cap, ell, ids, q_slot=0, chosen=False):
    L = cap.layer(ell)
    rows = [cap.rows[k] for k in ids]
    z = np.stack([L[r["world_id"]][q_slot] for r in rows])
    keys = np.concatenate([L[r["world_id"]][r["bsc" if chosen else "bs"]] for r in rows])
    cand = torch.arange(len(rows) * 2).view(len(rows), 2)
    y = torch.tensor([r["yc" if chosen else "y"] for r in rows])
    return z, keys, cand, y


def std(train, *xs):
    mu, sd = train.mean(0), train.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return [torch.from_numpy(((x - mu) / sd).astype(np.float32)) for x in (train,) + xs]


def fit_binary(cap, ell, r, q_slot=0, chosen=False, seeds=SEEDS, max_epochs=H.MAX_EPOCHS):
    zi, ki, ci, yi = binary_inputs(cap, ell, cap.itr, q_slot, chosen)
    zv, kv, cv, yv = binary_inputs(cap, ell, cap.iva, q_slot, chosen)
    Zi, Zv = std(zi, zv)
    Ki, Kv = std(ki, kv)
    fits = [_train_two_tables(Zi, ci, yi, Ki, Zv, cv, yv, Kv, r, WD, s, max_epochs) for s in seeds]
    return fits, (zi, ki)


def predict(fits, stats, z, keys, cand):
    (zi, ki) = stats
    Z = std(zi, z)[1]
    K = std(ki, keys)[1]
    return H.predict_proba(fits, Z, cand, K)


def paired(cap, ids, pred_ok, variants=("base_L", "dswap_L")):
    pos = {cap.rows[k]["world_id"]: j for j, k in enumerate(ids)}
    fams = defaultdict(list)
    for k in ids:
        r = cap.rows[k]
        if r["variant"] == variants[0]:
            b = f"{r['family_id']}-{variants[1]}"
            if b in pos:
                fams[r["family_id"]].append(pred_ok[pos[r["world_id"]]] * pred_ok[pos[b]])
    vals = [v for vs in fams.values() for v in vs]
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def signflip(cap, ids, ok, R=10_000, seed=0, variants=("base_L", "dswap_L")):
    """Robust primary statistic (owner decision BTC-H1c-DEV): per family, mean binary correctness over the worlds of its
    admissible late pairs minus 0.5; one-sided sign-flip randomisation p-value on the family means (valid under
    anti-correlated predictions because orientation is randomised by the generator)."""
    pos = {cap.rows[k]["world_id"]: j for j, k in enumerate(ids)}
    fam = defaultdict(list)
    for k in ids:
        r = cap.rows[k]
        if r["variant"] == variants[0]:
            b = f"{r['family_id']}-{variants[1]}"
            if b in pos:
                fam[r["family_id"]] += [ok[pos[r["world_id"]]], ok[pos[b]]]
    d = np.array([np.mean(v) - 0.5 for v in fam.values()])
    if len(d) == 0:
        return {"n_families": 0}
    rng = np.random.default_rng(seed)
    null = (rng.choice((-1.0, 1.0), size=(R, len(d))) * d).mean(1)
    p = float((np.sum(null >= d.mean()) + 1) / (R + 1))
    return {"n_families": int(len(d)), "single_acc_late": float(d.mean() + 0.5), "signflip_p_one_sided": p}


def run(cap: Cap, out: Path, seeds=SEEDS, max_epochs=H.MAX_EPOCHS, layers=None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    H.set_threads(4)
    layers = layers or cap.layers
    res = {"selection": [], "n": {"inner_train": len(cap.itr), "inner_val": len(cap.iva), "val": len(cap.va)}}
    for ell in layers:
        for r in GRID_R:
            fits, st = fit_binary(cap, ell, r, seeds=seeds, max_epochs=max_epochs)
            zv, kv, cv, yv = binary_inputs(cap, ell, cap.iva)
            acc = float((H.argmax_canonical(predict(fits, st, zv, kv, cv)) == yv).float().mean())
            res["selection"].append({"layer": ell, "r": r, "inner_val_binary_acc": acc})
    best = max(res["selection"], key=lambda x: (x["inner_val_binary_acc"], -x["r"], -x["layer"]))
    ell, r = best["layer"], best["r"]
    res["selected"] = best
    fits, st = fit_binary(cap, ell, r, seeds=seeds, max_epochs=max_epochs)
    zo, ko, co, yo = binary_inputs(cap, ell, cap.va)
    ok = (H.argmax_canonical(predict(fits, st, zo, ko, co)) == yo).float().numpy()
    res["P"] = {"val_single": float(ok.mean()), "val_paired_late": paired(cap, cap.va, ok), "val_paired_early": paired(cap, cap.va, ok, ("base_E", "dswap_E")),
                "robust_primary_late": signflip(cap, cap.va, ok), "robust_early": signflip(cap, cap.va, ok, variants=("base_E", "dswap_E"))}
    # controls (binary): key-only and choice+keys (linear) on the same candidates; C_bil (mu_choice query)
    zi, ki, ci, yi = binary_inputs(cap, ell, cap.itr)
    Ki, Ko = std(ki, ko)
    chi = torch.tensor([float(cap.rows[k]["choice"] == "A") for k in cap.itr])
    cho = torch.tensor([float(cap.rows[k]["choice"] == "A") for k in cap.va])
    for name, ca, cb in (("key_only", None, None), ("C_lin", chi, cho)):
        Xi, Mi = lin_features(Ki, ci, ca)
        Xo, Mo = lin_features(Ko, co, cb)
        w = H.train_conditional_logit(Xi, Mi, yi, 0.01)
        okc = (H.argmax_canonical(torch.softmax((Xo @ w).masked_fill(Mo, float("-inf")), -1)) == yo).float().numpy()
        res[name] = {"val_single": float(okc.mean()), "val_paired_late": paired(cap, cap.va, okc), "robust_primary_late": signflip(cap, cap.va, okc)}
    mus = {c: zi[[cap.rows[k]["choice"] == c for k in cap.itr]].mean(0) for c in ("A", "B")}
    Qi = np.stack([mus[cap.rows[k]["choice"]] for k in cap.itr])
    zv, kv, cv, yv = binary_inputs(cap, ell, cap.iva)
    Qv = np.stack([mus[cap.rows[k]["choice"]] for k in cap.iva])
    Qo = np.stack([mus[cap.rows[k]["choice"]] for k in cap.va])
    QI = std(zi, Qi)[1]
    QV, QO = std(zi, Qv, Qo)[1:]
    KI, KV = std(ki, kv)
    fb = [_train_two_tables(QI, ci, yi, KI, QV, cv, yv, KV, r, WD, s, max_epochs) for s in seeds]
    okb = (H.argmax_canonical(H.predict_proba(fb, QO, co, Ko)) == yo).float().numpy()
    res["C_bil"] = {"val_single": float(okb.mean()), "val_paired_late": paired(cap, cap.va, okb), "robust_primary_late": signflip(cap, cap.va, okb)}
    # P_perm: query from another family, same choice and variant
    rng = np.random.default_rng(PERM_SEED)
    by = defaultdict(list)
    for j, k in enumerate(cap.va):
        by[(cap.rows[k]["choice"], cap.rows[k]["variant"])].append(j)
    okp = np.zeros((len(cap.va), K_PERM))
    for t in range(K_PERM):
        part = [rng.choice([i for i in by[(cap.rows[k]["choice"], cap.rows[k]["variant"])] if cap.rows[cap.va[i]]["family_id"] != cap.rows[k]["family_id"]])
                for k in cap.va]
        okp[:, t] = (H.argmax_canonical(predict(fits, st, zo[part], ko, co)) == yo).float().numpy()
    pos = {cap.rows[k]["world_id"]: j for j, k in enumerate(cap.va)}
    pp = [float(np.mean(okp[pos[cap.rows[k]["world_id"]]] * okp[pos[f"{cap.rows[k]['family_id']}-dswap_L"]]))
          for k in cap.va if cap.rows[k]["variant"] == "base_L" and f"{cap.rows[k]['family_id']}-dswap_L" in pos]
    res["P_perm"] = {"val_paired_late": float(np.mean(pp)) if pp else float("nan"), "robust_primary_late": signflip(cap, cap.va, okp.mean(1))}
    res["P_minus_Pperm_robust"] = signflip(cap, cap.va, ok - okp.mean(1) + 0.5)
    # positive controls
    fc, stc = fit_binary(cap, ell, r, chosen=True, seeds=seeds, max_epochs=max_epochs)
    zc, kc, cc, yc = binary_inputs(cap, ell, cap.va, chosen=True)
    res["PC2_chosen_continuation"] = {"val_single": float((H.argmax_canonical(predict(fc, stc, zc, kc, cc)) == yc).float().mean())}
    L = cap.layer(ell)
    za = np.stack([L[cap.rows[k]["world_id"]][0] for k in cap.itr]); zb = np.stack([L[cap.rows[k]["world_id"]][0] for k in cap.va])
    Za, Zb = std(za, zb)
    from h1b_explore import logistic, logistic_acc
    wl = logistic(Za, chi, 1e-1)
    res["PC3_choice_letter"] = {"val": logistic_acc(wl, Zb, cho)}
    # PC1: 12-way c1 pointer at ANS x all 12 arrival keys
    def c1_inputs(ids):
        z = np.stack([L[cap.rows[k]["world_id"]][0] for k in ids])
        keys = np.concatenate([L[cap.rows[k]["world_id"]][1:13] for k in ids])
        return z, keys, torch.arange(len(ids) * 12).view(len(ids), 12), torch.tensor([cap.rows[k]["c1_slot"] for k in ids])
    a, b, o = c1_inputs(cap.itr), c1_inputs(cap.iva), c1_inputs(cap.va)
    Za, Zb2, Zo2 = std(a[0], b[0], o[0])
    Ka, Kb2, Ko2 = std(a[1], b[1], o[1])
    f1 = [_train_two_tables(Za, a[2], a[3], Ka, Zb2, b[2], b[3], Kb2, 8, WD, s, max_epochs) for s in seeds]
    pc1 = float((H.argmax_canonical(H.predict_proba(f1, Zo2, o[2], Ko2)) == o[3]).float().mean())
    res["PC1_c1_12way"] = {"val": pc1, "requirement": PC1_MIN, "sensitive": pc1 >= PC1_MIN}
    res["chance"] = {"binary_single": 0.5, "binary_paired": 0.25, "c1_12way": 1 / 12}
    (out / "DEV_ANALYSIS.json").write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")
    return res


def make_synthetic(base: Path, n_fam=120, d=16, signal=4.0, seed=0, break_identity=False):
    """Fake DEV capture from the real H1c DEV pool: per family, random arrival keys shared by the members of each pair
    (prefix identity), ANS = signal x key of the CONTINUING unchosen successor (+ c1 key) + noise."""
    rng = np.random.default_rng(seed)
    dev = ROOT / "data" / "h1c" / "dev"
    W = [json.loads(x) for x in (dev / "worlds.jsonl").read_text(encoding="utf-8").splitlines()]
    fams = sorted({w["family_id"] for w in W})[:n_fam]
    W = [w for w in W if w["family_id"] in fams]
    (base / "dev").mkdir(parents=True, exist_ok=True)
    (base / "dev" / "worlds.jsonl").write_text("".join(json.dumps(w) + "\n" for w in W), encoding="utf-8")
    keys_by_fam = {f: {} for f in fams}
    acts, idx = [], []
    for w in W:
        opt = w["oracle"]["optimal_action"]
        non = "B" if opt == "A" else "A"
        cands = sorted({e["target"] for e in w["transitions"]} - set(w["terminal_rewards"]))
        kf = keys_by_fam[w["family_id"]]
        a = rng.standard_normal((HP.P, len(HP.DEV_LAYERS), d)).astype(np.float32) * 0.5
        for j, c in enumerate(cands):
            if c not in kf:
                kf[c] = rng.standard_normal((len(HP.DEV_LAYERS), d)).astype(np.float32)
            a[1 + j] = kf[c] + (rng.standard_normal(kf[c].shape).astype(np.float32) if break_identity and w["variant"] == "dswap_L" else 0)
        a[0] += signal * kf[w["oracle"]["paths"][non][2]] + signal * 0.5 * kf[w["oracle"]["paths"][opt][1]]
        acts.append(torch.from_numpy(a).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16))
        idx.append({"world_id": w["world_id"], "logp_A": -0.1 if opt == "A" else -3.0, "logp_B": -3.0 if opt == "A" else -0.1})
    cap = base / "capture"
    cap.mkdir(parents=True, exist_ok=True)
    np.save(cap / "xshard_000_acts.npy", np.stack(acts))
    (cap / "xshard_000_index.jsonl").write_text("".join(json.dumps(x) + "\n" for x in idx), encoding="utf-8")
    (cap / "MANIFEST.json").write_text(json.dumps({"layers": list(HP.DEV_LAYERS), "shards": [{"name": "xshard_000", "n": len(acts), "files": {}}]}), encoding="utf-8")
    return cap, base / "dev"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--synthetic-test", default="")
    a = ap.parse_args()
    if a.synthetic_test:
        cap, dev = make_synthetic(Path(a.synthetic_test))
        print(json.dumps(run(Cap(cap, dev, None), Path(a.synthetic_test) / "out", seeds=(0,), max_epochs=40, layers=[40]), indent=1, default=float))
        return
    if not a.run:
        raise SystemExit("use --run (after an authorised capture whose gates passed) or --synthetic-test")
    g = json.loads((ROOT / "outputs" / "h1c_dev" / "capture" / "GATES.json").read_text(encoding="utf-8"))
    if not g["analysis_allowed"]:
        raise SystemExit(f"ANALYSIS REFUSED: gates failed {g}")
    cap = Cap(ROOT / "outputs" / "h1c_dev" / "capture", ROOT / "data" / "h1c" / "dev", ROOT / "outputs" / "h1c_dev" / "plan" / "positions.jsonl")
    print(json.dumps(run(cap, ROOT / "outputs" / "h1c_dev" / "analysis"), indent=1, default=float))


if __name__ == "__main__":
    main()
