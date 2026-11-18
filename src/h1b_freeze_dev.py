"""BTC-H1b v1.0 freeze preparation (CPU; PROBE-DEV only; owner decision "BTC-H1b v1.0").

    python src/h1b_freeze_dev.py --run                    # single run: DEV report + frozen decoders
    python src/h1b_freeze_dev.py --synthetic-test <dir>   # tests only

Part A — DEV report (EXPLORATORY; H1b inner/outer family splits; every choice on inner-val; DEV-val reported only):
  * C_lin in two training variants: AdamW (same optimiser / batch / early stopping / 3 seeds as P) and L-BFGS
    (lambda in {1e-2, 1, 100} chosen on inner-val). The COMPETITIVE variant = higher inner-val u2 accuracy
    (ties -> L-BFGS) is the one frozen (owner condition 1).
  * elimination control (D-8): P_c2 head (pointer, slot c2); u2_elim = the prompt-derived depth-2 candidate that is NOT
    P_c2's choice among the depth-2 candidates. Prompt-derived depth-2 set = targets of rule lines whose source is an
    action-line label (prompt only; no oracle).
Part B — frozen decoders (protocol v1.0): E chosen by early stopping on DEV-val after fitting on DEV-train, then refit
  on ALL DEV training-rule worlds with fixed E and seeds 0,1,2 (averaged); L-BFGS decoders refit with their inner-val
  lambda. Standardisation (z, keys) and mu_choice from DEV-train worlds. Decoders: P_u2 (layer 40; also 32 and 48,
  descriptive), P_c1 (sensitivity check), P_c2 (elimination), C_bil, C_lin (competitive variant), K (key-only), B3+.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import h1_features as F  # noqa: E402
import h1_probe as H  # noqa: E402
import h1_train_dev as T  # noqa: E402
import h1b_explore as X  # noqa: E402
import h1b_x3x4_analysis as A  # noqa: E402

LAYER, QUERY, REP, R, WD, SEEDS = 40, "ANS", "ctx_tgt", 8, 0.1, (0, 1, 2)
LAMS = (1e-2, 1.0, 100.0)


def depth2_from_prompt(prompt: str) -> set[str]:
    d1 = {t for _, t in F.ACT.findall(prompt)}
    return {t for s, t in F.NXT.findall(prompt) if s in d1}


# ------------------------------------------------------------------ linear choice+keys / key-only models
def lin_features(K: torch.Tensor, cand: torch.Tensor, choice_A: torch.Tensor | None):
    X_ = torch.zeros(len(cand), H.K, K.shape[1])
    M = cand == H.PAD
    X_[~M] = K[cand[~M]]
    if choice_A is not None:
        a = choice_A.view(-1, 1, 1)
        X_ = torch.cat([X_ * a, X_ * (1 - a)], -1)
    return X_, M


def train_linear_adamw(Xi, Mi, yi, Xv, Mv, yv, wd, seed, max_epochs=H.MAX_EPOCHS):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(Xi.shape[-1], generator=g) / math.sqrt(Xi.shape[-1])).requires_grad_()
    opt = torch.optim.AdamW([w], lr=H.LR, weight_decay=wd)
    best, best_ep, bad, state = math.inf, 0, 0, w.detach().clone()
    for ep in range(1, max_epochs + 1):
        perm = torch.randperm(len(Xi), generator=g)
        for i in range(0, len(perm), H.BATCH):
            b = perm[i:i + H.BATCH]
            loss = torch.nn.functional.cross_entropy((Xi[b] @ w).masked_fill(Mi[b], float("-inf")), yi[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            vl = float(torch.nn.functional.cross_entropy((Xv @ w).masked_fill(Mv, float("-inf")), yv))
        if vl < best - 1e-6:
            best, best_ep, bad, state = vl, ep, 0, w.detach().clone()
        else:
            bad += 1
            if bad >= H.PATIENCE:
                break
    return state, best_ep


def train_linear_fixed(Xa, Ma, ya, wd, seed, epochs):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(Xa.shape[-1], generator=g) / math.sqrt(Xa.shape[-1])).requires_grad_()
    opt = torch.optim.AdamW([w], lr=H.LR, weight_decay=wd)
    for _ in range(epochs):
        perm = torch.randperm(len(Xa), generator=g)
        for i in range(0, len(perm), H.BATCH):
            b = perm[i:i + H.BATCH]
            loss = torch.nn.functional.cross_entropy((Xa[b] @ w).masked_fill(Ma[b], float("-inf")), ya[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return w.detach()


def lin_proba(ws, X_, M):
    with torch.no_grad():
        return torch.stack([torch.softmax((X_ @ w).masked_fill(M, float("-inf")), -1) for w in ws]).mean(0)


def _bil_fixed(Z, c, y, Tt, seed, epochs):
    g = torch.Generator().manual_seed(seed)
    d, dk = Z.shape[1], Tt.shape[1]
    U = (torch.randn(d, R, generator=g) / math.sqrt(d)).requires_grad_()
    V = (torch.randn(dk, R, generator=g) / math.sqrt(dk)).requires_grad_()
    opt = torch.optim.AdamW([U, V], lr=H.LR, weight_decay=WD)
    for _ in range(max(epochs, 1)):
        perm = torch.randperm(len(Z), generator=g)
        for i in range(0, len(perm), H.BATCH):
            b = perm[i:i + H.BATCH]
            loss = torch.nn.functional.cross_entropy(H._scores(Z[b], c[b], Tt, U, V), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return H.Fit(U.detach(), V.detach(), epochs, float("nan"))


# ------------------------------------------------------------------ helpers
class Inputs:
    """Standardised query/keys at one layer for a set of ids, with stats from `fit_ids`."""

    def __init__(self, data, xd, layer, fit_ids):
        zf, _, kf = A.build_inputs(xd, layer, fit_ids, QUERY, REP)
        self.zmu, self.zsd = A.standardise_fit(zf)
        self.kmu, self.ksd = A.standardise_fit(kf)
        ch = np.array([data.dev.rows[i]["choice"] for i in fit_ids])
        self.mu_choice = {c: zf[ch == c].mean(0) for c in ("A", "B")}
        self.data, self.xd, self.layer = data, xd, layer

    def get(self, ids):
        z, c, k = A.build_inputs(self.xd, self.layer, ids, QUERY, REP)
        rows = [self.data.dev.rows[i] for i in ids]
        Z = torch.from_numpy(((z - self.zmu) / self.zsd).astype(np.float32))
        Kt = torch.from_numpy(((k - self.kmu) / self.ksd).astype(np.float32))
        Q = torch.from_numpy(np.stack([(self.mu_choice[r["choice"]] - self.zmu) / self.zsd for r in rows]).astype(np.float32))
        chA = torch.tensor([float(r["choice"] == "A") for r in rows])
        return {"Z": Z, "c": c, "K": Kt, "Q": Q, "chA": chA}

    def stats(self, prefix):
        return {f"{prefix}_zmu": self.zmu, f"{prefix}_zsd": self.zsd, f"{prefix}_kmu": self.kmu, f"{prefix}_ksd": self.ksd,
                f"{prefix}_mu_A": self.mu_choice["A"], f"{prefix}_mu_B": self.mu_choice["B"]}


def elim_predict(prob_c2, ids, data):
    """u2_elim canonical index per world (depth-2 candidate not chosen by P_c2 among the prompt-derived depth-2 set)."""
    out = []
    for k, i in enumerate(ids):
        r = data.dev.rows[i]
        d2 = [j for j, l in enumerate(r["cands"]) if l in depth2_from_prompt(r["prompt"])]
        assert len(d2) == 2, f"{r['world_id']}: {len(d2)} depth-2 candidates"
        p = prob_c2[k, d2]
        c2_hat = d2[int(torch.argmax(p))] if p[0] != p[1] else d2[0]
        out.append([j for j in d2 if j != c2_hat][0])
    return np.array(out)


def acc_idx(pred, ids, data, slot):
    return float(np.mean([pred[k] == data.dev.rows[i]["tgt"][slot] for k, i in enumerate(ids)]))


def paired_idx(pred, ids, data):
    pos = {i: k for k, i in enumerate(ids)}
    ps = [(a, b) for a, b in data.dev.pairs["val"] if a in pos and b in pos]
    ok = lambda i: float(pred[pos[i]] == data.dev.rows[i]["tgt"]["u2"])
    return float(np.mean([ok(a) * ok(b) for a, b in ps])), len(ps)


# ------------------------------------------------------------------ part A: DEV report
def part_a(data, xd, seeds, max_epochs):
    rep = {}
    keep = lambda ids, s: X.keep(data, ids, s)
    itr, iva, va = keep(data.itr, "u2"), keep(data.iva, "u2"), keep(data.va, "u2")
    inp = Inputs(data, xd, LAYER, itr)
    I, V_, O = inp.get(itr), inp.get(iva), inp.get(va)
    yi, yv = X.ytensor(data, itr, "u2"), X.ytensor(data, iva, "u2")
    (Xi, Mi), (Xv, Mv), (Xo, Mo) = (lin_features(d["K"], d["c"], d["chA"]) for d in (I, V_, O))
    res = {}
    ws = [train_linear_adamw(Xi, Mi, yi, Xv, Mv, yv, WD, s, max_epochs) for s in seeds]
    W = [w for w, _ in ws]
    res["C_lin_adamw"] = {"epochs": [e for _, e in ws], "inner_val": X.acc(lin_proba(W, Xv, Mv), iva, data, "u2"),
                          "val": X.acc(lin_proba(W, Xo, Mo), va, data, "u2")}
    res["C_lin_adamw"]["val_paired"] = X.paired(lin_proba(W, Xo, Mo), va, data)[0]
    best = None
    for lam in LAMS:
        w = H.train_conditional_logit(Xi, Mi, yi, lam)
        r = {"lambda": lam, "inner_val": X.acc(lin_proba([w], Xv, Mv), iva, data, "u2"), "val": X.acc(lin_proba([w], Xo, Mo), va, data, "u2"),
             "val_paired": X.paired(lin_proba([w], Xo, Mo), va, data)[0]}
        res.setdefault("C_lin_lbfgs_grid", []).append(r)
        if best is None or r["inner_val"] > best["inner_val"]:
            best = r
    res["C_lin_lbfgs"] = best
    res["C_lin_competitive"] = "adamw" if res["C_lin_adamw"]["inner_val"] > best["inner_val"] else "lbfgs"
    # elimination control on DEV-val
    c2i, c2v, c2o = keep(data.itr, "c2"), keep(data.iva, "c2"), keep(data.va, "c2")
    inp2 = Inputs(data, xd, LAYER, c2i)
    a, b, o = inp2.get(c2i), inp2.get(c2v), inp2.get(c2o)
    fits = [A._train_two_tables(a["Z"], a["c"], X.ytensor(data, c2i, "c2"), a["K"], b["Z"], b["c"], X.ytensor(data, c2v, "c2"), b["K"], R, WD, s, max_epochs)
            for s in seeds]
    oo = inp2.get(va)
    p_c2 = H.predict_proba(fits, oo["Z"], oo["c"], oo["K"])
    el = elim_predict(p_c2, va, data)
    res["elimination"] = {"P_c2_val": X.acc(H.predict_proba(fits, o["Z"], o["c"], o["K"]), c2o, data, "c2"),
                          "u2_elim_val": acc_idx(el, va, data, "u2"), "u2_elim_val_paired": paired_idx(el, va, data)[0]}
    rep["u2_dev_report"] = res
    return rep


# ------------------------------------------------------------------ part B: frozen decoders
def part_b(data, xd, outdir, seeds, max_epochs, layers=(32, 40, 48)):
    dec, stats, info = {}, {}, {}
    tr_all, va_all = data.tr, data.va                       # outer DEV split for E selection
    for layer in layers:
        slots = ("u2", "c1", "c2") if layer == LAYER else ("u2",)
        for slot in slots:
            tr, va = X.keep(data, tr_all, slot), X.keep(data, va_all, slot)
            allids = tr + va
            inp = Inputs(data, xd, layer, tr)
            t, v, al = inp.get(tr), inp.get(va), inp.get(allids)
            yt, yv, ya = (X.ytensor(data, ids, slot) for ids in (tr, va, allids))
            kinds = [("P", "Z")] + ([("C_bil", "Q")] if (slot == "u2" and layer == LAYER) else [])
            for kind, qk in kinds:
                es = [A._train_two_tables(t[qk], t["c"], yt, t["K"], v[qk], v["c"], yv, v["K"], R, WD, s, max_epochs).best_epoch for s in seeds]
                E = int(round(np.mean(es)))
                fits = [_bil_fixed(al[qk], al["c"], ya, al["K"], s, E) for s in seeds]
                name = f"{kind}_{slot}_L{layer}"
                dec[name] = {"U": [f.U for f in fits], "V": [f.V for f in fits], "E": E, "es_epochs": es, "query": qk, "layer": layer, "slot": slot}
            if slot == "u2" and layer == LAYER:
                (Xt, Mt), (Xv, Mv), (Xa, Ma) = (lin_features(d["K"], d["c"], d["chA"]) for d in (t, v, al))
                es = [train_linear_adamw(Xt, Mt, yt, Xv, Mv, yv, WD, s, max_epochs)[1] for s in seeds]
                E = int(round(np.mean(es)))
                dec["C_lin_adamw_u2_L40"] = {"w": [train_linear_fixed(Xa, Ma, ya, WD, s, max(E, 1)) for s in seeds], "E": E, "es_epochs": es}
                Ka, KMa = lin_features(al["K"], al["c"], None)
                dec["_pending_lbfgs"] = (Xa, Ma, ya, Ka, KMa)
            stats.update(inp.stats(f"L{layer}_{slot}"))
    return dec, stats, info


def run(data, xd, out: Path, seeds=SEEDS, max_epochs=H.MAX_EPOCHS) -> dict:
    if (out / "FREEZE.json").exists():
        raise SystemExit("FREEZE.json exists: the freeze is written once")
    H.set_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    meter = H.BudgetMeter(4.0)
    rep = part_a(data, xd, seeds, max_epochs)
    lam_c = rep["u2_dev_report"]["C_lin_lbfgs"]["lambda"]
    dec, stats, info = part_b(data, xd, out, seeds, max_epochs)
    Xa, Ma, ya, Ka, KMa = dec.pop("_pending_lbfgs")
    dec["C_lin_lbfgs_u2_L40"] = {"w": [H.train_conditional_logit(Xa, Ma, ya, lam_c)], "lambda": lam_c}
    comp = rep["u2_dev_report"]["C_lin_competitive"]
    dec["C_lin_u2_L40"] = dict(dec[f"C_lin_{comp}_u2_L40"], variant=comp)
    # key-only K: lambda on inner-val, refit on all DEV
    itr, iva = X.keep(data, data.itr, "u2"), X.keep(data, data.iva, "u2")
    inp = Inputs(data, xd, LAYER, itr)
    a, b = inp.get(itr), inp.get(iva)
    (Ki, KMi), (Kv, KMv) = lin_features(a["K"], a["c"], None), lin_features(b["K"], b["c"], None)
    lamK = max(LAMS, key=lambda l: X.acc(lin_proba([H.train_conditional_logit(Ki, KMi, X.ytensor(data, itr, "u2"), l)], Kv, KMv), iva, data, "u2"))
    dec["K_u2_L40"] = {"w": [H.train_conditional_logit(Ka, KMa, ya, lamK)], "lambda": lamK}
    # B3+ (prompt + choice): lambda on inner-val, refit on all DEV
    allu2 = X.keep(data, data.tr + data.va, "u2")
    (Bi, BMi), (Bv, BMv), (Ba, BMa) = X.b3plus(data, itr), X.b3plus(data, iva), X.b3plus(data, allu2)
    lamB = max((1e-4, 1e-2, 1.0), key=lambda l: X.acc(lin_proba([H.train_conditional_logit(Bi, BMi, X.ytensor(data, itr, "u2"), l)], Bv, BMv), iva, data, "u2"))
    dec["B3plus_u2"] = {"w": [H.train_conditional_logit(Ba, BMa, X.ytensor(data, allu2, "u2"), lamB)], "lambda": lamB}
    ddir = out / "decoders"
    ddir.mkdir(exist_ok=True)
    hashes = {}
    for name, obj in dec.items():
        p = ddir / f"{name}.pt"
        torch.save(obj, p)
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    np.savez(out / "STANDARDISATION.npz", **stats)
    freeze = {"protocol": "BTC-H1b v1.0 (proposal frozen on owner validation)", "config": {"layer": LAYER, "query": QUERY, "rep": REP, "r": R,
              "lambda": WD, "seeds": list(seeds)}, "dev_report": rep, "decoders_sha256": hashes,
              "standardisation_sha256": hashlib.sha256((out / "STANDARDISATION.npz").read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cpu_hours": meter.used_s() / 3600,
              "lambdas": {"C_lin_lbfgs": lam_c, "K": lamK, "B3plus": lamB}, "C_lin_frozen_variant": comp}
    (out / "FREEZE.json").write_text(json.dumps(freeze, indent=2, default=float) + "\n", encoding="utf-8")
    return freeze


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--synthetic-test", default="")
    a = ap.parse_args()
    if a.synthetic_test:
        data, xdir, pos = A.make_synthetic_x(Path(a.synthetic_test), n_worlds=960)
        f = run(data, A.XData(data, xdir, pos), Path(a.synthetic_test) / "out", seeds=(0,), max_epochs=10)
        print(json.dumps({k: f[k] for k in ("dev_report", "C_lin_frozen_variant", "lambdas")}, indent=1, default=float))
        return
    if not a.run:
        raise SystemExit("use --run or --synthetic-test")
    dev = T.Dev(T.REAL["capture"], T.REAL["dev"])
    data = X.Data(dev, T.REAL["dev"])
    xd = A.XData(data, ROOT / "outputs" / "d3x" / "probe_dev", ROOT / "outputs" / "d3x" / "plan" / "positions.jsonl")
    f = run(data, xd, ROOT / "outputs" / "h1b_freeze")
    print(json.dumps({k: f[k] for k in ("dev_report", "C_lin_frozen_variant", "lambdas", "cpu_hours")}, indent=1, default=float))


if __name__ == "__main__":
    main()
