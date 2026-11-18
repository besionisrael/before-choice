"""BTC-H1b — Understanding the Negative Result. EXPLORATORY block X1 + X2 + X7 (CPU, PROBE-DEV only).

    python src/h1b_explore.py --run                    # the single authorised block (4 threads, 5 CPU-h cap)
    python src/h1b_explore.py --synthetic-test <dir>   # tests only

Status of every number produced here: EXPLORATORY (POST HOC). PROBE-DEV was already used to select the H1 v0.3 probe;
nothing here is confirmatory, nothing here touches LOCKED-TEST, the archived H1 v0.3 artifacts are only READ.

Splits (family-disjoint):
  OUTER  = the existing PROBE-DEV split: DEV-train (226 families) / DEV-val (74 families)  -> reported "train" / "val"
  INNER  = DEV-train families split 80/20 (seed 2026111501) -> inner-train (fitting) / inner-val (early stopping,
           hyper-parameter choice). DEV-val is NEVER used for any choice in this block; it is only reported.
Training rows = H1 v0.3 training-rule worlds (non-decoy, forced choice = oracle optimum, no tie).

Blocks (declared in PLAN, written to PLAN.json before any fit):
  X1a scalar positive controls: linear logistic probes for (i) the model's chosen letter, (ii) depth 3 vs 4,
      (iii) reward gap above the DEV-train median; every sampled vector (17) + mean_emb; prompt-only reference stated.
  X1b candidate positive controls: the H1 bilinear candidate probe for every slot (c1, c2, c3, u1, u2, u3) and for the
      purely LEXICAL slot A1 (the label printed in "At S, action A leads to X"), at vectors {0, 16, 32, 48, 61};
      prompt baseline B3+ (B3 features + "printed in the action line of letter A").
  X2  capacity / regularisation / learning curves for targets u2 and c1 at vectors {16, 32, 48}: bilinear r x lambda,
      PCA-reduced inputs, a small MLP head; then learning curves of the inner-val-best configuration.
  X7  strata fixed BEFORE any accuracy is computed: model confidence = |logp_A - logp_B| tertiles (thresholds from
      DEV-train training-rule worlds) x depth {3, 4}; accuracy of X1b probes and B3+ per stratum on DEV-val with counts;
      admissibility rate per confidence tertile (all non-decoy DEV-val worlds).
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

THREADS, CPU_HOURS = 4, 5.0
INNER_SEED = 2026111501
SLOTS = ("c1", "c2", "c3", "u1", "u2", "u3", "A1")
PLAN = {
    "status": "EXPLORATORY (POST HOC); PROBE-DEV already used for H1 v0.3 selection; no LOCKED-TEST access",
    "splits": {"outer": "existing PROBE-DEV train/val (family-disjoint)", "inner": "DEV-train families 80/20, seed 2026111501",
               "choices_made_on": "inner-val only", "reported": "inner-train (train), DEV-val (val)"},
    "X1a": {"targets": ["choice_is_A", "depth_is_4", "gap_above_train_median"], "inputs": "17 sampled vectors + mean_emb",
            "model": "L2 logistic (LBFGS)", "lambda_grid": [1e-3, 1e-1, 10.0], "baseline": "majority class (train)",
            "prompt_only_reference": {"depth_is_4": "exact from rule-line count (1.0)", "gap_above_train_median": "exact from printed rewards (1.0)",
                                      "choice_is_A": "requires path following (no trivial prompt rule)"}},
    "X1b": {"slots": list(SLOTS), "vectors": [0, 16, 32, 48, 61], "probe": "H1 bilinear, r=8", "lambda_grid": [1e-2, 1e-1],
            "seeds": [0, 1, 2], "baseline": "B3+ conditional logit, lambda grid [1e-4, 1e-2, 1.0]",
            "lexical_vs_relational": {"lexical (printed in an action line)": ["A1", "c1", "u1"], "1 hop": ["c2", "u2"], "2 hops": ["c3", "u3"]}},
    "X2": {"targets": ["u2", "c1"], "vectors": [16, 32, 48],
           "bilinear": {"r": [2, 8, 32], "lambda": [1e-1, 1.0, 10.0]},
           "pca_bilinear": {"k": [16, 64, 256], "r": 8, "lambda": 1e-1},
           "mlp": {"pca_z": 64, "pca_e": 64, "hidden": 64, "weight_decay": 1e-2},
           "seeds": [0, 1, 2], "learning_curve": {"fractions_of_inner_train_families": [0.25, 0.5, 1.0],
                                                  "config": "inner-val-best X2 configuration per target"}},
    "X7": {"confidence": "|logp_A - logp_B| tertiles of DEV-train training-rule worlds", "depth": [3, 4],
           "evaluated": ["X1b u2 and c1 probes at vector 32 (inner-val lambda)", "B3+ u2 and c1"],
           "also": "admissibility rate per confidence tertile, all non-decoy DEV-val worlds", "inference": "none (counts + Wilson 95% CI)"},
    "budget": {"cpu_hours": CPU_HOURS, "threads": THREADS, "meter": "time.process_time()"},
}


# ------------------------------------------------------------------ data
class Data:
    def __init__(self, dev: T.Dev, devdir: Path):
        self.dev = dev
        worlds = {r["world_id"]: r for r in (json.loads(x) for x in (devdir / "worlds.jsonl").read_text(encoding="utf-8").splitlines())}
        idx = {r["world_id"]: r for r in dev.idx}
        rows = dev.rows
        for r in rows:
            w, ix = worlds[r["world_id"]], idx[r["world_id"]]
            r["margin"] = abs(ix["logp_A"] - ix["logp_B"])
            r["reward_gap"] = w["reward_gap"]
            r["tgt"] = dict(r["tgt"])
            r["tgt"]["A1"] = r["tgt"]["c1"] if r["choice"] == "A" else r["tgt"]["u1"]
        self.tr, self.va = dev.by_split["train"], dev.by_split["val"]
        fams = sorted({rows[i]["family_id"] for i in self.tr})
        perm = np.random.default_rng(INNER_SEED).permutation(len(fams))
        inner_val_f = {fams[j] for j in perm[: round(0.2 * len(fams))]}
        self.itr = [i for i in self.tr if rows[i]["family_id"] not in inner_val_f]
        self.iva = [i for i in self.tr if rows[i]["family_id"] in inner_val_f]
        self.itr_fams = sorted({rows[i]["family_id"] for i in self.itr})
        med = float(np.median([rows[i]["reward_gap"] for i in self.tr]))
        for r in rows:
            r["y_choice"], r["y_depth"], r["y_gap"] = float(r["choice"] == "A"), float(r["depth"] == 4), float(r["reward_gap"] > med)
        self.gap_median = med
        # all non-decoy DEV-val worlds (incl. non-admissible) for the X7 admissibility table
        self.val_all = []
        for ix in dev.idx:
            w = worlds[ix["world_id"]]
            if w["split"] != "val" or w["variant"].startswith("decoy"):
                continue
            ch = "A" if ix["logp_A"] > ix["logp_B"] else "B" if ix["logp_A"] < ix["logp_B"] else "TIE"
            self.val_all.append({"margin": abs(ix["logp_A"] - ix["logp_B"]), "depth": w["depth"],
                                 "admissible": ch != "TIE" and ch == w["oracle"]["optimal_action"]})
        self.layer_cache = {}

    def rows(self, ids):
        return [self.dev.rows[i] for i in ids]

    def z(self, ell, ids):
        if ell not in self.layer_cache:
            self.layer_cache.clear()
            allids = self.tr + self.va
            a = self.dev.mean_emb(allids) if ell == "mean_emb" else self.dev.layer(ell, allids)
            posa = {i: k for k, i in enumerate(allids)}
            sub = a[[posa[i] for i in self.itr]]
            mu, sd = sub.mean(0), sub.std(0)
            sd = np.where(sd < 1e-6, 1.0, sd)
            self.layer_cache[ell] = ({i: k for k, i in enumerate(allids)}, torch.from_numpy(((a - mu) / sd).astype(np.float32)))
        pos, Z = self.layer_cache[ell]
        return Z[[pos[i] for i in ids]]

    def table(self):
        lab = sorted({self.dev.label_index[l] for i in self.itr for l in self.dev.rows[i]["cands"]})
        mu, sd = self.dev.T_raw[lab].mean(0), self.dev.T_raw[lab].std(0)
        return torch.from_numpy((self.dev.T_raw - mu) / np.where(sd < 1e-6, 1.0, sd)).float()


# ------------------------------------------------------------------ metrics
def acc(prob, ids, data, slot):
    pred = H.argmax_canonical(prob).numpy()
    return float(np.mean([pred[k] == data.dev.rows[i]["tgt"][slot] for k, i in enumerate(ids)])) if ids else float("nan")


def paired(prob, ids, data, slot="u2"):
    pos = {i: k for k, i in enumerate(ids)}
    pred = H.argmax_canonical(prob).numpy()
    ps = [(a, b) for a, b in data.dev.pairs["val"] if a in pos and b in pos]
    ok = lambda i: float(pred[pos[i]] == data.dev.rows[i]["tgt"][slot])
    return (float(np.mean([ok(a) * ok(b) for a, b in ps])) if ps else float("nan")), len(ps)


def chance(ids, data):
    return float(np.mean([1 / len(data.dev.rows[i]["cands"]) for i in ids])) if ids else float("nan")


def wilson(k, n):
    if n == 0:
        return [float("nan"), float("nan")]
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


# ------------------------------------------------------------------ models
def logistic(X, y, lam):
    w = torch.zeros(X.shape[1] + 1, requires_grad=True)
    Xb = torch.cat([X, torch.ones(len(X), 1)], 1)
    opt = torch.optim.LBFGS([w], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xb @ w, y) + lam * (w[:-1] ** 2).sum() / len(X)
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach()


def logistic_acc(w, X, y):
    return float((((torch.cat([X, torch.ones(len(X), 1)], 1) @ w) > 0).float() == y).float().mean())


def cand_tensor(data, ids):
    return data.dev.cand_tensor(ids)


def ix(ids, sub):
    pos = {i: k for k, i in enumerate(ids)}
    return [pos[i] for i in sub]


def keep(data, ids, slot):
    return [i for i in ids if slot in data.dev.rows[i]["tgt"]]


def ytensor(data, ids, slot):
    return torch.tensor([data.dev.rows[i]["tgt"][slot] for i in ids])


def b3plus(data, ids):
    x = torch.zeros(len(ids), H.K, len(F.B3_FEATURES) + 1)
    m = torch.ones(len(ids), H.K, dtype=torch.bool)
    for k, i in enumerate(ids):
        r = data.dev.rows[i]
        f = F.b3_features(r["prompt"], r["cands"], r["choice"])
        a_lab = {t for a, t in F.ACT.findall(r["prompt"]) if a == "A"}
        x[k, :len(r["cands"]), :-1] = torch.from_numpy(f)
        x[k, :len(r["cands"]), -1] = torch.tensor([float(c in a_lab) for c in r["cands"]])
        m[k, :len(r["cands"])] = False
    return x, m


class PCAProj:
    def __init__(self, X, k):
        self.mu = X.mean(0)
        _, _, V = torch.pca_lowrank(X - self.mu, q=k, center=False, niter=4)
        self.V = V[:, :k]

    def __call__(self, X):
        return (X - self.mu) @ self.V


def train_mlp(zp_tr, c_tr, y_tr, zp_va, c_va, y_va, Ep, hidden, wd, seed, max_epochs=H.MAX_EPOCHS):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    dz, de = zp_tr.shape[1], Ep.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(dz + de + min(dz, de), hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=H.LR, weight_decay=wd)
    m = min(dz, de)

    def scores(z, c):
        mask = c == H.PAD
        e = Ep[c.clamp(min=0)]
        zz = z.unsqueeze(1).expand(-1, c.shape[1], -1)
        s = net(torch.cat([zz, e, zz[..., :m] * e[..., :m]], -1)).squeeze(-1)
        return s.masked_fill(mask, float("-inf"))
    best, state, bad, best_ep = math.inf, None, 0, 0
    for ep in range(1, max_epochs + 1):
        perm = torch.randperm(len(zp_tr), generator=g)
        for i in range(0, len(perm), H.BATCH):
            b = perm[i:i + H.BATCH]
            loss = torch.nn.functional.cross_entropy(scores(zp_tr[b], c_tr[b]), y_tr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            vl = float(torch.nn.functional.cross_entropy(scores(zp_va, c_va), y_va))
        if vl < best - 1e-6:
            best, best_ep, bad, state = vl, ep, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= H.PATIENCE:
                break
    net.load_state_dict(state)

    def predict(z, c):
        with torch.no_grad():
            return torch.softmax(scores(z, c), -1)
    return predict, best_ep


# ------------------------------------------------------------------ runner
def run(data: Data, out: Path, layers=H.LAYERS, x1b_layers=(0, 16, 32, 48, 61), x2_layers=(16, 32, 48),
        seeds=(0, 1, 2), max_epochs=H.MAX_EPOCHS, plan=PLAN, x7_vector=32) -> dict:
    H.set_threads(THREADS)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.jsonl").exists():
        raise SystemExit("results.jsonl exists: a single block is authorised")
    plan = dict(plan, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                n={"inner_train": len(data.itr), "inner_val": len(data.iva), "val": len(data.va),
                   "inner_train_families": len(data.itr_fams), "val_pairs_u2": len(data.dev.pairs["val"])})
    tert = np.quantile([data.dev.rows[i]["margin"] for i in data.tr], [1 / 3, 2 / 3]).tolist()
    plan["X7_thresholds_from_train_margins"] = tert
    (out / "PLAN.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")   # BEFORE any fit
    meter = H.BudgetMeter(CPU_HOURS, out / "budget.tsv")
    log = open(out / "results.jsonl", "a", encoding="utf-8")

    def rec(d):
        d["cpu_s"] = meter.used_s()
        log.write(json.dumps(d) + "\n")
        log.flush()

    Tt = data.table()
    summary = {"X1a": {}, "X1b": {}, "B3plus": {}, "X2": {}, "LC": {}, "X7": {}}
    keep_probs = {}
    try:
        # ---------------- B3+ baseline (prompt only), lambda on inner-val
        for slot in SLOTS:
            itr, iva, va = keep(data, data.itr, slot), keep(data, data.iva, slot), keep(data, data.va, slot)
            Xi, Mi = b3plus(data, itr)
            Xv, Mv = b3plus(data, iva)
            Xo, Mo = b3plus(data, va)
            best = None
            for lam in (1e-4, 1e-2, 1.0):
                w = H.train_conditional_logit(Xi, Mi, ytensor(data, itr, slot), lam, meter, f"B3plus|{slot}|{lam}")
                pr = lambda X, M: torch.softmax((X @ w).masked_fill(M, float("-inf")), -1)
                r = {"block": "B3plus", "slot": slot, "lambda": lam, "train": acc(pr(Xi, Mi), itr, data, slot),
                     "inner_val": acc(pr(Xv, Mv), iva, data, slot), "val": acc(pr(Xo, Mo), va, data, slot),
                     "chance_val": chance(va, data)}
                if slot == "u2":
                    r["val_paired"], r["n_pairs"] = paired(pr(Xo, Mo), va, data)
                rec(r)
                if best is None or r["inner_val"] > best[0]["inner_val"]:
                    best = (r, pr(Xo, Mo))
            summary["B3plus"][slot] = best[0]
            keep_probs[("B3plus", slot)] = (best[1], va)
        # ---------------- X1a scalars + X1b candidate probes, layer by layer (each layer loaded once)
        for ell in list(layers) + ["mean_emb"]:
            Zi, Zv, Zo = data.z(ell, data.itr), data.z(ell, data.iva), data.z(ell, data.va)
            for tname, key in (("choice_is_A", "y_choice"), ("depth_is_4", "y_depth"), ("gap_above_train_median", "y_gap")):
                yi, yv, yo = (torch.tensor([data.dev.rows[i][key] for i in ids]) for ids in (data.itr, data.iva, data.va))
                maj = float(max(yi.mean(), 1 - yi.mean()))
                maj_val = float(yo.mean()) if yi.mean() >= 0.5 else float(1 - yo.mean())
                best = None
                for lam in (1e-3, 1e-1, 10.0):
                    w = logistic(Zi, yi, lam)
                    meter.tick(f"X1a|{tname}|{ell}|{lam}")
                    r = {"block": "X1a", "target": tname, "vector": ell, "lambda": lam, "train": logistic_acc(w, Zi, yi),
                         "inner_val": logistic_acc(w, Zv, yv), "val": logistic_acc(w, Zo, yo), "majority_train": maj, "majority_val": maj_val}
                    rec(r)
                    if best is None or r["inner_val"] > best["inner_val"]:
                        best = r
                summary["X1a"].setdefault(tname, {})[str(ell)] = best
            if ell in x1b_layers:
                c_all = {k: cand_tensor(data, ids) for k, ids in (("i", data.itr), ("v", data.iva), ("o", data.va))}
                for slot in SLOTS:
                    itr, iva, va = keep(data, data.itr, slot), keep(data, data.iva, slot), keep(data, data.va, slot)
                    ki, kv, ko = ix(data.itr, itr), ix(data.iva, iva), ix(data.va, va)
                    best = None
                    for lam in (1e-2, 1e-1):
                        fits = [H.train_bilinear(Zi[ki], c_all["i"][ki], ytensor(data, itr, slot), Zv[kv], c_all["v"][kv],
                                                 ytensor(data, iva, slot), Tt, 8, lam, s, max_epochs=max_epochs) for s in seeds]
                        for s in seeds:
                            meter.tick(f"X1b|{slot}|{ell}|{lam}|{s}")
                        pi, pv, po = (H.predict_proba(fits, Z[k], c_all[n][k], Tt) for Z, k, n in ((Zi, ki, "i"), (Zv, kv, "v"), (Zo, ko, "o")))
                        r = {"block": "X1b", "slot": slot, "vector": ell, "r": 8, "lambda": lam, "epochs": [f.best_epoch for f in fits],
                             "hit_epoch_cap": any(f.best_epoch >= max_epochs for f in fits),
                             "train": acc(pi, itr, data, slot), "inner_val": acc(pv, iva, data, slot), "val": acc(po, va, data, slot),
                             "chance_val": chance(va, data)}
                        if slot == "u2":
                            r["val_paired"], r["n_pairs"] = paired(po, va, data)
                        rec(r)
                        if best is None or r["inner_val"] > best[0]["inner_val"]:
                            best = (r, po)
                    summary["X1b"].setdefault(slot, {})[str(ell)] = best[0]
                    if ell == x7_vector:
                        keep_probs[("X1b", slot)] = (best[1], va)
        # ---------------- X2 capacity / regularisation (targets u2, c1)
        for ell in x2_layers:
            Zi, Zv, Zo = data.z(ell, data.itr), data.z(ell, data.iva), data.z(ell, data.va)
            for slot in ("u2", "c1"):
                itr, iva, va = keep(data, data.itr, slot), keep(data, data.iva, slot), keep(data, data.va, slot)
                ki, kv, ko = ix(data.itr, itr), ix(data.iva, iva), ix(data.va, va)
                ci, cv, co = cand_tensor(data, itr), cand_tensor(data, iva), cand_tensor(data, va)
                yi, yv = ytensor(data, itr, slot), ytensor(data, iva, slot)

                def report(cfg, pi, pv, po, epochs):
                    r = dict(cfg, block="X2", slot=slot, vector=ell, epochs=epochs, hit_epoch_cap=any(e >= max_epochs for e in epochs),
                             train=acc(pi, itr, data, slot), inner_val=acc(pv, iva, data, slot), val=acc(po, va, data, slot),
                             chance_val=chance(va, data))
                    if slot == "u2":
                        r["val_paired"], r["n_pairs"] = paired(po, va, data)
                    rec(r)
                    summary["X2"].setdefault(slot, []).append(r)
                for rr in (2, 8, 32):
                    for lam in (1e-1, 1.0, 10.0):
                        fits = [H.train_bilinear(Zi[ki], ci, yi, Zv[kv], cv, yv, Tt, rr, lam, s, max_epochs=max_epochs) for s in seeds]
                        for s in seeds:
                            meter.tick(f"X2|bil|{slot}|{ell}|{rr}|{lam}|{s}")
                        report({"model": "bilinear", "r": rr, "lambda": lam},
                               *(H.predict_proba(fits, Z[k], c, Tt) for Z, k, c in ((Zi, ki, ci), (Zv, kv, cv), (Zo, ko, co))),
                               [f.best_epoch for f in fits])
                for kk in (16, 64, 256):
                    kq = min(kk, Zi.shape[1] - 1)
                    P = PCAProj(Zi[ki], kq)
                    Pi, Pv, Po = P(Zi[ki]), P(Zv[kv]), P(Zo[ko])
                    sd = Pi.std(0).clamp(min=1e-6)
                    Pi, Pv, Po = Pi / sd, Pv / sd, Po / sd
                    fits = [H.train_bilinear(Pi, ci, yi, Pv, cv, yv, Tt, 8, 1e-1, s, max_epochs=max_epochs) for s in seeds]
                    for s in seeds:
                        meter.tick(f"X2|pca|{slot}|{ell}|{kk}|{s}")
                    report({"model": "pca_bilinear", "k": kq, "r": 8, "lambda": 1e-1},
                           *(H.predict_proba(fits, X, c, Tt) for X, c in ((Pi, ci), (Pv, cv), (Po, co))), [f.best_epoch for f in fits])
                kz = min(64, Zi.shape[1] - 1)
                Pz = PCAProj(Zi[ki], kz)
                labs = sorted({int(x) for x in ci[ci >= 0].tolist()})
                Pe = PCAProj(Tt[labs], min(64, Tt.shape[1] - 1, len(labs) - 1))
                Ep = Pe(Tt)
                Ep = Ep / Ep[labs].std(0).clamp(min=1e-6)
                zs = Pz(Zi[ki]).std(0).clamp(min=1e-6)
                Qi, Qv, Qo = Pz(Zi[ki]) / zs, Pz(Zv[kv]) / zs, Pz(Zo[ko]) / zs
                preds, eps = [], []
                for s in seeds:
                    p, e = train_mlp(Qi, ci, yi, Qv, cv, yv, Ep, 64, 1e-2, s, max_epochs=max_epochs)
                    preds.append(p)
                    eps.append(e)
                    meter.tick(f"X2|mlp|{slot}|{ell}|{s}")
                avg = lambda X, c: torch.stack([p(X, c) for p in preds]).mean(0)
                report({"model": "mlp", "pca_z": kz, "pca_e": Ep.shape[1], "hidden": 64}, avg(Qi, ci), avg(Qv, cv), avg(Qo, co), eps)
        # ---------------- learning curves: inner-val-best X2 bilinear/pca configuration per target
        for slot in ("u2", "c1"):
            cands = [r for r in summary["X2"][slot] if r["model"] in ("bilinear", "pca_bilinear")]
            b = max(cands, key=lambda r: (r["inner_val"], -r.get("r", 0)))
            ell = b["vector"]
            Zi, Zv, Zo = data.z(ell, data.itr), data.z(ell, data.iva), data.z(ell, data.va)
            iva, va = keep(data, data.iva, slot), keep(data, data.va, slot)
            kv, ko = ix(data.iva, iva), ix(data.va, va)
            cv, co = cand_tensor(data, iva), cand_tensor(data, va)
            for frac in (0.25, 0.5, 1.0):
                fams = set(np.random.default_rng(INNER_SEED + 1).permutation(data.itr_fams)[: max(2, round(frac * len(data.itr_fams)))].tolist())
                itr = [i for i in keep(data, data.itr, slot) if data.dev.rows[i]["family_id"] in fams]
                ki = ix(data.itr, itr)
                ci = cand_tensor(data, itr)
                Xi, Xv, Xo = Zi[ki], Zv[kv], Zo[ko]
                if b["model"] == "pca_bilinear":
                    P = PCAProj(Xi, min(b["k"], len(itr) - 1))
                    Xi, Xv, Xo = P(Xi), P(Xv), P(Xo)
                    sd = Xi.std(0).clamp(min=1e-6)
                    Xi, Xv, Xo = Xi / sd, Xv / sd, Xo / sd
                fits = [H.train_bilinear(Xi, ci, ytensor(data, itr, slot), Xv, cv, ytensor(data, iva, slot), Tt, b["r"], b["lambda"], s,
                                         max_epochs=max_epochs) for s in seeds]
                for s in seeds:
                    meter.tick(f"LC|{slot}|{frac}|{s}")
                pi, pv, po = H.predict_proba(fits, Xi, ci, Tt), H.predict_proba(fits, Xv, cv, Tt), H.predict_proba(fits, Xo, co, Tt)
                r = {"block": "LC", "slot": slot, "config": {k: b[k] for k in ("model", "vector", "r", "lambda") if k in b} | ({"k": b["k"]} if "k" in b else {}),
                     "fraction": frac, "n_train_worlds": len(itr), "n_train_families": len(fams), "train": acc(pi, itr, data, slot),
                     "inner_val": acc(pv, iva, data, slot), "val": acc(po, va, data, slot), "epochs": [f.best_epoch for f in fits]}
                if slot == "u2":
                    r["val_paired"], r["n_pairs"] = paired(po, va, data)
                rec(r)
                summary["LC"].setdefault(slot, []).append(r)
        # ---------------- X7 strata (thresholds fixed in PLAN.json before any fit)
        def tertile(m):
            return 0 if m <= tert[0] else 1 if m <= tert[1] else 2
        for (src, slot), (prob, ids) in keep_probs.items():
            if slot not in ("u2", "c1"):
                continue
            pred = H.argmax_canonical(prob).numpy()
            cells = {}
            for k, i in enumerate(ids):
                r = data.dev.rows[i]
                cells.setdefault(f"conf{tertile(r['margin'])}_d{r['depth']}", []).append(float(pred[k] == r["tgt"][slot]))
            summary["X7"][f"{src}|{slot}"] = {c: {"n": len(v), "acc": float(np.mean(v)), "wilson95": wilson(int(sum(v)), len(v))}
                                              for c, v in sorted(cells.items())}
            rec({"block": "X7", "source": src, "slot": slot, "cells": summary["X7"][f"{src}|{slot}"]})
        adm = {}
        for w in data.val_all:
            adm.setdefault(f"conf{tertile(w['margin'])}_d{w['depth']}", []).append(float(w["admissible"]))
        summary["X7"]["admissibility_val_all_nondecoy"] = {c: {"n": len(v), "rate": float(np.mean(v))} for c, v in sorted(adm.items())}
        rec({"block": "X7", "source": "admissibility", "cells": summary["X7"]["admissibility_val_all_nondecoy"]})
    except H.BudgetExceeded as e:
        (out / "BUDGET_EXCEEDED.json").write_text(json.dumps({"error": str(e), "partial_summary_keys": list(summary)}, indent=2), encoding="utf-8")
        raise SystemExit(f"BUDGET EXCEEDED: {e}")
    finally:
        log.close()
    summary["cpu_hours_used"] = meter.used_s() / 3600
    summary["n_fits"] = meter.n_fits
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--synthetic-test", default="")
    a = ap.parse_args()
    if a.synthetic_test:
        cap, devdir = T.make_synthetic(Path(a.synthetic_test), n_worlds=480, d=16)
        s = run(Data(T.Dev(cap, devdir), devdir), Path(a.synthetic_test) / "out", layers=(0, 12), x1b_layers=(12,), x2_layers=(12,),
                seeds=(0,), max_epochs=5, x7_vector=12)
        print(json.dumps({k: (v if k in ("cpu_hours_used", "n_fits") else list(v)) for k, v in s.items()}, indent=1))
        return
    if not a.run:
        raise SystemExit("use --run (single authorised block) or --synthetic-test")
    dev = T.Dev(T.REAL["capture"], T.REAL["dev"])
    s = run(Data(dev, T.REAL["dev"]), ROOT / "outputs" / "h1b")
    print(json.dumps({"cpu_hours_used": s["cpu_hours_used"], "n_fits": s["n_fits"]}, indent=1))


if __name__ == "__main__":
    main()
