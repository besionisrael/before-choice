"""BTC-H1b X3+X4 analysis (CPU, EXPLORATORY; PROBE-DEV only). PREPARED, NOT RUN on real data.

    python src/h1b_x3x4_analysis.py --run                    # after an authorised capture; 4 threads, 6 CPU-h cap
    python src/h1b_x3x4_analysis.py --synthetic-test <dir>   # tests only

Question: is the label-decoding failure of H1 due to the STATIC candidate keys (representation) or to the OBSERVATION
POSITION? Factorial design on the SAME worlds, SAME family splits (h1b_explore inner/outer), SAME metrics:

  representation of the candidates  R in {static (frozen e(c), as H1), ctx_tgt (model state at the label's arrival
                                    occurrence), ctx_src (state at its departure occurrence)}; key layer = query layer
  query position                    Q in {ANS (<channel|>), Q_END, ACT_END_A, ACT_END_B, LAST_RULE_END}
  layer                             9 layers {0, 8, 16, 24, 32, 40, 48, 56, 61}
  decoder                           the H1 bilinear form s(c) = z_Q^T U V^T k_c (r = 8, lambda = 0.1, seeds 0, 1, 2),
                                    early stopping on inner-val; every choice on inner-val; DEV-val reported only
  slots                             A1, c1 (positive controls), c2, u2 (+ all slots at the selected cell)

Controls against shortcuts: (i) KEY-ONLY decoder s(c) = w^T k_c (no query) for each contextual R and layer;
(ii) CHOICE+KEYS decoder s(c) = w_{choice}^T k_c (the model's chosen letter + the candidates' contextual keys, no
query activation): measures how much of c1 is reconstructible from the choice and the candidates' context alone;
(iii) layer 0 keys = token embedding of the label's last token (purely lexical); (iv) prompt baseline B3+;
(v) PRE-REWARD query positions (ACT_END_A/B, LAST_RULE_END) are analysed as a control: their performance is MEASURED
and reported, with no automatic causal interpretation (owner amendment, 2026-11-14); (vi) candidates and positions
from the prompt only; no oracle input; canonical candidate order.
Missing positions (declared): depth-3 worlds have 4 candidates (slots 9-12 masked) and 4 rule lines (slots 19-20
masked); masked candidate slots receive probability 0 (PAD), LAST_RULE_END = last valid rule slot; c3/u3 exist at
depth 4 only and are evaluated on depth-4 worlds; no other position can be missing (audit 3,480/3,480) and any
unexpected missing required position raises (never imputed).
Reporting: c1 and A1 accuracies at the selected cell are also reported separately for model choice A and choice B.

Pre-declared instrument-sensitivity rule (on DEV-val, cell chosen on inner-val by c1 accuracy among R x Q x layer):
  S-A  c1 accuracy >= 0.80;   S-B  c1 - key-only(c1; same R, layer) >= 0.25;   S-C  A1 accuracy >= 0.80 (same cell).
  All three -> "SENSITIVE": u2 at that cell is reported against B3+ and key-only (single + paired) and is considered
  informative about E1. Otherwise -> "NOT SENSITIVE": u2 is reported but declared uninformative.
  The thresholds are unchanged by the 2026-11-14 amendments; the CHOICE+KEYS control accompanies the interpretation.
Precondition of --run: the capture's replication check reports ALL pre-answer vectors bitwise identical to D3-DEV-R1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import d3x_positions as XP  # noqa: E402
import h1_probe as H  # noqa: E402
import h1_train_dev as T  # noqa: E402
import h1b_explore as X  # noqa: E402

THREADS, CPU_HOURS = 4, 6.0
REPS = ("static", "ctx_tgt", "ctx_src")
QUERIES = ("ANS", "Q_END", "ACT_END_A", "ACT_END_B", "LAST_RULE_END")
SLOTS_GRID = ("A1", "c1", "c2", "u2")
SLOTS_ALL = ("A1", "c1", "c2", "c3", "u1", "u2", "u3")
CRIT = {"S-A_c1_min": 0.80, "S-B_c1_minus_keyonly_min": 0.25, "S-C_A1_min": 0.80}
PLAN = {"status": "EXPLORATORY; PROBE-DEV only (already used for H1 selection); LOCKED-TEST not accessed",
        "factorial": {"representation": REPS, "query": QUERIES, "layers": list(XP.LAYERS), "slots": SLOTS_GRID},
        "decoder": {"form": "bilinear z^T U V^T k", "r": 8, "lambda": 0.1, "seeds": [0, 1, 2], "early_stopping": "inner-val"},
        "key_only": {"reps": ["ctx_tgt", "ctx_src"], "lambda_grid": [1e-2, 1.0, 100.0], "choice": "inner-val"},
        "choice_plus_keys": {"reps": ["ctx_tgt", "ctx_src"], "form": "w_A^T k 1[choice=A] + w_B^T k 1[choice=B]",
                             "lambda_grid": [1e-2, 1.0, 100.0], "choice": "inner-val"},
        "pre_reward_queries": "ACT_END_A, ACT_END_B, LAST_RULE_END: measured as controls, no automatic causal reading",
        "missing_positions": "depth-3 masks (slots 9-12, 19-20); PAD -> probability 0; never imputed; unexpected gaps raise",
        "per_choice_report": "c1 and A1 at the selected cell split by model choice A / B",
        "baselines": ["B3+ (h1b_explore)", "chance"], "sensitivity_rule": CRIT,
        "selected_cell_extra": {"slots": SLOTS_ALL, "seeds": [0, 1, 2]},
        "precondition": "replication check: all ANS vectors bitwise identical to D3-DEV-R1",
        "splits": "h1b_explore inner/outer family splits (identical worlds and partitions)",
        "budget": {"cpu_hours": CPU_HOURS, "threads": THREADS}}


class XData:
    """Multi-position activations aligned with h1b_explore.Data rows."""

    def __init__(self, data: X.Data, xdir: Path, positions: Path):
        self.d = data
        man = json.loads((xdir / "MANIFEST.json").read_text(encoding="utf-8"))
        self.layers = list(man["layers"])
        self.files, self.pos = [], {}
        for s in man["shards"]:
            ids = [json.loads(x)["world_id"] for x in (xdir / f"{s['name']}_index.jsonl").read_text(encoding="utf-8").splitlines()]
            self.files.append((xdir / f"{s['name']}_acts.npy", ids))
        self.plan = {r["world_id"]: r for r in map(json.loads, positions.read_text(encoding="utf-8").splitlines())}
        for r in data.dev.rows:
            assert self.plan[r["world_id"]]["cands"] == r["cands"], "candidate order differs between plan and rows"
        self.cache = {}

    def load_layer(self, ell: int):
        """{world_id: (P, d) float32} for one layer (all slots)."""
        if ell in self.cache:
            return self.cache[ell]
        self.cache.clear()
        j = self.layers.index(ell)
        out = {}
        for path, ids in self.files:
            a = np.load(path, mmap_mode="r")
            for k, w in enumerate(ids):
                out[w] = u16f(np.asarray(a[k, :, j]))
        self.cache[ell] = out
        return out


def u16f(a):
    return (a.astype(np.uint32) << 16).view(np.float32)


def query_slot(rec, q):
    if q == "ANS":
        return 0
    if q == "Q_END":
        return 21
    if q == "ACT_END_A":
        return 13
    if q == "ACT_END_B":
        return 14
    return max(k for k in range(15, 21) if rec["mask"][k])


def build_inputs(xd: XData, ell: int, ids, q: str, rep: str):
    """z (N, d) from query slot; per-world key table rows for contextual reps; cand tensor indexing the key table."""
    L = xd.load_layer(ell)
    rows = [xd.d.dev.rows[i] for i in ids]
    z = np.stack([L[r["world_id"]][query_slot(xd.plan[r["world_id"]], q)] for r in rows]).astype(np.float32)
    if rep == "static":
        return z, xd.d.dev.cand_tensor(ids), None
    off = 1 if rep == "ctx_tgt" else 2
    keys, cand = [], torch.full((len(ids), H.K), H.PAD, dtype=torch.long)
    for k, r in enumerate(rows):
        a = L[r["world_id"]]
        for j in range(len(r["cands"])):
            cand[k, j] = len(keys)
            keys.append(a[off + 2 * j])
    return z, cand, np.stack(keys).astype(np.float32)


def standardise_fit(train):
    mu, sd = train.mean(0), train.std(0)
    return mu, np.where(sd < 1e-6, 1.0, sd)


def run(data: X.Data, xd: XData, out: Path, layers=None, queries=QUERIES, reps=REPS, slots=SLOTS_GRID, seeds=(0, 1, 2),
        max_epochs=H.MAX_EPOCHS) -> dict:
    if (out / "results.jsonl").exists():
        raise SystemExit("results.jsonl exists: a single analysis is authorised")
    H.set_threads(THREADS)
    layers = layers or xd.layers
    out.mkdir(parents=True, exist_ok=True)
    (out / "PLAN.json").write_text(json.dumps(dict(PLAN, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                                              indent=2) + "\n", encoding="utf-8")
    meter = H.BudgetMeter(CPU_HOURS, out / "budget.tsv")
    log = open(out / "results.jsonl", "a", encoding="utf-8")
    rec = lambda d: (log.write(json.dumps(dict(d, cpu_s=meter.used_s())) + "\n"), log.flush())
    res = {"grid": [], "key_only": [], "choice_keys": [], "B3plus": {}}
    Tstatic = data.table()

    def fit_cell(ell, q, rep, slot, seeds_, per_choice=False):
        itr, iva, va = (X.keep(data, ids, slot) for ids in (data.itr, data.iva, data.va))
        zi, ci, ki = build_inputs(xd, ell, itr, q, rep)
        zv, cv, kv = build_inputs(xd, ell, iva, q, rep)
        zo, co, ko = build_inputs(xd, ell, va, q, rep)
        mu, sd = standardise_fit(zi)
        Zi, Zv, Zo = (torch.from_numpy((z - mu) / sd) for z in (zi, zv, zo))
        if rep == "static":
            Ti = Tv = To = Tstatic
        else:
            km, ks = standardise_fit(ki)
            Ti, Tv, To = (torch.from_numpy((k - km) / ks) for k in (ki, kv, ko))
        yi, yv = X.ytensor(data, itr, slot), X.ytensor(data, iva, slot)
        fits = []
        for s in seeds_:
            if rep == "static":
                fits.append(H.train_bilinear(Zi, ci, yi, Zv, cv, yv, Ti, 8, 0.1, s, max_epochs=max_epochs))
            else:
                fits.append(_train_two_tables(Zi, ci, yi, Ti, Zv, cv, yv, Tv, 8, 0.1, s, max_epochs))
            meter.tick(f"cell|{ell}|{q}|{rep}|{slot}|{s}")
        pi, pv, po = (H.predict_proba(fits, Z, c, Tt) for Z, c, Tt in ((Zi, ci, Ti), (Zv, cv, Tv), (Zo, co, To)))
        r = {"block": "grid", "layer": ell, "query": q, "rep": rep, "slot": slot, "epochs": [f.best_epoch for f in fits],
             "train": X.acc(pi, itr, data, slot), "inner_val": X.acc(pv, iva, data, slot), "val": X.acc(po, va, data, slot),
             "chance_val": X.chance(va, data)}
        if slot == "u2":
            r["val_paired"], r["n_pairs"] = X.paired(po, va, data)
        if per_choice:
            for L_ in ("A", "B"):
                k_ = [k for k, i in enumerate(va) if data.dev.rows[i]["choice"] == L_]
                r[f"val_choice_{L_}"] = X.acc(po[k_], [va[k] for k in k_], data, slot)
                r[f"n_val_choice_{L_}"] = len(k_)
        rec(r)
        return r

    def key_only(ell, rep, slot, with_choice=False):
        itr, iva, va = (X.keep(data, ids, slot) for ids in (data.itr, data.iva, data.va))
        _, ci, ki = build_inputs(xd, ell, itr, "ANS", rep)
        _, cv, kv = build_inputs(xd, ell, iva, "ANS", rep)
        _, co, ko = build_inputs(xd, ell, va, "ANS", rep)
        km, ks = standardise_fit(ki)
        def tens(c, k, ids):
            K = torch.from_numpy((k - km) / ks)
            Xk = torch.zeros(len(c), H.K, K.shape[1])
            M = c == H.PAD
            Xk[~M] = K[c[~M]]
            if with_choice:   # separate weights for model choice A / B (choice = model output, not oracle)
                a = torch.tensor([float(data.dev.rows[i]["choice"] == "A") for i in ids]).view(-1, 1, 1)
                Xk = torch.cat([Xk * a, Xk * (1 - a)], -1)
            return Xk, M
        (Xi, Mi), (Xv, Mv), (Xo, Mo) = tens(ci, ki, itr), tens(cv, kv, iva), tens(co, ko, va)
        best = None
        for lam in (1e-2, 1.0, 100.0):
            tag = "choice_keys" if with_choice else "key_only"
            w = H.train_conditional_logit(Xi, Mi, X.ytensor(data, itr, slot), lam, meter, f"{tag}|{ell}|{rep}|{slot}|{lam}")
            pr = lambda A, M: torch.softmax((A @ w).masked_fill(M, float("-inf")), -1)
            r = {"block": tag, "layer": ell, "rep": rep, "slot": slot, "lambda": lam, "train": X.acc(pr(Xi, Mi), itr, data, slot),
                 "inner_val": X.acc(pr(Xv, Mv), iva, data, slot), "val": X.acc(pr(Xo, Mo), va, data, slot)}
            if slot == "u2":
                r["val_paired"], r["n_pairs"] = X.paired(pr(Xo, Mo), va, data)
            rec(r)
            if best is None or r["inner_val"] > best["inner_val"]:
                best = r
        return best

    try:
        for slot in SLOTS_ALL:   # prompt baseline (same as H1b, recomputed for the record)
            itr, iva, va = (X.keep(data, ids, slot) for ids in (data.itr, data.iva, data.va))
            (Xi, Mi), (Xv, Mv), (Xo, Mo) = X.b3plus(data, itr), X.b3plus(data, iva), X.b3plus(data, va)
            best = None
            for lam in (1e-4, 1e-2, 1.0):
                w = H.train_conditional_logit(Xi, Mi, X.ytensor(data, itr, slot), lam, meter, f"B3plus|{slot}|{lam}")
                pr = lambda A, M: torch.softmax((A @ w).masked_fill(M, float("-inf")), -1)
                r = {"block": "B3plus", "slot": slot, "lambda": lam, "inner_val": X.acc(pr(Xv, Mv), iva, data, slot), "val": X.acc(pr(Xo, Mo), va, data, slot)}
                if slot == "u2":
                    r["val_paired"], r["n_pairs"] = X.paired(pr(Xo, Mo), va, data)
                rec(r)
                if best is None or r["inner_val"] > best["inner_val"]:
                    best = r
            res["B3plus"][slot] = best
        for ell in layers:
            for slot in slots:
                for rep in reps:
                    if rep != "static":
                        res["key_only"].append(key_only(ell, rep, slot))
                        res["choice_keys"].append(key_only(ell, rep, slot, with_choice=True))
                    for q in queries:
                        res["grid"].append(fit_cell(ell, q, rep, slot, seeds))
        # ---- pre-declared sensitivity decision
        c1 = [r for r in res["grid"] if r["slot"] == "c1"]
        cell = max(c1, key=lambda r: (r["inner_val"], -r["layer"]))
        sel = {k: cell[k] for k in ("layer", "query", "rep")}
        a1 = next(r for r in res["grid"] if r["slot"] == "A1" and all(r[k] == sel[k] for k in sel))
        ko = next((r for r in res["key_only"] if r["slot"] == "c1" and r["layer"] == sel["layer"] and r["rep"] == sel["rep"]), None)
        ko_val = ko["val"] if ko else 1 / 2   # static keys: no key-only decoder (random labels); 1/2 = two action-line labels
        crit = {"S-A": cell["val"] >= CRIT["S-A_c1_min"], "S-B": cell["val"] - ko_val >= CRIT["S-B_c1_minus_keyonly_min"],
                "S-C": a1["val"] >= CRIT["S-C_A1_min"]}
        res["selected_cell"] = dict(sel, c1_val=cell["val"], c1_keyonly_val=ko_val, A1_val=a1["val"], criteria=crit,
                                    verdict="SENSITIVE" if all(crit.values()) else "NOT SENSITIVE")
        # ---- all slots at the selected cell (3 seeds), u2 vs B3+ and key-only
        res["selected_all_slots"] = {s: fit_cell(sel["layer"], sel["query"], sel["rep"], s, seeds, per_choice=True) for s in SLOTS_ALL}
        if sel["rep"] != "static":
            res["selected_keyonly_u2"] = key_only(sel["layer"], sel["rep"], "u2")
            res["selected_choice_keys"] = {s: key_only(sel["layer"], sel["rep"], s, with_choice=True) for s in ("c1", "A1", "u2")}
        ck = next((r for r in res["choice_keys"] if r["slot"] == "c1" and r["layer"] == sel["layer"] and r["rep"] == sel["rep"]), None)
        res["selected_cell"]["c1_choice_keys_val"] = ck["val"] if ck else None
        rec({"block": "decision", **res["selected_cell"]})
    except H.BudgetExceeded as e:
        (out / "BUDGET_EXCEEDED.json").write_text(json.dumps({"error": str(e)}), encoding="utf-8")
        raise SystemExit(f"BUDGET EXCEEDED: {e}")
    finally:
        log.close()
    res["cpu_hours_used"] = meter.used_s() / 3600
    (out / "SUMMARY.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    return res


def _train_two_tables(Zi, ci, yi, Ti, Zv, cv, yv, Tv, r, wd, seed, max_epochs):
    """H1 bilinear form with per-world key tables (train and inner-val tables differ). Same optimiser/early stopping."""
    import math
    g = torch.Generator().manual_seed(seed)
    d, dk = Zi.shape[1], Ti.shape[1]
    U = (torch.randn(d, r, generator=g) / math.sqrt(d)).requires_grad_()
    V = (torch.randn(dk, r, generator=g) / math.sqrt(dk)).requires_grad_()
    opt = torch.optim.AdamW([U, V], lr=H.LR, weight_decay=wd)
    best, best_ep, bad, state = math.inf, 0, 0, (U.detach().clone(), V.detach().clone())
    for ep in range(1, max_epochs + 1):
        perm = torch.randperm(len(Zi), generator=g)
        for i in range(0, len(perm), H.BATCH):
            b = perm[i:i + H.BATCH]
            loss = torch.nn.functional.cross_entropy(H._scores(Zi[b], ci[b], Ti, U, V), yi[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            vl = float(torch.nn.functional.cross_entropy(H._scores(Zv, cv, Tv, U, V), yv))
        if vl < best - 1e-6:
            best, best_ep, bad, state = vl, ep, 0, (U.detach().clone(), V.detach().clone())
        else:
            bad += 1
            if bad >= H.PATIENCE:
                break
    return H.Fit(state[0], state[1], best_ep, best)


# ------------------------------------------------------------------ synthetic data (tests only)
def make_synthetic_x(base: Path, n_worlds=480, d=16, signal=4.0, seed=0):
    """Fake multi-position shards on synthetic DEV rows: each candidate's ARRIVAL key = a random per-world vector g;
    the ANS / Q_END query carries g of the chosen depth-1 label (c1) -> a pointer decoder must find c1, static keys
    cannot. Positions plan without tokenizer."""
    cap, devdir = T.make_synthetic(base / "syn", n_worlds=n_worlds, d=d)
    dev = T.Dev(cap, devdir)
    data = X.Data(dev, devdir)
    rng = np.random.default_rng(seed)
    xdir = base / "xshards"
    xdir.mkdir(parents=True)
    L = len(XP.LAYERS)
    plan, acts, idx = [], [], []
    for r in dev.rows:
        P = XP.P
        mask = [True] + [j < len(r["cands"]) for j in range(6) for _ in (0, 1)] + [True, True] + [k < len(r["cands"]) for k in range(6)] + [True]
        a = np.zeros((P, L, d), np.float32)
        a[mask] = rng.standard_normal((sum(mask), L, d))
        g = rng.standard_normal((len(r["cands"]), d))
        for j in range(len(r["cands"])):
            a[1 + 2 * j] += signal * g[j]
        c1 = r["tgt"]["c1"]
        a[0] += signal * g[c1]
        a[21] += signal * g[c1]
        acts.append(torch.from_numpy(a).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16))
        plan.append({"world_id": r["world_id"], "cands": r["cands"], "mask": mask, "tok": list(range(P))})
        idx.append({"world_id": r["world_id"], "mask": mask})
    np.save(xdir / "xshard_000_acts.npy", np.stack(acts))
    (xdir / "xshard_000_index.jsonl").write_text("".join(json.dumps(x) + "\n" for x in idx), encoding="utf-8")
    (xdir / "MANIFEST.json").write_text(json.dumps({"layers": list(XP.LAYERS), "shards": [{"name": "xshard_000", "n": len(acts), "files": {}}]}), encoding="utf-8")
    (base / "positions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in plan), encoding="utf-8")
    return data, xdir, base / "positions.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--synthetic-test", default="")
    a = ap.parse_args()
    if a.synthetic_test:
        data, xdir, pos = make_synthetic_x(Path(a.synthetic_test))
        s = run(data, XData(data, xdir, pos), Path(a.synthetic_test) / "out", layers=[8], queries=("ANS", "ACT_END_A"),
                seeds=(0,), max_epochs=20)
        print(json.dumps(s["selected_cell"], indent=1))
        return
    if not a.run:
        raise SystemExit("use --run (after an authorised capture) or --synthetic-test")
    rm = json.loads((ROOT / "outputs" / "d3x" / "probe_dev" / "x_run_manifest.json").read_text(encoding="utf-8"))
    rep = rm["replication_vs_D3_DEV_R1"]
    if rep["bitwise_equal"] != rep["n_worlds"] or rep["n_worlds"] != 3480:
        raise SystemExit(f"PRECONDITION FAILED: ANS vectors not identical to D3-DEV-R1: {rep}")
    dev = T.Dev(T.REAL["capture"], T.REAL["dev"])
    data = X.Data(dev, T.REAL["dev"])
    xd = XData(data, ROOT / "outputs" / "d3x" / "probe_dev", ROOT / "outputs" / "d3x" / "plan" / "positions.jsonl")
    s = run(data, xd, ROOT / "outputs" / "h1b_x3x4")
    print(json.dumps(s["selected_cell"], indent=1))


if __name__ == "__main__":
    main()
