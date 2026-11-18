"""BTC-H1c instrument diagnostic (CPU; POST HOC / EXPLORATORY; existing H1c DEV activations only).

    python src/h1c_instrument_diagnostic.py   -> outputs/h1c_dev/diagnostic/DIAGNOSTIC.json   (1 CPU-h cap, 4 threads)

Nothing here changes the frozen protocol or the historical DEV result. Same splits as the DEV analysis (inner-train /
inner-val from DEV-train families; DEV-val families reported). Decoders: pointer s(c) = z_ANS^T U V^T k_c, r = 8,
lambda = 0.1, seeds 0,1,2, early stopping on inner-val (no new hyper-parameter search).
  D1  PC1 (12-way c1) at each of the 5 captured layers: train / inner-val / DEV-val
  D2  binary unchosen continuation (P) and binary chosen continuation (PC2) at each layer: train / inner-val / DEV-val
  D3  c1 reconstruction without the query: choice x keys (linear, L-BFGS) and a text baseline (candidate printed in the
      action line of the chosen letter) -> effective decoding vs reconstruction
  D4  learning curve of PC1 at the inner-val-best layer (25 / 50 / 100 % of inner-train families) and role confusion of
      PC1 errors on DEV-val (which role the wrong prediction had: c2, c3, chosen dead-end, u1, ...)
  D5  layer-56 effect: PC1 at 56 vs at the inner-val-best layer
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_h1c as G  # noqa: E402
import h1_probe as H  # noqa: E402
import h1c_dev_analysis as A  # noqa: E402
from h1b_x3x4_analysis import _train_two_tables  # noqa: E402

R, WD, SEEDS = 8, 0.1, (0, 1, 2)


def roles(w, chosen):
    non = "B" if chosen == "A" else "A"
    o = w["oracle"]
    out = {}
    for tag, a in (("c", chosen), ("u", non)):
        p = o["paths"][a]
        for k in (1, 2, 3):
            out[p[k]] = f"{tag}{k}"
        nxt = {e["source"]: e["target"] for e in w["transitions"] if "action" not in e and e["source"] != p[1]}
        e1 = o["dead_ends"][a][0]
        e2 = nxt.get(e1)
        e3 = nxt.get(e2)
        for k, lab in ((1, e1), (2, e2), (3, e3)):
            if lab:
                out[lab] = f"{tag}e{k}"
    return out


def c1_inputs(cap, ell, ids):
    L = cap.layer(ell)
    z = np.stack([L[cap.rows[k]["world_id"]][0] for k in ids])
    keys = np.concatenate([L[cap.rows[k]["world_id"]][1:13] for k in ids])
    return z, keys, torch.arange(len(ids) * 12).view(len(ids), 12), torch.tensor([cap.rows[k]["c1_slot"] for k in ids])


def acc(prob, y):
    return float((H.argmax_canonical(prob) == y).float().mean())


def fit_eval(inp_fn, cap, ell, train_ids, seeds=SEEDS):
    a, b, o = inp_fn(cap, ell, train_ids), inp_fn(cap, ell, cap.iva), inp_fn(cap, ell, cap.va)
    Za, Zb, Zo = A.std(a[0], b[0], o[0])
    Ka, Kb, Ko = A.std(a[1], b[1], o[1])
    fits = [_train_two_tables(Za, a[2], a[3], Ka, Zb, b[2], b[3], Kb, R, WD, s, H.MAX_EPOCHS) for s in seeds]
    pa, pb, po = H.predict_proba(fits, Za, a[2], Ka), H.predict_proba(fits, Zb, b[2], Kb), H.predict_proba(fits, Zo, o[2], Ko)
    return {"train": acc(pa, a[3]), "inner_val": acc(pb, b[3]), "val": acc(po, o[3]), "epochs": [f.best_epoch for f in fits]}, po, o


def main():
    H.set_threads(4)
    meter = H.BudgetMeter(1.0)
    out_dir = ROOT / "outputs" / "h1c_dev" / "diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = A.Cap(ROOT / "outputs" / "h1c_dev" / "capture", ROOT / "data" / "h1c" / "dev", None)
    res = {"status": "POST HOC / EXPLORATORY", "n": {"inner_train": len(cap.itr), "inner_val": len(cap.iva), "val": len(cap.va)},
           "D1_PC1_by_layer": {}, "D2_binary_by_layer": {}}
    for ell in cap.layers:
        r, _, _ = fit_eval(c1_inputs, cap, ell, cap.itr)
        res["D1_PC1_by_layer"][str(ell)] = r
        meter.tick(f"PC1|{ell}")
        bu, _, _ = fit_eval(lambda c, l, ids: A.binary_inputs(c, l, ids), cap, ell, cap.itr)
        bc, _, _ = fit_eval(lambda c, l, ids: A.binary_inputs(c, l, ids, chosen=True), cap, ell, cap.itr)
        res["D2_binary_by_layer"][str(ell)] = {"unchosen_P": bu, "chosen_PC2": bc}
        meter.tick(f"bin|{ell}")
        print(ell, r, bu["val"], bc["val"], flush=True)
    best = max(cap.layers, key=lambda l: (res["D1_PC1_by_layer"][str(l)]["inner_val"], -l))
    res["D5_layer_effect"] = {"inner_val_best_layer_for_PC1": best, "PC1_val_at_best": res["D1_PC1_by_layer"][str(best)]["val"],
                              "PC1_val_at_56": res["D1_PC1_by_layer"]["56"]["val"]}
    # D3: c1 reconstruction without the query (choice x keys; text baseline)
    L = best
    a, o = c1_inputs(cap, L, cap.itr), c1_inputs(cap, L, cap.va)
    Ka, Ko = A.std(a[1], o[1])
    chA = torch.tensor([float(cap.rows[k]["choice"] == "A") for k in cap.itr])
    chO = torch.tensor([float(cap.rows[k]["choice"] == "A") for k in cap.va])

    def feats(K, cand, ch):
        X = K[cand]
        return torch.cat([X * ch.view(-1, 1, 1), X * (1 - ch.view(-1, 1, 1))], -1), torch.zeros(cand.shape, dtype=torch.bool)
    Xa, Ma = feats(Ka, a[2], chA)
    Xo, Mo = feats(Ko, o[2], chO)
    w = H.train_conditional_logit(Xa, Ma, a[3], 0.01, meter, "D3|choice_keys")
    ck = acc(torch.softmax(Xo @ w, -1), o[3])
    text_ok = []
    for k in cap.va:
        r = cap.rows[k]
        p = G.parse(cap.W[r["world_id"]]["prompt"])
        cands = p["cands"]
        text_ok.append(cands.index(p["act"][r["choice"]]) == r["c1_slot"])
    res["D3_c1_without_query"] = {"layer": L, "choice_x_keys_val": ck, "text_baseline_val": float(np.mean(text_ok)),
                                  "pointer_PC1_val": res["D1_PC1_by_layer"][str(L)]["val"]}
    # D4: learning curve + role confusion at the best layer
    fams = sorted({cap.rows[k]["family_id"] for k in cap.itr})
    rng = np.random.default_rng(7)
    curve = []
    for frac in (0.25, 0.5, 1.0):
        keep = set(rng.permutation(fams)[: max(2, round(frac * len(fams)))].tolist())
        ids = [k for k in cap.itr if cap.rows[k]["family_id"] in keep]
        r, po, o2 = fit_eval(c1_inputs, cap, L, ids)
        curve.append({"fraction": frac, "n_train_worlds": len(ids), **r})
        meter.tick(f"LC|{frac}")
    res["D4_learning_curve_PC1"] = curve
    _, po, o3 = fit_eval(c1_inputs, cap, L, cap.itr)
    pred = H.argmax_canonical(po).numpy()
    conf = Counter()
    for j, k in enumerate(cap.va):
        if pred[j] == o3[3][j]:
            continue
        rr = cap.rows[k]
        w = cap.W[rr["world_id"]]
        cands = G.parse(w["prompt"])["cands"]
        conf[roles(w, rr["choice"]).get(cands[pred[j]], "?")] += 1
    res["D4_PC1_error_roles_val"] = dict(conf.most_common())
    res["cpu_hours"] = meter.used_s() / 3600
    (out_dir / "DIAGNOSTIC.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("D5_layer_effect", "D3_c1_without_query", "D4_PC1_error_roles_val", "cpu_hours")}, indent=1))


if __name__ == "__main__":
    main()
