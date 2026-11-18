"""BTC-H1b protocol revision (CPU, EXPLORATORY, PROBE-DEV only): the capacity-matched control C, the pointer P and the
query-permutation ablation P_perm at the proposed primary configuration (ANS query, contextual arrival keys, layer 40).

    python src/h1b_control_c_dev.py --run                    # single run on the existing X3/X4 PROBE-DEV capture
    python src/h1b_control_c_dev.py --synthetic-test <dir>   # tests only

Splits identical to H1b (family-disjoint): inner-train fits, inner-val early stopping, DEV-val reported (never used for
any choice). No hyper-parameter is tuned here: r = 8, lambda = 0.1, 3 seeds, as proposed.
  P       s(c) = z^T U V^T k_c, z = ANS state (layer 40), k = arrival key (layer 40)
  C       identical form, U, V, optimiser, epochs rule and seeds; z replaced by mu_choice = mean RAW ANS state over the
          INNER-TRAIN worlds with the same model choice (then the same standardisation as z) -> sees only the choice bit
  P_perm  the trained P evaluated on DEV-val with each world's query replaced by the query of a world from a DIFFERENT
          family in the same stratum (model choice x variant x depth); K = 50 seeded draws; per-pair expectation
Also reported: P for slot c1 (elementary sensitivity check), partner counts per stratum, effective capacity of C.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import h1_probe as H  # noqa: E402
import h1_train_dev as T  # noqa: E402
import h1b_explore as X  # noqa: E402
import h1b_x3x4_analysis as A  # noqa: E402

LAYER, QUERY, REP, R, WD, SEEDS, K_PERM = 40, "ANS", "ctx_tgt", 8, 0.1, (0, 1, 2), 50
PERM_SEED = 2026111502


def stratum(row):
    return (row["choice"], row["variant"], row["depth"])


def perm_partners(rows_va, k, seed):
    """For each val world, K partner indices (into rows_va) from DIFFERENT families in the same stratum."""
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for j, r in enumerate(rows_va):
        by[stratum(r)].append(j)
    out, n_avail = [], []
    for j, r in enumerate(rows_va):
        cand = [i for i in by[stratum(r)] if rows_va[i]["family_id"] != r["family_id"]]
        n_avail.append(len(cand))
        assert cand, f"no partner from another family for {r['world_id']}"
        out.append(rng.choice(cand, size=k, replace=True))
    return np.stack(out), n_avail


def run(data: X.Data, xd: A.XData, out: Path, layer=LAYER, seeds=SEEDS, max_epochs=H.MAX_EPOCHS, k_perm=K_PERM) -> dict:
    if (out / "RESULT.json").exists():
        raise SystemExit("RESULT.json exists: single run")
    H.set_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    res = {"config": {"layer": layer, "query": QUERY, "rep": REP, "r": R, "lambda": WD, "seeds": list(seeds), "k_perm": k_perm},
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    meter = H.BudgetMeter(2.0, out / "budget.tsv")
    for slot in ("u2", "c1"):
        itr, iva, va = (X.keep(data, ids, slot) for ids in (data.itr, data.iva, data.va))
        zi, ci, ki = A.build_inputs(xd, layer, itr, QUERY, REP)
        zv, cv, kv = A.build_inputs(xd, layer, iva, QUERY, REP)
        zo, co, ko = A.build_inputs(xd, layer, va, QUERY, REP)
        mu, sd = A.standardise_fit(zi)
        km, ks = A.standardise_fit(ki)
        Zi, Zv, Zo = (torch.from_numpy((z - mu) / sd) for z in (zi, zv, zo))
        Ti, Tv, To = (torch.from_numpy((k - km) / ks) for k in (ki, kv, ko))
        yi, yv = X.ytensor(data, itr, slot), X.ytensor(data, iva, slot)
        rows = lambda ids: [data.dev.rows[i] for i in ids]
        # control C query: per-choice mean of RAW inner-train z, standardised like z (no DEV-val or inner-val information)
        chi = np.array([r["choice"] for r in rows(itr)])
        mus = {c: zi[chi == c].mean(0) for c in ("A", "B")}
        Q = lambda ids: torch.from_numpy(np.stack([(mus[r["choice"]] - mu) / sd for r in rows(ids)]).astype(np.float32))
        Ci, Cv, Co = Q(itr), Q(iva), Q(va)
        fits = {"P": [], "C": []}
        for s in seeds:
            fits["P"].append(A._train_two_tables(Zi, ci, yi, Ti, Zv, cv, yv, Tv, R, WD, s, max_epochs))
            fits["C"].append(A._train_two_tables(Ci, ci, yi, Ti, Cv, cv, yv, Tv, R, WD, s, max_epochs))
            meter.tick(f"{slot}|{s}")
        rec = {}
        for name, (qi, qv, qo) in (("P", (Zi, Zv, Zo)), ("C", (Ci, Cv, Co))):
            pi, pv, po = (H.predict_proba(fits[name], q, c, Tt) for q, c, Tt in ((qi, ci, Ti), (qv, cv, Tv), (qo, co, To)))
            r = {"train": X.acc(pi, itr, data, slot), "inner_val": X.acc(pv, iva, data, slot), "val": X.acc(po, va, data, slot),
                 "epochs": [f.best_epoch for f in fits[name]], "n_params": int(sum(f.U.numel() + f.V.numel() for f in fits[name][:1])),
                 "distinct_query_vectors_train": int(len({tuple(np.round(q.numpy(), 5)) for q in qi[:200]})) if name == "C" else None}
            for L_ in ("A", "B"):
                kk = [k for k, i in enumerate(va) if data.dev.rows[i]["choice"] == L_]
                r[f"val_choice_{L_}"] = X.acc(po[kk], [va[k] for k in kk], data, slot)
            if slot == "u2":
                r["val_paired"], r["n_pairs"] = X.paired(po, va, data)
            rec[name] = r
        if slot == "u2":
            # P_perm: trained P, queries swapped within stratum across families
            rv = rows(va)
            partners, n_avail = perm_partners(rv, k_perm, PERM_SEED)
            pos = {i: k for k, i in enumerate(va)}
            ok = np.zeros((len(va), k_perm))
            for kk in range(k_perm):
                Zp = Zo[partners[:, kk]]
                pred = H.argmax_canonical(H.predict_proba(fits["P"], Zp, co, To)).numpy()
                ok[:, kk] = [pred[k] == data.dev.rows[i]["tgt"]["u2"] for k, i in enumerate(va)]
            ps = [(a, b) for a, b in data.dev.pairs["val"] if a in pos and b in pos]
            paired_perm = float(np.mean([np.mean(ok[pos[a]] * ok[pos[b]]) for a, b in ps])) if ps else float("nan")
            by = defaultdict(int)
            for r_ in rv:
                by[str(stratum(r_))] += 1
            rec["P_perm"] = {"val": float(ok.mean()), "val_paired": paired_perm, "n_pairs": len(ps),
                             "partners_per_world": {"min": int(min(n_avail)), "median": float(np.median(n_avail)), "max": int(max(n_avail))},
                             "strata_sizes_val": dict(sorted(by.items())), "same_family_partners": 0}
            pr, cr = rec["P"], rec["C"]
            rec["contrasts_dev_val"] = {"P_minus_C_paired": pr["val_paired"] - cr["val_paired"],
                                        "P_minus_Pperm_paired": pr["val_paired"] - paired_perm,
                                        "P_minus_C_single": pr["val"] - cr["val"], "P_minus_Pperm_single": pr["val"] - rec["P_perm"]["val"]}
        res[slot] = rec
    res["cpu_hours"] = meter.used_s() / 3600
    res["capacity_note"] = ("C has the same parameter shapes as P (U: d x r, V: d x r); its query takes only 2 distinct values "
                            "(mu_A, mu_B), so its effective score class is s(c) = w_choice^T k_c with w_A, w_B in the rank-r image "
                            "of V U^T: a rank-constrained choice + keys decoder trained exactly like P")
    (out / "RESULT.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--synthetic-test", default="")
    a = ap.parse_args()
    if a.synthetic_test:
        data, xdir, pos = A.make_synthetic_x(Path(a.synthetic_test), n_worlds=960)
        print(json.dumps(run(data, A.XData(data, xdir, pos), Path(a.synthetic_test) / "out", layer=8, seeds=(0,), max_epochs=30, k_perm=5)["u2"], indent=1))
        return
    if not a.run:
        raise SystemExit("use --run or --synthetic-test")
    dev = T.Dev(T.REAL["capture"], T.REAL["dev"])
    data = X.Data(dev, T.REAL["dev"])
    xd = A.XData(data, ROOT / "outputs" / "d3x" / "probe_dev", ROOT / "outputs" / "d3x" / "plan" / "positions.jsonl")
    r = run(data, xd, ROOT / "outputs" / "h1b_control_c")
    print(json.dumps({k: r[k] for k in ("u2", "c1", "cpu_hours")}, indent=1))


if __name__ == "__main__":
    main()
