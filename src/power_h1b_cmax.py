"""BTC-H1b v1.0: joint power with the EXACT decision rule (owner condition 3).

    python src/power_h1b_cmax.py   -> outputs/metrics/power_h1b_cmax.json

Rule simulated exactly: family-cluster percentile bootstrap (shared resamples); in EVERY resample
  Delta_C    = mean(P) - max(mean(C_bil), mean(C_lin))      (C_max chosen inside the replicate)
  Delta_perm = mean(P) - mean(P_perm)
H1b MET iff the 2.5 % percentile of both is > 0. Point estimates use the full sample.
Per world: latent-Gaussian correctness of P, C_bil, C_lin, Q (=P_perm) with a shared family effect (ICC 0.15) and the
correlation matrix of each scenario; paired success = both worlds of a pair correct. Pairs per family from PROBE-DEV
(0.02 / 0.45 / 0.53) unless a scenario says otherwise. Bootstrap B = 1,000 per simulated dataset (the protocol uses
10,000; B = 1,000 changes percentile noise only marginally); 400 simulated datasets per cell.
Scenarios (assumptions reported in the JSON):
  base         C_lin paired 0.55, C_bil 0.40 (DEV pattern), rho(P,C_lin)=0.7, rho(P,C_bil)=0.5, rho(C_bil,C_lin)=0.7,
               rho(P,Q)=0.5, rho(C,Q)=0.4
  equal_ctrl   ADVERSE: C_bil = C_lin = 0.55 (max of two equally strong noisy controls is biased upwards)
  low_corr     ADVERSE: all correlations 0.3 (decoders disagree more -> larger variance of differences)
  low_adm      ADVERSE: fewer pairs per family (0.10 / 0.50 / 0.40), 1,000 and 875 (INSUFFICIENT boundary) families
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ICC, N_SIM, B = 0.15, 400, 1000
ppf = NormalDist().inv_cdf
ORDER = ("P", "Cb", "Cl", "Q")


def corr_matrix(c):
    S = np.eye(4)
    for (a, b), v in c.items():
        i, j = ORDER.index(a), ORDER.index(b)
        S[i, j] = S[j, i] = v
    return S


SCEN = {
    "base": {"Cl": 0.55, "Cb": 0.40, "pairs": (0.02, 0.45, 0.53),
             "corr": {("P", "Cl"): 0.7, ("P", "Cb"): 0.5, ("Cb", "Cl"): 0.7, ("P", "Q"): 0.5, ("Cb", "Q"): 0.4, ("Cl", "Q"): 0.4}},
    "equal_ctrl": {"Cl": 0.55, "Cb": 0.55, "pairs": (0.02, 0.45, 0.53),
                   "corr": {("P", "Cl"): 0.7, ("P", "Cb"): 0.7, ("Cb", "Cl"): 0.7, ("P", "Q"): 0.5, ("Cb", "Q"): 0.4, ("Cl", "Q"): 0.4}},
    "low_corr": {"Cl": 0.55, "Cb": 0.40, "pairs": (0.02, 0.45, 0.53),
                 "corr": {k: 0.3 for k in (("P", "Cl"), ("P", "Cb"), ("Cb", "Cl"), ("P", "Q"), ("Cb", "Q"), ("Cl", "Q"))}},
    "low_adm": {"Cl": 0.55, "Cb": 0.40, "pairs": (0.10, 0.50, 0.40),
                "corr": {("P", "Cl"): 0.7, ("P", "Cb"): 0.5, ("Cb", "Cl"): 0.7, ("P", "Q"): 0.5, ("Cb", "Q"): 0.4, ("Cl", "Q"): 0.4}},
}


def world_acc(target, rng, n=300_000):
    lo, hi = 0.05, 0.999
    for _ in range(36):
        mid = (lo + hi) / 2
        f = rng.standard_normal(n) * np.sqrt(ICC)
        a = (f + rng.standard_normal(n) * np.sqrt(1 - ICC)) < ppf(mid)
        b = (f + rng.standard_normal(n) * np.sqrt(1 - ICC)) < ppf(mid)
        lo, hi = (mid, hi) if (a & b).mean() < target else (lo, mid)
    return (lo + hi) / 2


def one(nf, thr, Lc, pairs, rng):
    m = rng.choice(3, size=nf, p=pairs)
    keep = m > 0
    fam = np.repeat(np.arange(nf), m)
    n = int(m.sum())
    f = rng.standard_normal(nf)[fam] * np.sqrt(ICC)

    def world():
        return (f[:, None] + np.sqrt(1 - ICC) * (rng.standard_normal((n, 4)) @ Lc.T)) < thr
    pair = (world() & world()).astype(float)                     # n x 4
    S = np.stack([np.bincount(fam, weights=pair[:, j], minlength=nf)[keep] for j in range(4)], 1)
    cnt = m[keep].astype(float)
    k = int(keep.sum())
    W = rng.multinomial(k, np.full(k, 1 / k), size=B).astype(float)  # family resampling weights
    means = (W @ S) / (W @ cnt)[:, None]                             # B x 4
    dC = means[:, 0] - np.maximum(means[:, 1], means[:, 2])
    dQ = means[:, 0] - means[:, 3]
    lc, lq = np.quantile(dC, 0.025), np.quantile(dQ, 0.025)
    return lc > 0, lq > 0, k


def main():
    rng = np.random.default_rng(20261116)
    out = {"doc": __doc__.strip(), "n_sim": N_SIM, "B": B, "scenarios": {k: {**v, "corr": {f"{a}-{b}": c for (a, b), c in v["corr"].items()}}
                                                                          for k, v in SCEN.items()}, "cells": []}
    for name, sc in SCEN.items():
        Lc = np.linalg.cholesky(corr_matrix(sc["corr"]))
        fams = (1000, 875) if name == "low_adm" else (1000,)
        for dC in (0.05, 0.075, 0.10):
            for dQ in (0.0, 0.025, 0.05, 0.10):
                accs = {"P": sc["Cl"] + dC, "Cb": sc["Cb"], "Cl": sc["Cl"], "Q": sc["Cl"] + dC - dQ}
                thr = np.array([ppf(world_acc(accs[k], rng)) for k in ORDER])
                for nf in fams:
                    r = np.array([one(nf, thr, Lc, sc["pairs"], rng) for _ in range(N_SIM)])
                    cell = {"scenario": name, "delta_C_vs_Clin": dC, "delta_perm": dQ, "families": nf,
                            "power_C": float(r[:, 0].mean()), "power_perm": float(r[:, 1].mean()),
                            "power_joint": float((r[:, 0].astype(bool) & r[:, 1].astype(bool)).mean()),
                            "mean_contributing_families": float(r[:, 2].mean())}
                    out["cells"].append(cell)
                    print(f"{name:10s} dC {dC:.3f} dQ {dQ:.3f} fam {nf}  C {cell['power_C']:.3f} perm {cell['power_perm']:.3f} joint {cell['power_joint']:.3f} contrib {cell['mean_contributing_families']:.0f}", flush=True)
    (ROOT / "outputs" / "metrics" / "power_h1b_cmax.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
