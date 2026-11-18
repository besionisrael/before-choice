"""BTC-H1b protocol revision: JOINT power of the two primary contrasts (intersection-union: both lower bounds > 0).

    python src/power_h1b_joint.py   -> outputs/metrics/power_h1b_joint.json

Same family model as src/power_h1b.py (PROBE-DEV pairs-per-family 0.02/0.45/0.53, intra-family correlation 0.15).
Three decoders per world with latent Gaussian correctness: P (tested), C (control), Q (= P_perm, query ablation).
Paired accuracies: C = 0.55; P = 0.55 + delta_C; Q = P - delta_perm (delta_perm in {0, 0.025, 0.05, 0.075, 0.10}).
Within-world latent correlations: rho_PC, rho_PQ, rho_CQ. P_perm is an expectation over 50 permutations in the
protocol; modelling it as a single Bernoulli draw is conservative (adds variance).
Both contrasts use the SAME family resample (cluster-robust normal approximation of the shared bootstrap).
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P_PAIRS, ICC, C_PAIRED, N_SIM = (0.02, 0.45, 0.53), 0.15, 0.55, 2000
FAMILIES = (600, 800, 1000)
D_C = (0.05, 0.075, 0.10)
D_PERM = (0.0, 0.025, 0.05, 0.075, 0.10)
CORR = ({"rho_PC": 0.7, "rho_PQ": 0.5, "rho_CQ": 0.5}, {"rho_PC": 0.5, "rho_PQ": 0.3, "rho_CQ": 0.3})
ppf = NormalDist().inv_cdf


def world_acc(target, rng, n=300_000):
    lo, hi = 0.2, 0.999
    for _ in range(36):
        mid = (lo + hi) / 2
        f = rng.standard_normal(n) * np.sqrt(ICC)
        a = (f + rng.standard_normal(n) * np.sqrt(1 - ICC)) < ppf(mid)
        b = (f + rng.standard_normal(n) * np.sqrt(1 - ICC)) < ppf(mid)
        lo, hi = (mid, hi) if (a & b).mean() < target else (lo, mid)
    return (lo + hi) / 2


def sim(nf, acc, corr, rng):
    m = rng.choice(3, size=nf, p=P_PAIRS)
    fam = np.repeat(np.arange(nf), m)
    keep = m > 0
    n = int(m.sum())
    f = rng.standard_normal(nf)[fam] * np.sqrt(ICC)
    S = np.array([[1, corr["rho_PC"], corr["rho_PQ"]], [corr["rho_PC"], 1, corr["rho_CQ"]], [corr["rho_PQ"], corr["rho_CQ"], 1]])
    Lc = np.linalg.cholesky(S)
    thr = np.array([ppf(acc["P"]), ppf(acc["C"]), ppf(acc["Q"])])

    def world():
        z = rng.standard_normal((n, 3)) @ Lc.T
        return (f[:, None] + np.sqrt(1 - ICC) * z) < thr
    w1, w2 = world(), world()
    pair = (w1 & w2).astype(float)                   # columns P, C, Q
    cnt = m[keep].astype(float)
    k = keep.sum()

    def lower(d):
        sums = np.bincount(fam, weights=d, minlength=nf)[keep]
        est = sums.sum() / cnt.sum()
        resid = sums - est * cnt
        se = np.sqrt(k / (k - 1) * (resid ** 2).sum()) / cnt.sum()
        return est - 1.96 * se
    lc, lp = lower(pair[:, 0] - pair[:, 1]), lower(pair[:, 0] - pair[:, 2])
    return lc > 0, lp > 0


def main():
    rng = np.random.default_rng(20261115)
    pc = world_acc(C_PAIRED, rng)
    out = {"doc": __doc__.strip().splitlines()[0], "n_sim": N_SIM, "cells": []}
    for corr in CORR:
        for dc in D_C:
            pp = world_acc(C_PAIRED + dc, rng)
            for dq in D_PERM:
                pq = world_acc(max(C_PAIRED + dc - dq, 0.01), rng) if dq > 0 else pp
                for nf in FAMILIES:
                    r = np.array([sim(nf, {"P": pp, "C": pc, "Q": pq}, corr, rng) for _ in range(N_SIM)])
                    cell = {**corr, "delta_C": dc, "delta_perm": dq, "families": nf, "power_C": float(r[:, 0].mean()),
                            "power_perm": float(r[:, 1].mean()), "power_joint": float((r[:, 0] & r[:, 1]).mean())}
                    out["cells"].append(cell)
                    print(f"rhoPC {corr['rho_PC']} dC {dc:.3f} dP {dq:.3f} fam {nf:4d}  C {cell['power_C']:.3f}  perm {cell['power_perm']:.3f}  joint {cell['power_joint']:.3f}")
    (ROOT / "outputs" / "metrics" / "power_h1b_joint.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
