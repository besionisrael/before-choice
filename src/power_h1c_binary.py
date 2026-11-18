"""BTC-H1c (revised C-2): power of the BINARY primary rule on late dead-end-swap pairs, simulated with the exact rule.

    python src/power_h1c_binary.py   -> outputs/metrics/power_h1c_binary.json

Primary: for each admissible late pair, the binary decision between the two successors of the unchosen depth-1 label,
scored from their arrival keys; pair success = both members correct. Rule: in every family-bootstrap resample
  Delta = mean(P) - max(0.25, mean(C_max), mean(P_perm))
H1c MET iff the 2.5 % percentile of Delta > 0.
Model: m = 2 late pairs per family; pair admissible iff both worlds admissible (per-world rate a); family ICC 0.15;
P pair success marginal 0.25 + delta; C_max pair success = 0 (identical inputs across members, proven on CPU);
P_perm pair value = mean over K = 50 draws of Bernoulli products with per-world success 0.5 (independent of P),
i.e. expectation 0.25 with sampling noise. B = 1,000 resamples, 400 datasets per cell.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ppf = NormalDist().inv_cdf
ICC, N_SIM, B, M, K = 0.15, 400, 1000, 2, 50


def one(nf, a, p, rng):
    adm = (rng.random((nf, M, 2)) < a).all(-1)
    f = rng.standard_normal((nf, 1)) * np.sqrt(ICC)
    P = ((f + np.sqrt(1 - ICC) * rng.standard_normal((nf, M))) < ppf(p)).astype(float)
    Q = ((rng.random((nf, M, K)) < 0.5) & (rng.random((nf, M, K)) < 0.5)).mean(-1)
    keep = adm.sum(1) > 0
    cnt = adm.sum(1)[keep].astype(float)
    SP = np.where(adm, P, 0).sum(1)[keep]
    SQ = np.where(adm, Q, 0).sum(1)[keep]
    k = int(keep.sum())
    W = rng.multinomial(k, np.full(k, 1 / k), size=B).astype(float)
    den = W @ cnt
    d = (W @ SP) / den - np.maximum(0.25, (W @ SQ) / den)   # C_max = 0 by construction
    return np.quantile(d, 0.025) > 0, int(cnt.sum())


def main():
    rng = np.random.default_rng(20261118)
    out = {"doc": __doc__.strip(), "cells": []}
    for a in (0.6, 0.75, 0.9):
        for delta in (0.0, 0.05, 0.10, 0.15):
            for nf in (400, 600, 800):
                r = [one(nf, a, 0.25 + delta, rng) for _ in range(N_SIM)]
                c = {"admissibility": a, "delta": delta, "families": nf, "power": float(np.mean([x[0] for x in r])),
                     "mean_pairs": float(np.mean([x[1] for x in r]))}
                out["cells"].append(c)
                print(f"adm {a:.2f} delta {delta:.2f} fam {nf} pairs {c['mean_pairs']:5.0f} power {c['power']:.3f}", flush=True)
    (ROOT / "outputs" / "metrics" / "power_h1c_binary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
