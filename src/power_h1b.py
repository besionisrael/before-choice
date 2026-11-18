"""BTC-H1b confirmatory protocol (PROPOSAL): family-level power for a PAIRED advantage of the pointer over a
capacity-matched control (and over the query-permutation ablation).

    python src/power_h1b.py   -> outputs/metrics/power_h1b.json

Model (simulation; no real data used except the PROBE-DEV pair-per-family distribution):
  * per depth-4 family: 0 / 1 / 2 admissible upath pairs with probabilities measured on PROBE-DEV (0.02 / 0.45 / 0.53);
  * per world, correctness of the two decoders (pointer P, control C) = thresholded latent Gaussians with a shared
    family effect (intra-family correlation `icc` on the liability scale) and a within-world correlation `rho` between
    P and C (they read the same keys and both know the choice, so their errors are strongly correlated);
  * marginal world accuracy of C chosen so its paired accuracy is ~0.55 (DEV: 0.52-0.58); P shifted so the paired
    advantage is delta in {0.05, 0.075, 0.10};
  * test: one-sided 97.5 % lower bound of the family-cluster mean paired difference > 0 (normal approximation of the
    percentile family bootstrap used in the protocol); power = rejection rate over simulations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
P_PAIRS = (0.02, 0.45, 0.53)
FAMILIES = (200, 300, 400, 500, 600, 800, 1000)
DELTAS = (0.05, 0.075, 0.10)
RHOS = (0.5, 0.7, 0.85)
ICC = 0.15
C_PAIRED = 0.55
N_SIM = 2000


def ppf(p):
    from statistics import NormalDist
    return NormalDist().inv_cdf(p)


def world_acc_for_paired(target, icc, rng, n=400_000):
    """World-level accuracy giving the target paired accuracy (both worlds of a pair share the family effect)."""
    lo, hi = 0.3, 0.999
    for _ in range(40):
        mid = (lo + hi) / 2
        t = ppf(mid) * np.sqrt(1)  # threshold on unit-variance liability
        f = rng.standard_normal(n) * np.sqrt(icc)
        a = (f + rng.standard_normal(n) * np.sqrt(1 - icc)) < t
        b = (f + rng.standard_normal(n) * np.sqrt(1 - icc)) < t
        if (a & b).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def simulate(n_fam, pw_c, pw_p, rho, icc, rng):
    m = rng.choice(3, size=n_fam, p=P_PAIRS)
    fam = np.repeat(np.arange(n_fam), m)
    keep = m > 0
    n_pairs = int(m.sum())
    f = rng.standard_normal(n_fam)[fam] * np.sqrt(icc)

    def world():
        z1 = rng.standard_normal(n_pairs)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n_pairs)
        s = np.sqrt(1 - icc)
        return (f + s * z1) < ppf(pw_p), (f + s * z2) < ppf(pw_c)
    p1, c1 = world()
    p2, c2 = world()
    d = (p1 & p2).astype(float) - (c1 & c2).astype(float)
    sums = np.bincount(fam, weights=d, minlength=n_fam)[keep]
    cnt = m[keep].astype(float)
    k = keep.sum()
    est = sums.sum() / cnt.sum()
    resid = sums - est * cnt                          # cluster-robust (ratio estimator) variance
    se = np.sqrt(k / (k - 1) * (resid ** 2).sum()) / cnt.sum()
    return est, est - 1.96 * se > 0, n_pairs, int(k)


def main():
    rng = np.random.default_rng(20261114)
    pw_c = world_acc_for_paired(C_PAIRED, ICC, rng)
    out = {"model": __doc__.strip().splitlines()[0], "p_pairs_per_family": P_PAIRS, "icc": ICC, "control_paired": C_PAIRED,
           "control_world_acc": pw_c, "n_sim": N_SIM, "cells": []}
    for delta in DELTAS:
        pw_p = world_acc_for_paired(C_PAIRED + delta, ICC, rng)
        for rho in RHOS:
            for nf in FAMILIES:
                r = [simulate(nf, pw_c, pw_p, rho, ICC, rng) for _ in range(N_SIM)]
                out["cells"].append({"delta_target": delta, "rho": rho, "families": nf, "pointer_world_acc": pw_p,
                                     "power": float(np.mean([x[1] for x in r])), "mean_est": float(np.mean([x[0] for x in r])),
                                     "mean_pairs": float(np.mean([x[2] for x in r])), "mean_contributing_families": float(np.mean([x[3] for x in r]))})
    (ROOT / "outputs" / "metrics").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "metrics" / "power_h1b.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for c in out["cells"]:
        print(f"delta {c['delta_target']:.3f} rho {c['rho']:.2f} fam {c['families']:4d} pairs {c['mean_pairs']:6.0f} power {c['power']:.3f} est {c['mean_est']:.3f}")


if __name__ == "__main__":
    main()
