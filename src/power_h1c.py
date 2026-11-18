"""BTC-H1c design: family-level power for the primary contrast on late dead-end-swap pairs.

    python src/power_h1c.py   -> outputs/metrics/power_h1c.json

Primary statistic (proposed): paired u2 accuracy of the pointer P on admissible (base_L, dswap_L) pairs, against the
threshold max(0.25, strongest control). On these pairs every control that reads only the arrival keys, the 1-hop
structure, occurrence counts or the choice gives IDENTICAL outputs for both members (paired = 0 by construction), and
P_perm is expected <= 0.25; the binding threshold is therefore 0.25 (independent guessing between u2 and ue1).
Model: m primary pairs per family (m = 2 line orders), each admissible iff both worlds are admissible (per-world
admissibility a, i.i.d.); pair success ~ latent Gaussian with family effect (ICC 0.15), marginal p = 0.25 + delta;
test: family-cluster lower 97.5 % bound of mean(success - 0.25) > 0. 2,000 simulations per cell.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ppf = NormalDist().inv_cdf
ICC, N_SIM, P0 = 0.15, 2000, 0.25


def sim(nf, m, a, p, rng):
    adm = (rng.random((nf, m, 2)) < a).all(-1)                 # pair admissible iff both members admissible
    f = rng.standard_normal((nf, 1)) * np.sqrt(ICC)
    succ = (f + np.sqrt(1 - ICC) * rng.standard_normal((nf, m))) < ppf(p)
    d = np.where(adm, succ - P0, 0.0)
    cnt = adm.sum(1).astype(float)
    keep = cnt > 0
    sums, cnt = d.sum(1)[keep], cnt[keep]
    k = keep.sum()
    est = sums.sum() / cnt.sum()
    resid = sums - est * cnt
    se = np.sqrt(k / (k - 1) * (resid ** 2).sum()) / cnt.sum()
    return est - 1.96 * se > 0, int(cnt.sum()), int(k)


def main():
    rng = np.random.default_rng(20261117)
    out = {"doc": __doc__.strip(), "cells": []}
    for a in (0.6, 0.75, 0.9):
        for delta in (0.05, 0.10, 0.15):
            for nf in (300, 450, 600, 800, 1000):
                r = [sim(nf, 2, a, P0 + delta, rng) for _ in range(N_SIM)]
                c = {"admissibility": a, "delta": delta, "families": nf, "power": float(np.mean([x[0] for x in r])),
                     "mean_pairs": float(np.mean([x[1] for x in r])), "mean_contributing_families": float(np.mean([x[2] for x in r]))}
                out["cells"].append(c)
                print(f"adm {a:.2f} delta {delta:.2f} fam {nf:4d} pairs {c['mean_pairs']:6.0f} power {c['power']:.3f}")
    (ROOT / "outputs" / "metrics" / "power_h1c.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
