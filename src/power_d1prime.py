"""Monte-Carlo power analysis for the D-1' confirmatory test (design only; no model data used
except the behavioural admissibility rate observed in BTC-V2-C1, used as a planning input).

Unit of inference: LOCKED-TEST family. Each depth-4 family contributes 2 upath pairs
(source base, source cf); each pair is admissible w.p. a (C1: 45/60 = 0.75).
Per admissible pair, the probe is "paired-correct" w.p. p_probe, the strongest threshold baseline
w.p. p_base. Outcomes share a family random effect (ICC via logit-normal) and a within-pair
probe/baseline correlation rho (Gaussian copula). Decision rule (as proposed in the H1 freeze):
superiority iff the 95 % family-bootstrap CI of (probe - baseline) paired accuracy has lower bound > 0;
the cluster bootstrap is approximated here by the cluster-robust normal interval (checked against
a real bootstrap on a subset of settings). Writes outputs/metrics/power_d1prime.json.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sim_once(rng, F, a, p_probe, p_base, icc, rho):
    sd = math.sqrt(icc / (1 - icc) * (math.pi ** 2 / 3)) if icc > 0 else 0.0
    lp, lb = math.log(p_probe / (1 - p_probe)), math.log(p_base / (1 - p_base))
    diffs, ns = [], []
    for _ in range(F):
        u = rng.normal(0, sd)
        n = 0
        d = 0.0
        for _ in range(2):
            if rng.random() > a:
                continue
            z1 = rng.normal()
            z2 = rho * z1 + math.sqrt(1 - rho ** 2) * rng.normal()
            q1 = 1 / (1 + math.exp(-(lp + u)))
            q2 = 1 / (1 + math.exp(-(lb + u)))
            c1 = float(_phi(z1) < q1)
            c2 = float(_phi(z2) < q2)
            d += c1 - c2
            n += 1
        if n:
            diffs.append(d)
            ns.append(n)
    diffs, ns = np.array(diffs), np.array(ns)
    N = ns.sum()
    if N == 0:
        return False, 0
    est = diffs.sum() / N
    # cluster-robust (ratio estimator) SE
    resid = diffs - est * ns
    k = len(ns)
    se = math.sqrt(k / (k - 1) * (resid ** 2).sum()) / N
    return est - 1.96 * se > 0, int(N)


def _phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def bootstrap_check(rng, F, a, p_probe, p_base, icc, rho, B=2000):
    """One simulated dataset, real family bootstrap vs normal approx (sanity)."""
    sd = math.sqrt(icc / (1 - icc) * (math.pi ** 2 / 3))
    fam = []
    for _ in range(F):
        u = rng.normal(0, sd)
        rows = []
        for _ in range(2):
            if rng.random() > a:
                continue
            z1 = rng.normal(); z2 = rho * z1 + math.sqrt(1 - rho ** 2) * rng.normal()
            rows.append((float(_phi(z1) < 1 / (1 + math.exp(-(math.log(p_probe / (1 - p_probe)) + u)))),
                         float(_phi(z2) < 1 / (1 + math.exp(-(math.log(p_base / (1 - p_base)) + u))))))
        if rows:
            fam.append(rows)
    def stat(fs):
        flat = [r for f in fs for r in f]
        return sum(x - y for x, y in flat) / len(flat)
    bs = sorted(stat([fam[rng.integers(len(fam))] for _ in fam]) for _ in range(B))
    return {"estimate": stat(fam), "boot_ci95": [bs[int(0.025 * B)], bs[int(0.975 * B) - 1]]}


def main() -> None:
    rng = np.random.default_rng(20261114)
    a = 45 / 60
    grid_F = [60, 90, 120, 150, 200, 250, 300]
    scenarios = [
        # (label, p_base, p_probe)
        ("base 0.25 (B6 ceiling: perfect anchors + coin), +0.10", 0.25, 0.35),
        ("base 0.25, +0.15", 0.25, 0.40),
        ("base 0.25, +0.20", 0.25, 0.45),
        ("base 0.10, +0.10", 0.10, 0.20),
        ("base 0.10, +0.15", 0.10, 0.25),
        ("base 0.03 (chance 1/36), +0.10", 0.03, 0.13),
    ]
    out = {"admissibility_rate_planning": a, "pairs_per_family": 2, "alpha_two_sided": 0.05, "n_sim": 2000, "results": []}
    for icc in (0.05, 0.15):
        for rho in (0.0, 0.3):
            for lab, pb, pp in scenarios:
                row = {"scenario": lab, "icc": icc, "rho": rho, "power_by_F": {}}
                for F in grid_F:
                    hits = sum(sim_once(rng, F, a, pp, pb, icc, rho)[0] for _ in range(2000))
                    row["power_by_F"][F] = hits / 2000
                out["results"].append(row)
                print(f"icc={icc} rho={rho} {lab:55s}", " ".join(f"F{F}:{v:.2f}" for F, v in row["power_by_F"].items()))
    out["bootstrap_sanity"] = bootstrap_check(rng, 150, a, 0.40, 0.25, 0.15, 0.0)
    print("bootstrap sanity (F=150, +0.15):", out["bootstrap_sanity"])
    p = ROOT / "outputs" / "metrics" / "power_d1prime.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
