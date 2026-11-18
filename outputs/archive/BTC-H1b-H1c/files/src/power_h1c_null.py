"""BTC-H1c: calibration of the primary binary rule under the NULL (owner condition 1), >= 5,000 simulations per scenario.

    python src/power_h1c_null.py   -> outputs/metrics/power_h1c_null.json

Exact rule: Delta = mean(P) - max(0.25, mean(C_max), mean(P_perm)) on admissible late pairs, family-cluster percentile
bootstrap; reject iff the 2.5 % percentile > 0. B = 2,000 resamples per dataset (protocol: 10,000).
Null model (P carries no continuation information). The correct label FLIPS between the two members of a pair, so
under the null it is P's PREFERENCE (X vs Y), not its correctness, that may be correlated: member a is correct iff it
prefers X, member b iff it prefers Y. Preference latents g_a, g_b ~ N(0,1) with a family-level component (share icc)
and a within-pair correlation r.
  N1  r = 0, no family component: preferences independent -> pair success exactly 0.25   <- boundary (least favourable)
  N2  family component icc 0.40 (preferences correlated within family and pair)  -> pair success < 0.25
  N3  deterministic: identical preference in both members -> pair success 0
  N4  as N1 with P_perm at 0.15 (the 0.25 floor binds)
Monte-Carlo uncertainty: binomial SE of the rejection rate and a Clopper-Pearson-type 95 % interval (normal approx.).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ppf = NormalDist().inv_cdf
N_SIM, B, M, K, NF, A = 5000, 2000, 2, 50, 600, 0.75


def one(rng, icc, r, perm_p):
    adm = (rng.random((NF, M, 2)) < A).all(-1)
    f = rng.standard_normal((NF, 1)) * math.sqrt(icc)
    if r >= 1.0:
        P = np.zeros((NF, M))
    else:
        g0 = rng.standard_normal((NF, M))
        ga = f + math.sqrt(1 - icc) * g0
        gb = f + math.sqrt(1 - icc) * (r * g0 + math.sqrt(1 - r * r) * rng.standard_normal((NF, M)))
        P = ((ga < 0) & (gb > 0)).astype(float)       # a prefers X (correct), b prefers Y (correct)
    w = math.sqrt(perm_p)
    Q = ((rng.random((NF, M, K)) < w) & (rng.random((NF, M, K)) < w)).mean(-1)
    keep = adm.sum(1) > 0
    cnt = adm.sum(1)[keep].astype(float)
    SP, SQ = np.where(adm, P, 0).sum(1)[keep], np.where(adm, Q, 0).sum(1)[keep]
    k = int(keep.sum())
    W = rng.multinomial(k, np.full(k, 1 / k), size=B).astype(float)
    den = W @ cnt
    d = (W @ SP) / den - np.maximum(0.25, (W @ SQ) / den)
    return bool(np.quantile(d, 0.025) > 0), float(SP.sum() / cnt.sum())


def main():
    rng = np.random.default_rng(20261119)
    scen = {"N1_boundary_independent": (0.0, 0.0, 0.25), "N2_family_icc0.40": (0.40, 0.0, 0.25),
            "N3_deterministic": (0.0, 1.0, 0.25), "N4_perm_below_floor": (0.0, 0.0, 0.15)}
    out = {"doc": __doc__.strip(), "n_sim": N_SIM, "B": B, "families": NF, "admissibility": A, "nominal_alpha_one_sided": 0.025, "results": {}}
    for name, (icc, rr, pp) in scen.items():
        res = [one(rng, icc, rr, pp) for _ in range(N_SIM)]
        r = np.array([x[0] for x in res])
        p = float(r.mean())
        se = math.sqrt(max(p * (1 - p), 1e-12) / N_SIM)
        out["results"][name] = {"mean_pair_success_P": float(np.mean([x[1] for x in res])), "rejection_rate": p, "mc_se": se, "ci95": [max(0.0, p - 1.96 * se), p + 1.96 * se],
                                "calibrated_le_0.025": bool(p - 1.96 * se <= 0.025)}
        print(name, out["results"][name], flush=True)
    (ROOT / "outputs" / "metrics" / "power_h1c_null.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
