"""BTC-H1c: robustness of the primary rule to DEPENDENT predictions (owner condition, "BTC-H1c-DEV" decision).

    python src/power_h1c_robust.py   -> outputs/metrics/power_h1c_robust.json

Problem. In a late pair the correct binary label flips between the members. A decoder with NO continuation information
can still produce ANTI-CORRELATED predictions (X in one member, Y in the other), e.g. from a line-order cue: then the
pair is either fully right or fully wrong, the pair success is 0.5 under the null, and the paired threshold 0.25 is not
valid.
Robust rule (proposed). Orientation is randomised by the generator (which binary label continues in base_L, and the
relative order of the two swapped lines, are independent coin flips per family; verified on the DEV pool). Under H0
(decoder output independent of the orientation coin given everything else), flipping a family's coin maps each member's
correctness c -> 1 - c, so the family statistic d_f = mean correctness over its late-pair worlds - 0.5 is symmetric
about 0. Test: exact-in-distribution SIGN-FLIP randomisation test on family means, one-sided alpha 0.025
(R = 2,000 flips per dataset in this simulation; 10,000 in the protocol); estimate + family-bootstrap CI reported.
Any decoder whose prediction is identical in both members (key-only, choice+keys, 1-hop prompt controls) has d_f = 0
exactly. Secondary contrast vs P_perm: paired sign-flip on family means of (P - P_perm) correctness.
Null scenarios (>= 5,000 datasets each), each also evaluated with the OLD paired rule (P_pair - 0.25 > 0):
  A  independent preferences            B  ANTI-CORRELATED predictions (both right or both wrong, coin per pair)
  C  identical predictions in a pair     D  anti-correlated with a family-level cue shared by the family's two pairs
Power: single-world accuracy 0.5 + delta with within-pair correlation of correctness (0, 0.5) under alternatives.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NF, M, A_ADM, R_FLIP = 600, 2, 0.75, 2000
N_NULL, N_ALT = 5000, 2000


def gen(rng, kind, p=0.5, rho=0.0):
    """Correctness array (NF, M, 2) for the two members of each of M late pairs per family + admissibility mask."""
    adm = rng.random((NF, M, 2)) < A_ADM
    if kind == "independent":
        c = rng.random((NF, M, 2)) < p
    elif kind == "anticorrelated":                   # both members right or both wrong (prediction flips with the pair)
        c = np.repeat(rng.random((NF, M, 1)) < p, 2, axis=2)
    elif kind == "identical":                       # same prediction in both members -> exactly one correct
        x = rng.random((NF, M, 1)) < 0.5
        c = np.concatenate([x, ~x], axis=2)
    elif kind == "anticorrelated_family":           # one cue per family drives both pairs
        c = np.repeat(np.repeat(rng.random((NF, 1, 1)) < p, M, axis=1), 2, axis=2)
    elif kind == "alt":                             # alternative: marginal p, within-pair correctness correlation rho
        z0 = rng.standard_normal((NF, M, 1))
        z = math.sqrt(rho) * z0 + math.sqrt(1 - rho) * rng.standard_normal((NF, M, 2))
        from statistics import NormalDist
        c = z < NormalDist().inv_cdf(p)
    return c.astype(float), adm


def stats(c, adm):
    pair_ok = adm.all(-1)                             # late pair admissible iff both members admissible
    w = np.repeat(pair_ok[..., None], 2, axis=2)
    n_f = w.sum((1, 2))
    keep = n_f > 0
    d_f = (np.where(w, c, 0).sum((1, 2))[keep] / n_f[keep]) - 0.5
    pairs = np.where(pair_ok, c.prod(-1), np.nan)
    return d_f, n_f[keep], pairs


def signflip_reject(d_f, rng):
    obs = d_f.mean()
    flips = rng.choice((-1.0, 1.0), size=(R_FLIP, len(d_f)))
    null = (flips * d_f).mean(1)
    return float((np.sum(null >= obs) + 1) / (R_FLIP + 1)) <= 0.025


def old_rule_reject(pairs):
    v = pairs[~np.isnan(pairs)]
    # family-level normal approximation of the old paired rule's lower bound (P_pair - 0.25)
    return v.mean() - 1.96 * v.std(ddof=1) / math.sqrt(len(v)) > 0.25


def main():
    rng = np.random.default_rng(20261120)
    out = {"doc": __doc__.strip(), "families": NF, "late_pairs_per_family": M, "admissibility": A_ADM, "null": {}, "power": []}
    for kind in ("independent", "anticorrelated", "identical", "anticorrelated_family"):
        new, old = [], []
        for _ in range(N_NULL):
            c, adm = gen(rng, kind)
            d_f, _, pairs = stats(c, adm)
            new.append(signflip_reject(d_f, rng))
            old.append(old_rule_reject(pairs))
        pn, po = float(np.mean(new)), float(np.mean(old))
        se = lambda p: math.sqrt(max(p * (1 - p), 1e-12) / N_NULL)
        out["null"][kind] = {"new_rule_rejection": pn, "new_mc_se": se(pn), "new_ci95": [max(0, pn - 1.96 * se(pn)), pn + 1.96 * se(pn)],
                             "old_paired_rule_rejection": po, "old_mc_se": se(po)}
        print(kind, out["null"][kind], flush=True)
    for rho in (0.0, 0.5):
        for delta in (0.025, 0.05, 0.075, 0.10):
            rej = []
            for _ in range(N_ALT):
                c, adm = gen(rng, "alt", 0.5 + delta, rho)
                rej.append(signflip_reject(stats(c, adm)[0], rng))
            cell = {"rho_within_pair": rho, "delta_single": delta, "paired_equiv_if_rho0": (0.5 + delta) ** 2, "power": float(np.mean(rej))}
            out["power"].append(cell)
            print(cell, flush=True)
    (ROOT / "outputs" / "metrics" / "power_h1c_robust.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
