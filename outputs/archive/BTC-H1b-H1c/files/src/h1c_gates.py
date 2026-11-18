"""BTC-H1c DEV capture gates (CPU, numpy; importable without Modal).
G2 BLOCKING: bitwise identity of the two binary candidates' arrival-key vectors between base_L and dswap_L of every late
pair, at every captured layer. G3: competence (forced-choice accuracy >= 0.70 overall, >= 0.60 on late-pair worlds)."""

from __future__ import annotations

import json
from pathlib import Path

COMPETENCE_MIN = {"overall": 0.70, "late_pairs": 0.60}


def gates(cap_dir: Path, dev_dir: Path, plan: Path) -> dict:
    """G2 (blocking bitwise identity of binary keys within late pairs) and G3 (competence). Local, CPU, numpy only."""
    import numpy as np
    import h1c_positions as HP
    W = {r["world_id"]: r for r in map(json.loads, (dev_dir / "worlds.jsonl").read_text(encoding="utf-8").splitlines())}
    man = json.loads((cap_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    idx, where = {}, {}
    for s in man["shards"]:
        for k, x in enumerate((cap_dir / f"{s['name']}_index.jsonl").read_text(encoding="utf-8").splitlines()):
            r = json.loads(x)
            idx[r["world_id"]] = r
            where[r["world_id"]] = (s["name"], k)
    cache = {}

    def acts(wid):
        sh, k = where[wid]
        if sh not in cache:
            cache[sh] = np.load(cap_dir / f"{sh}_acts.npy", mmap_mode="r")
        return cache[sh][k]
    g2 = {"late_pairs": 0, "identical": 0, "mismatches": []}
    for f in sorted({w["family_id"] for w in W.values()}):
        a, b = f"{f}-base_L", f"{f}-dswap_L"
        opt = W[a]["oracle"]["optimal_action"]
        bs = HP.binary_slots(W[a]["prompt"], opt)
        same = all(np.array_equal(np.asarray(acts(a)[s]), np.asarray(acts(b)[s])) for s in bs)
        g2["late_pairs"] += 1
        g2["identical"] += same
        if not same:
            g2["mismatches"].append(f)
    correct = {w: (("A" if r["logp_A"] > r["logp_B"] else "B" if r["logp_A"] < r["logp_B"] else "TIE") == W[w]["oracle"]["optimal_action"])
               for w, r in idx.items()}
    late = [w for w in correct if W[w]["variant"] in ("base_L", "dswap_L")]
    g3 = {"overall": float(np.mean(list(correct.values()))), "late_pairs": float(np.mean([correct[w] for w in late])),
          "by_variant": {v: float(np.mean([c for w, c in correct.items() if W[w]["variant"] == v])) for v in sorted({W[w]["variant"] for w in correct})}}
    out = {"G2_bitwise_identity": dict(g2, passed=g2["identical"] == g2["late_pairs"]),
           "G3_competence": dict(g3, thresholds=COMPETENCE_MIN,
                                 passed=g3["overall"] >= COMPETENCE_MIN["overall"] and g3["late_pairs"] >= COMPETENCE_MIN["late_pairs"])}
    out["analysis_allowed"] = out["G2_bitwise_identity"]["passed"] and out["G3_competence"]["passed"]
    return out


