"""Local scoring of BTC-V2 runs (BTC-V2-PROTOCOL_FREEZE §3, §4, §6). CPU only.

    python src/score_v2.py BTC-V2-S1            # score one run  -> outputs/v2/<run>/scored.json
    python src/score_v2.py --select [--exclude BTC-V2-S3 --reason "..."]
                                                 # apply the pre-registered selection rule -> outputs/v2/selection.json

Criteria (frozen): S1 depth-2 base+cf primary acc >= 0.70 | S2 depth-2 base/cf both-correct >= 0.50 |
S3 A+B mass >= 0.5 in >= 95% of worlds | S4 top-1 (FULL vocabulary) in {A,B} in >= 95% of worlds.
C1 = S1 + family-bootstrap 95% lower bound > 0.50 (n=60) | C2 = S2 | C3a = S3 | C3b = S4 |
C4 depth-4 base+cf primary >= 0.70 | C5 admissible depth-4 upath pairs >= 50% of constructed.
Tie rule: log p(A) == log p(B) exactly -> 0.5 (strict 0 / lenient 1 reported). Format conformity (S3/S4)
is computed from the full-vocabulary fields and cannot be rescued by the forced A/B choice.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from score_d2_competence import acc, boot_ci, decide, summarise  # noqa: E402
from v2_common import MODELS, POOLS, RUNS, sha256_file  # noqa: E402

THR_ACC, THR_PAIR, THR_FORMAT, THR_ADMISSIBLE = 0.70, 0.50, 0.95, 0.50


def load_oracle(pool: str) -> dict[str, dict]:
    p = ROOT / "data" / "v2" / f"{POOLS[pool]['file']}.jsonl"
    assert sha256_file(p) == POOLS[pool]["sha256"], "pool hash mismatch"
    return {r["world_id"]: r for r in (json.loads(x) for x in p.read_text(encoding="utf-8").splitlines())}


def per_world(raw: list[dict], oracle: dict[str, dict]) -> dict[str, dict]:
    main = [r for r in raw if r["replay"] == 0]
    assert len(main) == len(oracle) and {r["world_id"] for r in main} == set(oracle), "results must cover the pool exactly"
    out = {}
    for r in main:
        o = oracle[r["world_id"]]
        out[r["world_id"]] = {"world_id": r["world_id"], "family_id": o["family_id"], "depth": o["depth"],
                              "variant": o["variant"], "optimal": o["oracle"]["optimal_action"], "choice": decide(r),
                              "margin": r["logp_A"] - r["logp_B"], "p_mass_AB": math.exp(r["logp_A"]) + math.exp(r["logp_B"]),
                              "top1_in_AB": bool(r["top1_in_AB"]), "top1_token": r["top1_token"]}
    return out


def pairs(per, a, b, depth=None):
    fam = defaultdict(dict)
    for p in per.values():
        fam[p["family_id"]][p["variant"]] = p
    return [(v[a], v[b]) for v in fam.values() if a in v and b in v and (depth is None or v[a]["depth"] == depth)]


def both_correct(pp):
    return sum(x["choice"] == x["optimal"] and y["choice"] == y["optimal"] for x, y in pp)


def score(run_id: str, raw: list[dict]) -> dict:
    stage, _ = RUNS[run_id]
    pool = {"S": "SCREEN", "C": "COMPETENCE"}[stage]
    per = per_world(raw, load_oracle(pool))
    P = list(per.values())
    pick = lambda f: [p for p in P if f(p)]
    d2 = summarise(pick(lambda p: p["depth"] == 2 and p["variant"] in ("base", "cf")))
    d4 = summarise(pick(lambda p: p["depth"] == 4 and p["variant"] in ("base", "cf")))
    pp2 = pairs(per, "base", "cf", depth=2)
    pair_rate = both_correct(pp2) / len(pp2)
    mass_rate = sum(p["p_mass_AB"] >= 0.5 for p in P) / len(P)
    top1_rate = sum(p["top1_in_AB"] for p in P) / len(P)
    m: dict = {"run_id": run_id, "stage": stage, "pool": pool, "n_worlds": len(P),
               "depth2_base_cf": d2, "depth4_base_cf": d4,
               "depth2_base_cf_pairs": {"n": len(pp2), "both_correct": both_correct(pp2), "rate": pair_rate},
               "format": {"p_mass_AB_ge_0.5_rate": mass_rate, "top1_in_AB_rate": top1_rate,
                          "top1_tokens_not_AB": dict(Counter(p["top1_token"] for p in P if not p["top1_in_AB"]).most_common(10))}}
    if stage == "S":
        crit = {"S1": d2["acc_primary"] >= THR_ACC, "S2": pair_rate >= THR_PAIR,
                "S3": mass_rate >= THR_FORMAT, "S4": top1_rate >= THR_FORMAT}
        m["criteria"] = crit
        m["SCREEN_PASS"] = all(crit.values())
    else:
        up4 = pairs(per, "base", "upath_base", 4) + pairs(per, "cf", "upath_cf", 4)
        adm = both_correct(up4) / len(up4)
        lo = d2["ci95_family_bootstrap"][0]
        crit = {"C1": d2["acc_primary"] >= THR_ACC and lo > 0.50, "C2": pair_rate >= THR_PAIR,
                "C3a": mass_rate >= THR_FORMAT, "C3b": top1_rate >= THR_FORMAT,
                "C4": d4["acc_primary"] >= THR_ACC, "C5": adm >= THR_ADMISSIBLE}
        m["upath_depth4"] = {"constructed": len(up4), "admissible": both_correct(up4), "rate": adm}
        m["criteria"] = crit
        comp = crit["C1"] and crit["C2"] and crit["C3a"] and crit["C3b"]
        m["COMPETENCE"] = "PASS" if comp else "FAIL"
        m["D1prime_FEASIBLE"] = bool(comp and crit["C4"] and crit["C5"])
    # ---- descriptive
    m["base_cf_by_depth"] = {d: summarise(pick(lambda p, d=d: p["depth"] == d and p["variant"] in ("base", "cf"))) for d in (2, 3, 4)}
    by_v = defaultdict(list)
    for p in P:
        by_v[p["variant"]].append(p)
    m["by_variant"] = {v: summarise(g) for v, g in sorted(by_v.items())}
    m["cf_choice_changed"] = {f"{a}->{b}": {"n": len(pairs(per, a, b)), "changed": sum(x["choice"] != y["choice"] for x, y in pairs(per, a, b))}
                              for a, b in (("base", "cf"), ("swap_base", "swap_cf"), ("base", "swap_base"), ("alias_base", "alias_cf"),
                                           ("shuffle_base", "shuffle_cf"), ("decoy_base", "decoy_cf"), ("base", "shuffle_base"))
                              if pairs(per, a, b)}
    m["n_choice"] = dict(Counter(p["choice"] for p in P))
    rep = {r["world_id"]: r for r in raw if r["replay"] == 1}
    m["replay"] = {"n": len(rep), "identical_choices": sum(decide(r) == per[w]["choice"] for w, r in rep.items())}
    m["confident_errors"] = sum(p["choice"] not in ("TIE", p["optimal"]) and abs(p["margin"]) >= 3 for p in P)
    return {"metrics": m, "per_world": per}


def select(exclude: dict[str, str]) -> dict:
    cands = []
    for run_id in ("BTC-V2-S1", "BTC-V2-S2", "BTC-V2-S3"):
        if run_id in exclude:
            continue
        f = ROOT / "outputs" / "v2" / run_id / "scored.json"
        assert f.exists(), f"{run_id} not scored; all approved screenings must be complete before selection"
        m = json.loads(f.read_text(encoding="utf-8"))["metrics"]
        mk = RUNS[run_id][1]
        cands.append({"run_id": run_id, "model": mk, "pass": m["SCREEN_PASS"],
                      "depth4_acc": m["depth4_base_cf"]["acc_primary"], "cost_rank": MODELS[mk]["est_cost_rank"]})
    passers = [c for c in cands if c["pass"]]
    if not passers:
        return {"selected_model": None, "decision": "STOP_V2_NO_PASSER", "candidates": cands, "excluded": exclude}
    best = sorted(passers, key=lambda c: (-c["depth4_acc"], c["cost_rank"]))[0]
    return {"selected_model": best["model"], "decision": "SELECTED", "candidates": cands, "excluded": exclude}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--reason", default="")
    a = ap.parse_args()
    if a.select:
        assert not a.exclude or a.reason, "an exclusion needs a recorded reason"
        res = select({e: a.reason for e in a.exclude})
        out = ROOT / "outputs" / "v2" / "selection.json"
        out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(res, indent=1))
        return
    d = ROOT / "outputs" / "v2" / a.run_id
    raw = [json.loads(x) for x in (d / "behavior_raw.jsonl").read_text(encoding="utf-8").splitlines()]
    res = score(a.run_id, raw)
    (d / "scored.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: res["metrics"][k] for k in ("criteria", "depth2_base_cf", "format")}, indent=1))


if __name__ == "__main__":
    main()
