"""Local (CPU) scoring of D2 outputs. Joins model outputs with the private oracle.

Usage: python src/score_d2.py outputs/d2_smoke
Reports chosen/unchosen pair admissibility SEPARATELY and every count explicitly.
Smoke-scale numbers are pipeline checks, not estimates of competence.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_oracle() -> dict[str, dict]:
    rows = {}
    for f in ("pilot_v1.jsonl", "pilot_v1_pathcf.jsonl"):
        for x in (ROOT / "data" / "generated" / f).read_text(encoding="utf-8").splitlines():
            r = json.loads(x)
            rows[r["world_id"]] = r
    return rows


def score(raw: list[dict], oracle: dict[str, dict]) -> dict:
    rep = defaultdict(dict)
    for r in raw:
        rep[r["replay"]][r["world_id"]] = r
    main = rep[0]
    per = {}
    for wid, r in main.items():
        o = oracle[wid]
        choice = "A" if r["logp_A"] > r["logp_B"] else "B"
        per[wid] = {
            "variant": o["variant"], "depth": o["depth"], "family_id": o["family_id"],
            "optimal": o["oracle"]["optimal_action"], "choice": choice,
            "correct": choice == o["oracle"]["optimal_action"],
            "margin_logp_A_minus_B": r["logp_A"] - r["logp_B"],
            "p_mass_AB": math.exp(r["logp_A"]) + math.exp(r["logp_B"]),
            "top1": r["top5"][0][0],
        }
    m: dict = {"n_worlds": len(per)}
    m["accuracy"] = sum(p["correct"] for p in per.values()) / max(len(per), 1)
    m["n_choice_A"] = sum(p["choice"] == "A" for p in per.values())
    m["min_p_mass_AB"] = min((p["p_mass_AB"] for p in per.values()), default=None)
    m["n_top1_not_AB"] = sum(p["top1"] not in ("A", "B") for p in per.values())
    by = defaultdict(list)
    for p in per.values():
        by[f"{p['variant']}|d{p['depth']}"].append(p["correct"])
    m["correct_by_variant_depth"] = {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by.items())}

    def pairs(prefix_src, prefix_new):
        out = []
        for wid, p in per.items():
            if p["variant"].startswith(prefix_new):
                src = f"{p['family_id']}-{p['variant'][len(prefix_new):]}"
                if src in per:
                    out.append((per[src], p))
        return out

    # base/cf: both correct = paired flip
    bc = [(per[f], per[f.replace("-base", "-cf")]) for f in per
          if per[f]["variant"] == "base" and f.replace("-base", "-cf") in per]
    m["base_cf_pairs"] = {"n": len(bc), "both_correct": sum(a["correct"] and b["correct"] for a, b in bc)}
    sw = pairs("", "swap_")
    m["swap_pairs"] = {"n": len(sw), "both_correct": sum(a["correct"] and b["correct"] for a, b in sw),
                       "letter_flipped": sum(a["choice"] != b["choice"] for a, b in sw)}
    for kind, label in (("upath_", "unchosen_branch_manipulated"), ("cpath_", "chosen_branch_manipulated")):
        pp = pairs("", kind)
        m[kind.rstrip("_") + "_pairs"] = {
            "what": label, "n_constructed": len(pp),
            "choice_invariant": sum(a["choice"] == b["choice"] for a, b in pp),
            # admissible for D-1' (unchosen) / B7 (chosen): model chose optimal on BOTH members
            "n_admissible_optimal_both": sum(a["correct"] and b["correct"] for a, b in pp),
        }
    if 1 in rep:
        d = [abs(rep[0][w]["logp_A"] - rep[1][w]["logp_A"]) + abs(rep[0][w]["logp_B"] - rep[1][w]["logp_B"])
             for w in rep[0] if w in rep[1]]
        m["replay_max_abs_logp_diff"] = max(d) if d else None
        m["replay_choice_identical"] = all(
            (rep[0][w]["logp_A"] > rep[0][w]["logp_B"]) == (rep[1][w]["logp_A"] > rep[1][w]["logp_B"]) for w in rep[0])
    return {"metrics": m, "per_world": per}


def main() -> None:
    d = Path(sys.argv[1])
    raw = [json.loads(x) for x in (d / "behavior_raw.jsonl").read_text(encoding="utf-8").splitlines()]
    res = score(raw, load_oracle())
    (d / "scored.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res["metrics"], indent=1))


if __name__ == "__main__":
    main()
