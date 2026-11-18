"""Local scoring of BTC-D2-002 (pre-registered D2 competence checkpoint).

Usage: python src/score_d2_competence.py outputs/d2_competence

PRE-REGISTERED RULES (fixed before data, PROTOCOL_FREEZE v0.3 §6/§7 + tie addendum):
  choice      A if log p(A) > log p(B), B if <, TIE if exactly equal (bf16 logits equal)
  score       primary: correct = 1, wrong = 0, TIE = 0.5 (fair coin expectation)
              bounds:  strict (TIE = 0) and lenient (TIE = 1), always reported
  checkpoint  population: depth-2 {base, cf} worlds, train + val (16 worlds, 8 families)
              PASS iff primary accuracy >= 0.70 ; otherwise FAIL -> stop D3+
              flag TIE_SENSITIVE if strict and lenient fall on different sides of 0.70
              (decision still follows the primary score; flag is reported to the owner)
  CIs         95% percentile bootstrap, resampling families (10,000 draws, seed 0);
              descriptive only, the checkpoint uses the point estimate as pre-registered
Only reads train/val oracle files; never opens the locked test files.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILES = ("train.jsonl", "val.jsonl", "pathcf_train.jsonl", "pathcf_val.jsonl")
THRESHOLD = 0.70


def load_oracle() -> dict[str, dict]:
    out = {}
    for f in SPLIT_FILES:
        for x in (ROOT / "data" / "splits" / f).read_text(encoding="utf-8").splitlines():
            r = json.loads(x)
            assert r["split"] in ("train", "val")
            out[r["world_id"]] = r
    return out


def decide(r: dict) -> str:
    if r["logp_A"] > r["logp_B"]:
        return "A"
    if r["logp_A"] < r["logp_B"]:
        return "B"
    return "TIE"


def credit(choice: str, optimal: str, mode: str) -> float:
    if choice == "TIE":
        return {"primary": 0.5, "strict": 0.0, "lenient": 1.0}[mode]
    return float(choice == optimal)


def acc(items, mode="primary"):
    return sum(credit(p["choice"], p["optimal"], mode) for p in items) / len(items) if items else None


def boot_ci(items, n=10_000, seed=0):
    fams = defaultdict(list)
    for p in items:
        fams[p["family_id"]].append(p)
    keys = sorted(fams)
    if not keys:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        s = [p for k in (rng.choice(keys) for _ in keys) for p in fams[k]]
        vals.append(acc(s))
    vals.sort()
    return [vals[int(0.025 * n)], vals[int(0.975 * n) - 1]]


def summarise(items):
    return {"n": len(items), "n_families": len({p["family_id"] for p in items}),
            "n_ties": sum(p["choice"] == "TIE" for p in items),
            "acc_primary": acc(items), "acc_strict": acc(items, "strict"), "acc_lenient": acc(items, "lenient"),
            "ci95_family_bootstrap": boot_ci(items)}


def score(raw: list[dict], oracle: dict[str, dict]) -> dict:
    main = {r["world_id"]: r for r in raw if r["replay"] == 0}
    rep = {r["world_id"]: r for r in raw if r["replay"] == 1}
    per = {}
    for w, r in main.items():
        o = oracle[w]
        per[w] = {"world_id": w, "family_id": o["family_id"], "split": o["split"], "depth": o["depth"],
                  "variant": o["variant"], "optimal": o["oracle"]["optimal_action"], "choice": decide(r),
                  "margin": r["logp_A"] - r["logp_B"], "p_mass_AB": 2.718281828459045 ** r["logp_A"] + 2.718281828459045 ** r["logp_B"],
                  "top1": r["top5"][0][0], "reward_gap": o.get("reward_gap")}
    P = list(per.values())
    pick = lambda f: [p for p in P if f(p)]
    m: dict = {"n_worlds": len(P)}

    # ---- pre-registered checkpoint
    cp = pick(lambda p: p["depth"] == 2 and p["variant"] in ("base", "cf"))
    s = summarise(cp)
    s["decision"] = "PASS" if s["acc_primary"] >= THRESHOLD else "FAIL"
    s["tie_sensitive"] = (s["acc_strict"] >= THRESHOLD) != (s["acc_lenient"] >= THRESHOLD)
    m["CHECKPOINT_depth2_base_cf"] = s

    # ---- descriptive
    m["base_cf_by_depth"] = {d: summarise(pick(lambda p, d=d: p["depth"] == d and p["variant"] in ("base", "cf"))) for d in (2, 3, 4)}
    m["base_cf_all"] = summarise(pick(lambda p: p["variant"] in ("base", "cf")))
    m["base_cf_by_split"] = {sp: summarise(pick(lambda p, sp=sp: p["split"] == sp and p["variant"] in ("base", "cf"))) for sp in ("train", "val")}
    groups = defaultdict(list)
    for p in P:
        groups[p["variant"]].append(p)
    m["by_variant"] = {v: summarise(g) for v, g in sorted(groups.items())}
    fam = defaultdict(list)
    for p in P:
        fam[p["family_id"]].append(p)
    m["by_family"] = {f: {"depth": g[0]["depth"], "split": g[0]["split"], "n": len(g), "acc_primary": acc(g),
                          "base": next((q["choice"] for q in g if q["variant"] == "base"), None),
                          "cf": next((q["choice"] for q in g if q["variant"] == "cf"), None),
                          "optimal_base": next((q["optimal"] for q in g if q["variant"] == "base"), None)}
                      for f, g in sorted(fam.items())}

    # ---- counterfactual sensitivity
    def pairs(a, b):
        return [(per[f"{f}-{a}"], per[f"{f}-{b}"]) for f in fam if f"{f}-{a}" in per and f"{f}-{b}" in per]

    def pair_stats(pp, expect_flip):
        dec = [(x["choice"], y["choice"]) for x, y in pp]
        return {"n_pairs": len(pp),
                "both_correct": sum(x["choice"] == x["optimal"] and y["choice"] == y["optimal"] for x, y in pp),
                "choice_changed": sum(a != b and "TIE" not in (a, b) for a, b in dec),
                "same_choice": sum(a == b and a != "TIE" for a, b in dec),
                "involves_tie": sum("TIE" in (a, b) for a, b in dec),
                "optimal_changes_in_pair": expect_flip}
    m["cf_sensitivity"] = {
        "base_vs_cf": pair_stats(pairs("base", "cf"), True),
        "alias_base_vs_alias_cf": pair_stats(pairs("alias_base", "alias_cf"), True),
        "shuffle_base_vs_shuffle_cf": pair_stats(pairs("shuffle_base", "shuffle_cf"), True),
        "decoy_base_vs_decoy_cf": pair_stats(pairs("decoy_base", "decoy_cf"), True),
        "swap_base_vs_swap_cf": pair_stats(pairs("swap_base", "swap_cf"), True),
        "base_vs_swap_base (letters exchanged)": pair_stats(pairs("base", "swap_base"), True),
        "base_vs_alias_base (labels renamed)": pair_stats(pairs("base", "alias_base"), False),
        "base_vs_shuffle_base (lines reordered)": pair_stats(pairs("base", "shuffle_base"), False),
        "base_vs_decoy_base (decoy added)": pair_stats(pairs("base", "decoy_base"), False),
    }
    # ---- choice-preserving counterfactuals, chosen vs unchosen reported separately (§6b)
    m["path_cf"] = {}
    for kind in ("upath", "cpath", "xswap"):
        for src in ("base", "cf"):
            pp = pairs(src, f"{kind}_{src}")
            for d in sorted({x["depth"] for x, _ in pp}):
                q = [(x, y) for x, y in pp if x["depth"] == d]
                m["path_cf"][f"{kind}_{src}|d{d}"] = {
                    "constructed": len(q),
                    "choice_invariant": sum(x["choice"] == y["choice"] for x, y in q),
                    "admissible_optimal_both": sum(x["choice"] == x["optimal"] and y["choice"] == y["optimal"] for x, y in q)}
    # ---- response bias / format / determinism / negatives
    m["n_choice"] = {c: sum(p["choice"] == c for p in P) for c in ("A", "B", "TIE")}
    m["min_p_mass_AB"] = min(p["p_mass_AB"] for p in P)
    m["n_top1_not_AB"] = sum(p["top1"] not in ("A", "B") for p in P)
    m["replay"] = {"n": len(rep), "identical_choices": sum(decide(rep[w]) == per[w]["choice"] for w in rep),
                   "max_abs_dlogp": max((abs(rep[w]["logp_A"] - main[w]["logp_A"]) + abs(rep[w]["logp_B"] - main[w]["logp_B"]) for w in rep), default=None)}
    m["confident_errors_|margin|>=3"] = sorted([p["world_id"] for p in P if p["choice"] not in ("TIE", p["optimal"]) and abs(p["margin"]) >= 3])
    return {"metrics": m, "per_world": per}


def main() -> None:
    d = Path(sys.argv[1])
    raw = [json.loads(x) for x in (d / "behavior_raw.jsonl").read_text(encoding="utf-8").splitlines()]
    res = score(raw, load_oracle())
    (d / "scored.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res["metrics"]["CHECKPOINT_depth2_base_cf"], indent=1))


if __name__ == "__main__":
    main()
