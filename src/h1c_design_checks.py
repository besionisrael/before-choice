"""BTC-H1c design checks (CPU; no model, no activations, no pool): pair invariants, identifiability of the 1-hop
controls, the full-structure prompt control, causal ordering of arrival keys, chance levels.

    python src/h1c_design_checks.py   -> outputs/h1c_design/checks.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_h1c as G  # noqa: E402

N_FAM, SEED, FID = 400, 777, 90000   # design seed (NOT a pool seed)


def check_pair(a, b, kind):
    oa, ob = a["oracle"], b["oracle"]
    opt = oa["optimal_action"]
    non = "B" if opt == "A" else "A"
    pa, pb = G.parse(a["prompt"]), G.parse(b["prompt"])
    chosen_labels = set(oa["paths"][opt][1:-1]) | set(oa["dead_ends"][opt])
    ch_lines = lambda p: [l for l in p["lines"] if any(f" {x}," in l or f" {x}." in l for x in chosen_labels)]
    diff_lines = [i for i, (x, y) in enumerate(zip(a["prompt"].splitlines(), b["prompt"].splitlines())) if x != y]
    r = {"same_choice": ob["optimal_action"] == opt, "same_rewards": a["terminal_rewards"] == b["terminal_rewards"],
         "same_candidates": pa["cands"] == pb["cands"], "n_candidates": len(pa["cands"]),
         "same_depth2_sets": G.one_hop_view(a["prompt"], opt) == G.one_hop_view(b["prompt"], opt),
         "same_chosen_branch_text": ch_lines(pa) == ch_lines(pb), "same_c2": oa["paths"][opt][2] == ob["paths"][opt][2],
         "u2_changes": oa["paths"][non][2] != ob["paths"][non][2], "n_lines_differ": len(diff_lines),
         "same_occurrence_counts": G.counts_view(a["prompt"]) == G.counts_view(b["prompt"])}
    if kind == "cswap":
        r = {"same_choice": r["same_choice"], "c2_changes": oa["paths"][opt][2] != ob["paths"][opt][2],
             "same_u2": oa["paths"][non][2] == ob["paths"][non][2], "n_lines_differ": len(diff_lines)}
    return r


def main():
    rows = G.build(N_FAM, SEED, FID)
    fam = {}
    for w in rows:
        fam.setdefault(w["family_id"], {})[w["variant"]] = w
    agg, order_ok, full_ok, elim = {}, {"L": 0, "E": 0}, 0, {"det_both_correct": 0, "det_one_correct": 0, "pairs": 0}
    for f, v in fam.items():
        for kind, (x, y) in (("dswap_L", ("base_L", "dswap_L")), ("dswap_E", ("base_E", "dswap_E")), ("cswap", ("base_L", "cswap_L"))):
            r = check_pair(v[x], v[y], kind)
            for k, val in r.items():
                agg.setdefault(kind, {}).setdefault(k, set()).add(val)
        for w in v.values():
            opt = w["oracle"]["optimal_action"]
            non = "B" if opt == "A" else "A"
            full_ok += G.full_structure_u2(w["prompt"], opt) == w["oracle"]["paths"][non][2]
        # causal ordering: arrival lines of u2 / ue vs the continuation line
        for o in ("L", "E"):
            w = v[f"base_{o}"]
            non = "B" if w["oracle"]["optimal_action"] == "A" else "A"
            u1, u2, u3 = w["oracle"]["paths"][non][1:4]
            ue = w["oracle"]["dead_ends"][non][0]
            L = G.parse(w["prompt"])["lines"]
            ue2 = next(t for s_, t in G.parse(w["prompt"])["mv"] if s_ == ue)
            dep = [L.index(f"- From {u2}, you can move to {u3}."), L.index(f"- From {ue}, you can move to {ue2}.")]
            arr = [L.index(f"- From {u1}, you can move to {u2}."), L.index(f"- From {u1}, you can move to {ue}.")]
            order_ok[o] += (min(dep) > max(arr)) if o == "L" else (max(dep) < min(arr))
        # a deterministic 1-hop (c2 + branch membership) control: identical inputs -> identical output
        for o in ("L", "E"):
            a, b = v[f"base_{o}"], v[f"dswap_{o}"]
            opt = a["oracle"]["optimal_action"]
            non = "B" if opt == "A" else "A"
            va, vb = G.one_hop_view(a["prompt"], opt), G.one_hop_view(b["prompt"], opt)
            pred = va["d2_sets"][non][0]                    # any deterministic rule on the 1-hop view (+ counts)
            assert va == vb and G.counts_view(a["prompt"]) == G.counts_view(b["prompt"])
            ok = [pred == a["oracle"]["paths"][non][2], pred == b["oracle"]["paths"][non][2]]
            elim["pairs"] += 1
            elim["det_both_correct"] += all(ok)
            elim["det_one_correct"] += sum(ok) == 1
    out = {"n_families": N_FAM, "n_worlds": len(rows),
           "pair_invariants": {k: {kk: sorted(map(str, vv)) for kk, vv in d.items()} for k, d in agg.items()},
           "causal_order_ok": {k: f"{v}/{N_FAM}" for k, v in order_ok.items()},
           "full_structure_control_accuracy": full_ok / len(rows), "one_hop_deterministic_control": elim,
           "chance": {"single_world_10_candidates": 1 / 10, "paired_10": 1 / 100, "single_world_unchosen_2": 1 / 2,
                      "paired_unchosen_2_independent_guess": 1 / 4, "paired_unchosen_2_deterministic_1hop": 0.0}}
    (ROOT / "outputs" / "h1c_design").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "h1c_design" / "checks.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
