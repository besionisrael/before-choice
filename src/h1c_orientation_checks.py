"""BTC-H1c: orientation-randomisation audit of the generator on the H1c DEV pool (CPU; prompts + design oracle only).

    python src/h1c_orientation_checks.py   -> outputs/h1c_design/orientation_checks.json

For every late pair (base_L, dswap_L), "orientation" features that a decoder WITHOUT path information could exploit:
  o1  the continuing binary label in base_L is the canonically FIRST of the two
  o2  in base_L, the swapped line whose source continues (target = u3) comes BEFORE the other swapped line
  o3  in base_L, the continuing label's ARRIVAL line comes before the other binary label's arrival line
  o4  among families where the two binary labels differ in token length, the continuing label in base_L is the LONGER
      one (ties excluded from o4; Gemma tokenizer)
  o5  optimal action is A
Each should be ~ Bernoulli(0.5) across families and mutually independent (chi-square / exact binomial reported).
Within a pair, o2 and o3 are shared by both members while the continuing label flips: they are exactly the kind of cue
that produces ANTI-CORRELATED predictions; the sign-flip rule is valid because their direction is randomised.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_h1c as G  # noqa: E402


def binom_two_sided_p(k, n):
    from math import comb
    p = sum(comb(n, i) for i in range(n + 1) if abs(i - n / 2) >= abs(k - n / 2)) / 2 ** n
    return min(1.0, p)


def main():
    W = {r["world_id"]: r for r in map(json.loads, (ROOT / "data" / "h1c" / "dev" / "worlds.jsonl").read_text(encoding="utf-8").splitlines())}
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from h1c_token_checks import gemma
        go = gemma()
        ntok = lambda lab: len(go(lab)[1])
    except Exception:  # noqa: BLE001
        ntok = None
    feats = []
    for f in sorted({w["family_id"] for w in W.values()}):
        a = W[f"{f}-base_L"]
        opt = a["oracle"]["optimal_action"]
        non = "B" if opt == "A" else "A"
        p = G.parse(a["prompt"])
        u1 = p["act"][non]
        binary = sorted(t for s, t in p["mv"] if s == u1)
        cont = a["oracle"]["paths"][non][2]
        other = [x for x in binary if x != cont][0]
        u3 = a["oracle"]["paths"][non][3]
        L = p["lines"]
        i_cont = L.index(f"- From {cont}, you can move to {u3}.")
        i_other = next(i for i, l in enumerate(L) if l.startswith(f"- From {other}, you can move to"))
        arr = lambda x: L.index(f"- From {u1}, you can move to {x}.")
        rec = {"o1_cont_canonical_first": cont == binary[0], "o2_cont_line_first": i_cont < i_other,
               "o3_cont_arrival_first": arr(cont) < arr(other), "o5_opt_is_A": opt == "A", "split": a["split"]}
        if ntok:
            dlen = ntok(" " + cont) - ntok(" " + other)
            rec["o4_cont_longer_excl_ties"] = None if dlen == 0 else dlen > 0
        feats.append(rec)
    n = len(feats)
    keys = [k for k in feats[0] if k.startswith("o")]
    sub_ok = lambda k: [r for r in feats if r.get(k) is not None]
    out = {"n_families": n, "marginals": {}, "pairwise_independence": {}}
    for k in keys:
        rr = sub_ok(k)
        s = sum(r[k] for r in rr)
        out["marginals"][k] = {"rate": s / len(rr), "n": len(rr), "binomial_p": binom_two_sided_p(s, len(rr))}
    for k1, k2 in itertools.combinations(keys, 2):
        rr = [r for r in feats if r.get(k1) is not None and r.get(k2) is not None]
        n = len(rr)
        tab = [[sum(1 for r in rr if r[k1] == a and r[k2] == b) for b in (False, True)] for a in (False, True)]
        rows, cols = [sum(t) for t in tab], [tab[0][j] + tab[1][j] for j in range(2)]
        chi = sum((tab[i][j] - rows[i] * cols[j] / n) ** 2 / (rows[i] * cols[j] / n) for i in range(2) for j in range(2) if rows[i] * cols[j])
        pval = math.erfc(math.sqrt(chi / 2))                 # chi-square 1 df
        out["pairwise_independence"][f"{k1}|{k2}"] = {"table": tab, "chi2": chi, "p": pval}
    for sp in ("train", "val"):
        sub = [r for r in feats if r["split"] == sp]
        out[f"marginals_{sp}"] = {k: sum(r[k] for r in sub if r.get(k) is not None) / max(1, sum(r.get(k) is not None for r in sub)) for k in keys}
    pmin = min([v["binomial_p"] for v in out["marginals"].values() if True] + [v["p"] for v in out["pairwise_independence"].values()])
    out["min_p"] = pmin
    out["note"] = ("o5 is balanced by construction (200/200). Marginals of o1-o4 should not differ from 0.5; with ~14 tests a "
                   "single p < 0.05 is expected by chance; the generator draws o1 (labels), o2/o3 (line shuffle) independently.")
    (ROOT / "outputs" / "h1c_design").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "h1c_design" / "orientation_checks.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"marginals": out["marginals"], "min_p": pmin}, indent=1))


if __name__ == "__main__":
    main()
