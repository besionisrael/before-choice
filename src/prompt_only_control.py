"""Prompt-only trajectory reconstruction control (DESCRIPTIVE, not a threshold).

Parses ONLY the model-visible prompt text (no oracle) and executes explicit
symbolic strategies with a budget of k rule applications ("hops"):

  FWD   forward chaining from S via the action line            (state at depth d needs d hops)
  BWD-  backward chaining from the LOWEST printed reward       (non-optimal branch; depth-d needs depth-d hops)
  BWD+  backward chaining from the HIGHEST printed reward      (meant for the optimal branch;
        misleads on decoy worlds, where the max reward is unreachable)
  ELIM  FWD-1 and BWD-1 on both branches, then assign the remaining
        intermediate labels by one adjacency check (= 2 composed hops)

Output: for every (world, branch, depth) target, which strategies recover it at
which budget, checked against the oracle AFTER the strategy has run. This maps
the computational shortcuts the prompt affords; it says nothing about the model.
Writes outputs/metrics/prompt_only_control.json and prints a summary table.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACT = re.compile(r"^- At (\S+), action ([AB]) leads to (\S+)\.$")
NXT = re.compile(r"^- From (\S+), the next state is (\S+)\.$")
REW = re.compile(r"^- (\S+): (\d+) points$")


def parse(prompt: str):
    roots, succ, pred, rew = {}, {}, {}, {}
    for line in prompt.splitlines():
        if m := ACT.match(line):
            roots[m.group(2)] = m.group(3)
        elif m := NXT.match(line):
            succ[m.group(1)] = m.group(2)
            pred[m.group(2)] = m.group(1)
        elif m := REW.match(line):
            rew[m.group(1)] = int(m.group(2))
    return roots, succ, pred, rew


def fwd(roots, succ, action, d):
    """label at route depth d of `action`, using exactly d hops (None if chain breaks)."""
    s = roots.get(action)
    for _ in range(d - 1):
        s = succ.get(s)
    return s


def bwd(pred, terminal, j):
    s = terminal
    for _ in range(j):
        s = pred.get(s)
    return s


def strategies(prompt: str, depth: int, k: int) -> dict[str, dict[tuple[str, int], str]]:
    """Return {strategy: {(branch_role, depth): label}} predicted with budget k.
    branch_role is 'opt' / 'non' as the strategy itself infers it (from rewards)."""
    roots, succ, pred, rew = parse(prompt)
    out: dict[str, dict] = defaultdict(dict)
    t_min = min(rew, key=rew.get)
    t_max = max(rew, key=rew.get)
    # FWD: needs to know which action is optimal -> must chain to the terminal (depth hops)
    if k >= depth:
        term = {a: fwd(roots, succ, a, depth) for a in ("A", "B")}
        opt = max(term, key=lambda a: rew.get(term[a], -1))
        non = "B" if opt == "A" else "A"
        for d in range(1, depth + 1):
            out["FWD"][("opt", d)] = fwd(roots, succ, opt, d)
            out["FWD"][("non", d)] = fwd(roots, succ, non, d)
    for d in range(1, depth + 1):
        j = depth - d
        if j <= k:
            out["BWD-"][("non", d)] = bwd(pred, t_min, j)
            out["BWD+"][("opt", d)] = bwd(pred, t_max, j)
    if k >= 2:  # ELIM: depth-1 via root lines, depth-(depth-1) via predecessors, rest by one adjacency
        # anchors are the printed min / max rewards (no unbudgeted reachability check)
        for role, t in (("non", t_min), ("opt", t_max)):
            last = pred.get(t)
            first = next((r for a, r in roots.items() if _chain_contains(succ, r, last)), None) if depth >= 3 else None
            if depth == 4:
                out["ELIM"][(role, 2)] = succ.get(first) if first else None
                out["ELIM"][(role, 3)] = last
    return out


def _chain_contains(succ, start, target):
    s, n = start, 0
    while s is not None and n < 10:
        if s == target:
            return True
        s, n = succ.get(s), n + 1
    return False


def main() -> None:
    rows = []
    for f in ("pilot_v1.jsonl", "pilot_v1_pathcf.jsonl"):
        rows += [json.loads(x) for x in (ROOT / "data" / "generated" / f).read_text(encoding="utf-8").splitlines()]
    table = defaultdict(lambda: [0, 0])  # (variant_group, depth, role, d, strategy, k) -> [hit, n]
    for w in rows:
        o = w["oracle"]
        opt = o["optimal_action"]
        truth = {("opt", d): o["paths"][opt][d] for d in range(1, w["depth"] + 1)}
        non = "B" if opt == "A" else "A"
        truth.update({("non", d): o["paths"][non][d] for d in range(1, w["depth"] + 1)})
        grp = w["variant"].split("_")[0]
        for k in (1, 2, 3, 4):
            pred = strategies(w["prompt"], w["depth"], k)
            for (role, d), lab in truth.items():
                for s in ("FWD", "BWD-", "BWD+", "ELIM"):
                    key = (grp, w["depth"], role, d, s, k)
                    table[key][1] += 1
                    table[key][0] += int(pred.get(s, {}).get((role, d)) == lab)
    res = [{"group": g, "depth": dp, "branch": r, "target_depth": d, "strategy": s, "budget_k": k,
            "recovered": h, "n": n} for (g, dp, r, d, s, k), (h, n) in sorted(table.items())]
    out = ROOT / "outputs" / "metrics" / "prompt_only_control.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    # summary: minimal budget at which ANY strategy recovers >=99% of targets
    summ = defaultdict(dict)
    for r in res:
        if r["n"] and r["recovered"] / r["n"] >= 0.99:
            key = (r["group"], r["depth"], r["branch"], r["target_depth"])
            summ[key].setdefault(r["budget_k"], []).append(r["strategy"])
    print("group depth branch d -> min budget k: strategies")
    for key in sorted(summ):
        k = min(summ[key])
        print(*key, "->", k, ":", ",".join(sorted(summ[key][k])))


if __name__ == "__main__":
    main()
