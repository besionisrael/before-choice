"""BTC-H1c design (CPU): two-action task with DEAD ENDS and dead-end-swap minimal pairs. DESIGN TOOL ONLY — no pool is
written by default; `--sample` writes a small inspection sample (different seed) under outputs/h1c_design/.

World (route depth 4 on both branches):  S -a-> d1 -> d2 -> d3 -> T(reward),   d1 -> e1 -> e2 -> e3 (e3: dead end, no
move, no reward). The dead-end side chain has the SAME length as the route (revision 2): d2/e1 and d3/e2 appear exactly
twice each and T/e3 once in the rule lines, so the reward-free graph below u1 is symmetric (the swap is an automorphism
of it): only the REWARD LINES tell the continuing branch from the dead-end branch. Revision 1 (length-2 chain) removed
an occurrence-count cue at depth 2 but left one at depth 3 (u3 has a move, ue2 had none).
Candidates (prompt-derived, canonical order) = all non-start labels not listed with a reward:
d1, d2, d3, e1, e2, e3 of both branches = 12.

Template (H1c; differs from H1 because a node can now have two moves):
  "From X, you can move to Y."  for every non-root edge;  dead ends are stated generically in the instructions.

Variants per family (the line ORDER is shared inside a family except for the *_E variants):
  base_L    both unchosen depth-2 departure lines (u2 -> u3, ue1 -> ue2) placed AFTER both u1 lines ("late")
  dswap_L   dead-end swap on the UNCHOSEN branch: the SOURCES of those two lines are exchanged (u2 <-> ue1; two lines
            change, one label each) -> u2 and ue1 exchange roles; chosen branch text, rewards, candidates, depth-2 set,
            branch membership and every label's occurrence count unchanged; u2 changes; optimal action unchanged
  base_E / dswap_E   same pair with both departure lines placed BEFORE the u1 lines ("early")
  cswap_L   dead-end swap on the CHOSEN branch (reference: c2 changes, u2 unchanged)
Primary pairs: (base_L, dswap_L). Their arrival occurrences of u2 / ue1 precede both departure lines, so the
contextual ARRIVAL keys cannot (by causal attention) contain the continuation; only later positions can.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from build_dataset import Labeler  # noqa: E402

VARIANTS = ("base_L", "dswap_L", "base_E", "dswap_E", "cswap_L")
INSTRUCTIONS = ("Choose the initial action from which the highest reward can be reached. A state with no listed move and "
                "no reward is a dead end. Respond with A or B only.")


# ------------------------------------------------------------------ oracle (independent of the H1 oracle: branching)
def h1c_solve(world: dict) -> dict:
    nxt = {}
    for e in world["transitions"]:
        if "action" not in e:
            nxt.setdefault(e["source"], []).append(e["target"])
    rew = world["terminal_rewards"]

    def route(node, seen):
        if node in rew:
            return [node]
        outs = []
        for t in nxt.get(node, []):
            assert t not in seen, "cycle"
            r = route(t, seen | {t})
            if r is not None:
                outs.append([node] + r)
        assert len(outs) <= 1, f"non-unique rewarded route from {node}"
        return outs[0] if outs else None
    paths, dead = {}, {}
    for a in ("A", "B"):
        root = next(e["target"] for e in world["transitions"] if e.get("action") == a)
        p = route(root, {root})
        assert p is not None, f"no rewarded route for action {a}"
        paths[a] = ["S"] + p
        dead[a] = [t for t in nxt.get(p[0], []) if t != p[1]]
    rewards = {a: rew[p[-1]] for a, p in paths.items()}
    assert rewards["A"] != rewards["B"]
    opt = max(rewards, key=rewards.get)
    return {"paths": paths, "dead_ends": dead, "rewards": rewards, "optimal_action": opt,
            "states_by_depth": {a: {str(k): s for k, s in enumerate(p)} for a, p in paths.items()}}


def format_prompt(world: dict) -> str:
    lines = ["You are in a deterministic environment.", f"Start state: {world['start']}", "Available initial actions: A or B.",
             "Transition rules (unordered):"]
    for e in world["transitions"]:
        lines.append(f"- At {e['source']}, action {e['action']} leads to {e['target']}." if "action" in e
                     else f"- From {e['source']}, you can move to {e['target']}.")
    lines.append("Terminal rewards:")
    for s, r in sorted(world["terminal_rewards"].items()):
        lines.append(f"- {s}: {r} points")
    lines.append(INSTRUCTIONS)
    return "\n".join(lines)


# ------------------------------------------------------------------ generator
def _order(rng, br_opt, br_non, late: bool):
    """Random line order with both unchosen depth-2 departure lines after (late) / before (early) both u1 lines."""
    (Sx, x1, x2, x3, xT, xe1, xe2, xe3), (Sy, y1, y2, y3, yT, ye1, ye2, ye3) = br_opt, br_non
    edges = [("act", x1), ("mv", x1, x2), ("mv", x1, xe1), ("mv", x2, x3), ("mv", x3, xT), ("mv", xe1, xe2), ("mv", xe2, xe3),
             ("act", y1), ("mv", y1, y2), ("mv", y1, ye1), ("mv", y2, y3), ("mv", y3, yT), ("mv", ye1, ye2), ("mv", ye2, ye3)]
    while True:
        rng.shuffle(edges)
        dep = [edges.index(("mv", y2, y3)), edges.index(("mv", ye1, ye2))]
        arr = [edges.index(("mv", y1, y2)), edges.index(("mv", y1, ye1))]
        if (late and min(dep) > max(arr)) or (not late and max(dep) < min(arr)):
            return list(edges)


def build_family(fid: int, opt: str, rng: random.Random, lab: Labeler) -> list[dict]:
    non = "B" if opt == "A" else "A"
    lo, hi = sorted(rng.sample(range(1, 10), 2))
    X = ["S"] + [lab.new() for _ in range(7)]   # S, d1, d2, d3, T, e1, e2, e3  (optimal branch)
    Y = ["S"] + [lab.new() for _ in range(7)]   # unchosen branch
    orders = {"L": _order(rng, X, Y, True), "E": _order(rng, X, Y, False)}

    def world(variant, order, dswap_u=False, dswap_c=False):
        tr = []
        for e in order:
            if e[0] == "act":
                tr.append({"source": "S", "action": opt if e[1] == X[1] else non, "target": e[1]})
                continue
            s, t = e[1], e[2]
            for on, B in ((dswap_u, Y), (dswap_c, X)):   # exchange the sources of d2->d3 and e1->e2
                if on and (s, t) == (B[2], B[3]):
                    s = B[5]
                elif on and (s, t) == (B[5], B[6]):
                    s = B[2]
            tr.append({"source": s, "target": t})
        w = {"world_id": f"F{fid:05d}-{variant}", "family_id": f"F{fid:05d}", "variant": variant, "depth": 4, "start": "S",
             "transitions": tr, "terminal_rewards": {X[4]: hi, Y[4]: lo}, "task": "h1c_dead_end"}
        w["oracle"] = h1c_solve(w)
        w["prompt"] = format_prompt(w)
        w["prompt_sha256"] = hashlib.sha256(w["prompt"].encode()).hexdigest()
        return w
    return [world("base_L", orders["L"]), world("dswap_L", orders["L"], dswap_u=True),
            world("base_E", orders["E"]), world("dswap_E", orders["E"], dswap_u=True),
            world("cswap_L", orders["L"], dswap_c=True)]


def build(n_families: int, seed: int, fid_base: int, exclude: set[str] | None = None) -> list[dict]:
    rng = random.Random(seed)
    lab = Labeler(random.Random(seed + 1))
    lab.used |= exclude or set()
    opts = ["A", "B"] * (n_families // 2)
    rng.shuffle(opts)
    rows = []
    for k, opt in enumerate(opts):
        rows += build_family(fid_base + k, opt, rng, lab)
    return rows


# ------------------------------------------------------------------ prompt-only views (no oracle) used by the controls
def parse(prompt: str) -> dict:
    import re
    act = dict(re.findall(r"^- At S, action ([AB]) leads to (\S+)\.$", prompt, re.M))
    mv = re.findall(r"^- From (\S+), you can move to (\S+)\.$", prompt, re.M)
    rew = dict(re.findall(r"^- (\S+): (\d+) points$", prompt, re.M))
    labels = set(act.values()) | {s for s, _ in mv} | {t for _, t in mv}
    cands = sorted(labels - set(rew) - {"S"})
    lines = [l for l in prompt.splitlines() if l.startswith("- At ") or l.startswith("- From ")]
    return {"act": act, "mv": mv, "rew": rew, "cands": cands, "lines": lines}


def one_hop_view(prompt: str, chosen: str) -> dict:
    """What a c2 / branch-membership control may use: candidates, depth-1 labels per letter, depth-2 SET per branch
    (successors of each depth-1 label) and the model's choice. Continuation rules (moves FROM depth-2 labels) excluded."""
    p = parse(prompt)
    d1 = {a: t for a, t in p["act"].items()}
    d2 = {a: sorted(t for s, t in p["mv"] if s == d1[a]) for a in ("A", "B")}
    return {"cands": p["cands"], "d1": d1, "d2_sets": d2, "choice": chosen}


def counts_view(prompt: str) -> dict:
    """Occurrence count of every candidate label in the rule lines (lexical-frequency control)."""
    p = parse(prompt)
    return {c: sum(l.count(f" {c},") + l.count(f" {c}.") for l in p["lines"]) for c in p["cands"]}


def full_structure_u2(prompt: str, chosen: str) -> str:
    """Strong prompt-only control (no activations): the unchosen depth-2 candidate from which a rewarded terminal is
    reachable (follows the continuation rules to the reward lines)."""
    p = parse(prompt)
    non = "B" if chosen == "A" else "A"
    nxt = {}
    for s, t in p["mv"]:
        nxt.setdefault(s, []).append(t)

    def reaches(n):
        return n in p["rew"] or any(reaches(t) for t in nxt.get(n, []))
    cont = [t for t in nxt[p["act"][non]] if reaches(t)]
    assert len(cont) == 1
    return cont[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="write an inspection sample of N families (seed 777, ids F90000+)")
    a = ap.parse_args()
    if not a.sample:
        raise SystemExit("design tool: use --sample N (no pool is generated without a new owner decision)")
    rows = build(a.sample, 777, 90000)
    out = ROOT / "outputs" / "h1c_design"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sample.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} worlds to {out / 'sample.jsonl'}")


if __name__ == "__main__":
    main()
