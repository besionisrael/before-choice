"""Second, algorithmically independent oracle + leakage audit.

Does NOT import oracle.py. Instead of following unique successors, it builds an
adjacency map, enumerates *every* simple path from the start state to any
terminal by DFS, and checks uniqueness, acyclicity, reachability and ties.
Works for pilot_v0 and v1 world records (both share the `transitions`,
`start`, `terminal_rewards` schema; v1 adds `action_labels`).
"""

from __future__ import annotations

import re
from collections import defaultdict


def _actions(world: dict) -> list[str]:
    return list(world.get("action_labels", ["A", "B"]))


def enumerate_paths(world: dict) -> dict[str, list[list[str]]]:
    start = world["start"]
    terms = set(world["terminal_rewards"])
    succ: dict[str, list[str]] = defaultdict(list)
    roots: dict[str, list[str]] = defaultdict(list)
    for e in world["transitions"]:
        if "action" in e:
            if e["source"] != start:
                raise AssertionError("action edge not from start")
            roots[e["action"]].append(e["target"])
        else:
            succ[e["source"]].append(e["target"])
    out: dict[str, list[list[str]]] = {}
    for a in _actions(world):
        found: list[list[str]] = []
        stack = [[start, t] for t in roots[a]]
        while stack:
            p = stack.pop()
            if p[-1] in terms:
                found.append(p)
                continue
            for n in succ[p[-1]]:
                if n in p:
                    raise AssertionError(f"cycle at {n}")
                stack.append(p + [n])
        out[a] = found
    return out


def audit_world(world: dict) -> list[str]:
    """Return a list of violations (empty list == PASS)."""
    errs: list[str] = []
    try:
        allp = enumerate_paths(world)
    except AssertionError as exc:
        return [str(exc)]
    acts = _actions(world)
    for a in acts:
        if len(allp[a]) != 1:
            errs.append(f"action {a}: {len(allp[a])} complete routes (want 1)")
    if errs:
        return errs
    paths = {a: allp[a][0] for a in acts}
    rewards = {a: world["terminal_rewards"][paths[a][-1]] for a in acts}
    if len(set(rewards.values())) != len(acts):
        errs.append("reward tie")
    best = max(rewards, key=rewards.get)
    o = world["oracle"]
    if o["paths"] != paths:
        errs.append("oracle paths mismatch")
    if o["rewards"] != rewards:
        errs.append("oracle rewards mismatch")
    if o["optimal_action"] != best:
        errs.append("oracle optimal mismatch")
    # branch disjointness (primary task: two disjoint linear routes)
    if set(paths[acts[0]][1:]) & set(paths[acts[1]][1:]):
        errs.append("branches share states")
    # unique node labels in rule set
    nodes = [e["source"] for e in world["transitions"]] + [e["target"] for e in world["transitions"]]
    if len(set(e["target"] for e in world["transitions"])) != len(world["transitions"]):
        errs.append("a state has two incoming edges")
    del nodes
    errs += audit_prompt_leakage(world, paths)
    return errs


def audit_prompt_leakage(world: dict, paths: dict[str, list[str]]) -> list[str]:
    errs = []
    p = world["prompt"]
    for bad in ("optimal", "oracle", "paths", "rewards\":", "->"):
        if bad in p:
            errs.append(f"prompt contains '{bad}'")
    # an assembled route (>=3 consecutive states in order) must not appear as a run of names
    for a, path in paths.items():
        for i in range(len(path) - 2):
            pat = r"\W+".join(re.escape(s) for s in path[i : i + 3])
            if re.search(r"(?<![A-Z0-9])" + pat + r"(?![A-Z0-9])", p):
                errs.append(f"assembled route segment for {a} visible in prompt")
                break
    return errs
