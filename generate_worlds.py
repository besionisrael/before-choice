"""Reproducible two-branch symbolic mazes and counterfactual pairs."""

from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path

from oracle import format_prompt, solve


def create_label(rng: random.Random, occupied: set[str]) -> str:
    while True:
        candidate = "".join(rng.choices(string.ascii_uppercase + string.digits, k=4))
        if candidate not in occupied:
            occupied.add(candidate)
            return candidate


def create_pair(pair_id: int, seed: int, depth: int) -> tuple[dict, dict]:
    if not 2 <= depth <= 4:
        raise ValueError("Pilot depth must be 2, 3, or 4")
    rng = random.Random(seed)
    used = {"S"}
    branches = {}
    for action in ("A", "B"):
        branches[action] = ["S"] + [create_label(rng, used) for _ in range(depth)]

    edges = []
    for action, path in branches.items():
        for i in range(len(path) - 1):
            edge = {"source": path[i], "target": path[i + 1]}
            if i == 0:
                edge["action"] = action
            edges.append(edge)
    rng.shuffle(edges)

    low, high = sorted(rng.sample(range(1, 10), 2))
    terminal_rewards = {
        branches["A"][-1]: low,
        branches["B"][-1]: high,
    }
    if rng.randrange(2):
        terminal_rewards = {branches["A"][-1]: high, branches["B"][-1]: low}

    common = {"pair_id": pair_id, "seed": seed, "depth": depth, "start": "S", "terminal_rewards": terminal_rewards}
    base = {**common, "world_id": f"{pair_id:05d}-base", "version": "base", "transitions": edges}

    # Counterfactual: swap terminal destinations only, preserving all state names,
    # initial actions, terminal rewards, and every other transition.
    counter_edges = [dict(e) for e in edges]
    pen_a, pen_b = branches["A"][-2], branches["B"][-2]
    last_a = next(e for e in counter_edges if e["source"] == pen_a)
    last_b = next(e for e in counter_edges if e["source"] == pen_b)
    last_a["target"], last_b["target"] = last_b["target"], last_a["target"]
    counter = {**common, "world_id": f"{pair_id:05d}-cf", "version": "counterfactual", "transitions": counter_edges}

    for world in (base, counter):
        world["oracle"] = solve(world)
        world["prompt"] = format_prompt(world)
    assert base["oracle"]["optimal_action"] != counter["oracle"]["optimal_action"]
    return base, counter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/worlds.jsonl"))
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be at least 1")
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for pair_id in range(args.pairs):
            depth = rng.randint(2, 4)
            base, counter = create_pair(pair_id, rng.getrandbits(32), depth)
            for world in (base, counter):
                f.write(json.dumps(world, ensure_ascii=False) + "\n")
    print(f"Wrote {args.pairs * 2} worlds ({args.pairs} matched pairs) to {args.output}")


if __name__ == "__main__":
    main()
