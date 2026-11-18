"""Independent graph oracle for the Before the Choice pilot."""

from __future__ import annotations


def trace_path(world: dict, initial_action: str) -> list[str]:
    if initial_action not in ("A", "B"):
        raise ValueError("Initial action must be A or B")
    start = world["start"]
    edges = world["transitions"]
    root_candidates = [e for e in edges if e["source"] == start and e.get("action") == initial_action]
    if len(root_candidates) != 1:
        raise ValueError(f"Expected one root transition for {initial_action}")

    path = [start, root_candidates[0]["target"]]
    seen = set(path)
    terminal_rewards = world["terminal_rewards"]
    while path[-1] not in terminal_rewards:
        candidates = [e for e in edges if e["source"] == path[-1] and "action" not in e]
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous or absent transition at {path[-1]}")
        next_state = candidates[0]["target"]
        if next_state in seen:
            raise ValueError("Cycle in path")
        path.append(next_state)
        seen.add(next_state)
        if len(path) > len(edges) + 1:
            raise ValueError("Path unexpectedly long")
    return path


def solve(world: dict) -> dict:
    paths = {action: trace_path(world, action) for action in ("A", "B")}
    rewards = {action: world["terminal_rewards"][path[-1]] for action, path in paths.items()}
    if rewards["A"] == rewards["B"]:
        raise ValueError("Tie not supported in this pilot")
    return {
        "paths": paths,
        "rewards": rewards,
        "optimal_action": max(rewards, key=rewards.get),
    }


def format_prompt(world: dict) -> str:
    """Show only local transitions and reward rules: do NOT show solved paths."""
    lines = [
        "You are in a deterministic environment.",
        f"Start state: {world['start']}",
        "Available initial actions: A or B.",
        "Transition rules (unordered):",
    ]
    for e in world["transitions"]:
        if "action" in e:
            lines.append(f"- At {e['source']}, action {e['action']} leads to {e['target']}.")
        else:
            lines.append(f"- From {e['source']}, the next state is {e['target']}.")
    lines.append("Terminal rewards:")
    for s, r in sorted(world["terminal_rewards"].items()):
        lines.append(f"- {s}: {r} points")
    lines.append("Choose the initial action that reaches the highest reward. Respond with A or B only.")
    return "\n".join(lines)
