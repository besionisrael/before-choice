"""Oracle-free construction of H1 decoder inputs from the MODEL-VISIBLE prompt (+ the model's forced choice).

Inputs used for prediction:  prompt text, the model's chosen letter, activations, frozen label embeddings.
The oracle is used ONLY by `supervision_targets` to compute training/evaluation labels.
"""

from __future__ import annotations

import re

import numpy as np

ACT = re.compile(r"^- At S, action ([AB]) leads to (\S+)\.$", re.M)
NXT = re.compile(r"^- From (\S+), the next state is (\S+)\.$", re.M)
REW = re.compile(r"^- (\S+): (\d+) points$", re.M)
B3_FEATURES = ("in_action_line", "action_letter_is_model_choice", "n_as_source", "n_as_target",
               "first_line_index_norm", "in_reward_line", "printed_reward")


def candidates_from_prompt(prompt: str) -> list[str]:
    """Intermediate states = labels in rule lines that are neither S nor a rewarded terminal. Prompt-only."""
    labs = set()
    for a, t in ACT.findall(prompt):
        labs.add(t)
    for s, t in NXT.findall(prompt):
        labs.update((s, t))
    terminals = {s for s, _ in REW.findall(prompt)}
    return sorted(labs - terminals - {"S"})


def canonical_candidates(labels) -> list[str]:
    """Canonical input order: sorted by label string (random 4-char ids -> carries no slot information)."""
    return sorted(labels)


def b3_features(prompt: str, cands: list[str], chosen_letter: str) -> np.ndarray:
    """Prompt-text features per candidate (no chaining / multi-hop); chosen_letter = MODEL output."""
    lines = prompt.splitlines()
    rule_lines = [i for i, l in enumerate(lines) if l.startswith("- At ") or l.startswith("- From ")]
    rew = {s: int(v) for s, v in REW.findall(prompt)}
    act = {t: a for a, t in ACT.findall(prompt)}
    src = [s for s, _ in NXT.findall(prompt)]
    tgt = [t for _, t in NXT.findall(prompt)] + [t for _, t in ACT.findall(prompt)]
    out = []
    for c in cands:
        first = min((k for k, i in enumerate(rule_lines) if re.search(rf"(?<![A-Z0-9]){re.escape(c)}(?![A-Z0-9])", lines[i])), default=0)
        out.append([float(c in act), float(act.get(c) == chosen_letter), float(src.count(c)), float(tgt.count(c)),
                    first / max(len(rule_lines) - 1, 1), float(c in rew), float(rew.get(c, 0))])
    return np.array(out, dtype=np.float32)


def supervision_targets(world: dict, chosen_letter: str) -> dict[str, str]:
    """SUPERVISION ONLY (uses the oracle): slot -> label, slots relative to the MODEL'S forced choice."""
    o = world["oracle"]
    other = "B" if chosen_letter == "A" else "A"
    ch, un = o["paths"][chosen_letter], o["paths"][other]
    out = {}
    for d in (1, 2, 3):
        if d < len(ch) - 1:
            out[f"c{d}"] = ch[d]
        if d < len(un) - 1:
            out[f"u{d}"] = un[d]
    return out
