"""BTC-H1c token positions (CPU; prompt + pinned tokenizer only; no oracle).

30 slots per world, located from tokenizer offsets of the rendered Gemma chat text:
  0        ANS (last token, `<channel|>`)
  1..12    ARRIVAL key of each candidate (canonical label order): last token of the label in the unique line where it is
           a target ("leads to X." or "you can move to X.")
  13..26   end ("." token) of each of the 14 rule lines, in prompt order
  27, 28   end of the two reward lines, in prompt order
  29       end of the instruction line (Q_END)
The sha256 of the token ids is stored so the container asserts an identical tokenisation.
"""

from __future__ import annotations

import hashlib
import re

import build_h1c as G

P, N_CAND, N_RULES = 30, 12, 14
SLOT_NAMES = ["ANS"] + [f"ARR_c{j}" for j in range(N_CAND)] + [f"RULE_END_{i}" for i in range(N_RULES)] + ["REW_END_0", "REW_END_1", "Q_END"]
DEV_LAYERS = (24, 32, 40, 48, 56)


def _tok_ending_at(offsets, end_char):
    hits = [i for i, (a, b) in enumerate(offsets) if a < end_char <= b]
    assert hits and offsets[hits[-1]][1] == end_char, "label/line end not on a token boundary"
    return hits[-1]


def positions_for(prompt: str, text: str, ids, offsets) -> dict:
    base = text.index(prompt)
    lines, starts, pos = prompt.splitlines(keepends=True), [], 0
    for ln in lines:
        starts.append(pos)
        pos += len(ln)
    p = G.parse(prompt)
    cands = p["cands"]
    assert len(cands) == N_CAND, f"{len(cands)} candidates"
    tok = [-1] * P
    tok[0] = len(ids) - 1
    arrival, rule_end, rew_end, q_end = {}, [], [], None
    for ln, s in zip(lines, starts):
        body = ln.rstrip("\n")
        m = re.match(r"^- (?:At S, action [AB] leads to|From \S+, you can move to) (\S+)\.$", body)
        if m:
            t = m.group(1)
            if t in cands:
                assert t not in arrival, f"{t} has two arrival lines"
                arrival[t] = _tok_ending_at(offsets, base + s + body.rindex(t) + len(t))
            rule_end.append(_tok_ending_at(offsets, base + s + len(body)))
        elif re.match(r"^- \S+: \d+ points$", body):
            rew_end.append(_tok_ending_at(offsets, base + s + len(body)))
        elif body == G.INSTRUCTIONS:
            q_end = _tok_ending_at(offsets, base + s + len(body))
    assert set(arrival) == set(cands) and len(rule_end) == N_RULES and len(rew_end) == 2 and q_end is not None
    for j, c in enumerate(cands):
        tok[1 + j] = arrival[c]
    tok[13:27] = rule_end
    tok[27:29] = rew_end
    tok[29] = q_end
    assert all(0 <= t < len(ids) for t in tok)
    return {"tok": tok, "cands": cands, "n_tokens": len(ids), "ids_sha256": hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()}


def binary_slots(prompt: str, chosen: str) -> list[int]:
    """Slot indices (1..12) of the two successors of the depth-1 label of the letter NOT chosen (prompt + model choice)."""
    p = G.parse(prompt)
    non = "B" if chosen == "A" else "A"
    succ = sorted(t for s, t in p["mv"] if s == p["act"][non])
    return [1 + p["cands"].index(x) for x in succ]
