"""BTC-H1b X3+X4: token positions to capture, computed on CPU from the MODEL-VISIBLE prompt only.

For each world, positions are token indices in the rendered Gemma-4 chat text (same template as all captures):
  slot 0                  ANS        the last token (`<channel|>`, pre-answer position used by H1)
  slots 1..12             LAB_TGT / LAB_SRC of each prompt-derived candidate, canonical (label-sorted) order:
                          last token of the label where it ARRIVES ("leads to X." / "next state is X.") and where it
                          DEPARTS ("From X, ..."); depth-3 worlds have 4 candidates -> slots 9..12 masked
  slots 13, 14            ACT_END_A, ACT_END_B: final "." token of the action lines of letters A and B
  slots 15..20            RULE_END: final "." token of each "From ..." rule line, in prompt order (4 at depth 3)
  slot 21                 Q_END: last token of the instruction line ("... Respond with A or B only.")
No oracle field is read. Candidates = h1_features.candidates_from_prompt (verified equal to the oracle intermediates
in all non-decoy DEV worlds at the H1 freeze). The sha256 of the token ids is stored so the container can assert an
identical tokenisation (hence valid positions).
"""

from __future__ import annotations

import hashlib
import re

import h1_features as F

P = 22
K = 6
SLOT_NAMES = (["ANS"] + [f"{k}_c{j}" for j in range(K) for k in ("LAB_TGT", "LAB_SRC")]
              + ["ACT_END_A", "ACT_END_B"] + [f"RULE_END_{i}" for i in range(6)] + ["Q_END"])
LAYERS = (0, 8, 16, 24, 32, 40, 48, 56, 61)   # 61 = raw block-60 pre-norm (as in D3)
Q_LINE = "Respond with A or B only."


def _tok_ending_at(offsets, end_char: int) -> int:
    hits = [i for i, (a, b) in enumerate(offsets) if a < end_char <= b]
    assert hits, f"no token covers char {end_char}"
    i = hits[-1]
    assert offsets[i][1] == end_char, "label/line end not on a token boundary"
    return i


def positions_for(prompt: str, text: str, ids: list[int], offsets: list[tuple[int, int]]) -> dict:
    base = text.index(prompt)
    lines, pos = prompt.splitlines(keepends=True), 0
    starts = []
    for ln in lines:
        starts.append(pos)
        pos += len(ln)
    cands = F.canonical_candidates(F.candidates_from_prompt(prompt))
    assert len(cands) in (4, 6), f"unexpected candidate count {len(cands)}"
    tok = [-1] * P
    tok[0] = len(ids) - 1
    occ = {c: {} for c in cands}
    act_end, rule_end, q_end = {}, [], None
    for ln, s in zip(lines, starts):
        body = ln.rstrip("\n")
        m = F.ACT.match(body)
        n = F.NXT.match(body)
        if m:
            a, t = m.group(1), m.group(2)
            if t in occ:
                occ[t]["TGT"] = _tok_ending_at(offsets, base + s + body.rindex(t) + len(t))
            act_end[a] = _tok_ending_at(offsets, base + s + len(body))
        elif n:
            src, t = n.group(1), n.group(2)
            if src in occ:
                occ[src]["SRC"] = _tok_ending_at(offsets, base + s + body.index(src) + len(src))
            if t in occ:
                occ[t]["TGT"] = _tok_ending_at(offsets, base + s + body.rindex(t) + len(t))
            rule_end.append(_tok_ending_at(offsets, base + s + len(body)))
        elif body.endswith(Q_LINE):
            q_end = _tok_ending_at(offsets, base + s + len(body))
    for j, c in enumerate(cands):
        assert set(occ[c]) == {"TGT", "SRC"}, f"candidate {c} occurrences {occ[c]}"
        tok[1 + 2 * j], tok[2 + 2 * j] = occ[c]["TGT"], occ[c]["SRC"]
    assert set(act_end) == {"A", "B"} and q_end is not None and len(rule_end) == len(cands)
    tok[13], tok[14] = act_end["A"], act_end["B"]
    for i, t in enumerate(rule_end):
        tok[15 + i] = t
    tok[21] = q_end
    assert all(t < len(ids) for t in tok)
    return {"tok": tok, "mask": [t >= 0 for t in tok], "cands": cands, "n_tokens": len(ids),
            "ids_sha256": hashlib.sha256((",".join(map(str, ids))).encode()).hexdigest()}


def gemma_tokenise():
    """(render, encode-with-offsets) using the SAME template/tokenizer files as the container (pinned revision)."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from v2_common import MODELS, render
    from v2_token_audit import encoder
    m = MODELS["gemma4_31b"]
    rend, _, _, _ = encoder(m["repo"], m["revision"], m["template_kwargs"])
    tk = Tokenizer.from_file(hf_hub_download(m["repo"], "tokenizer.json", revision=m["revision"]))

    def go(prompt: str):
        text = render(rend, "gemma4_31b", prompt)
        e = tk.encode(text, add_special_tokens=False)
        return text, e.ids, e.offsets
    return go


def audit_world(prompt: str, text: str, ids, offsets, rec: dict, decode) -> dict:
    """Checks that every captured label position decodes to the end of that label and line ends to '.'."""
    out = {"labels_ok": True, "line_ends_ok": True}
    for j, c in enumerate(rec["cands"]):
        for t in (rec["tok"][1 + 2 * j], rec["tok"][2 + 2 * j]):
            piece = text[offsets[t][0]:offsets[t][1]]
            out["labels_ok"] &= bool(piece) and c.endswith(piece.strip())
    for t in [rec["tok"][13], rec["tok"][14]] + [x for x in rec["tok"][15:21] if x >= 0] + [rec["tok"][21]]:
        out["line_ends_ok"] &= text[offsets[t][0]:offsets[t][1]].endswith(".")
    out["ans_is_channel"] = decode(ids[-1]) == "<channel|>"
    return out


_ = re  # (regexes live in h1_features)
