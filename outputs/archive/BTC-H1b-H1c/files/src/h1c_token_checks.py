"""BTC-H1c revision (CPU; Gemma tokenizer only; no model, no activations): token- and position-level invariances of the
late dead-end-swap pairs, the reward-free symmetry, and the prompt-only control ladder.

    .venv-gemma4\\Scripts\\python src/h1c_token_checks.py   -> outputs/h1c_design/token_checks.json

For every late pair (base_L, dswap_L):
  * first differing token index between the two rendered chat texts (the first swapped line);
  * the two binary candidates = the successors of the UNCHOSEN depth-1 label (prompt-derived; unchosen = not the model's
    choice, here the oracle optimum for design purposes): their arrival occurrence (last token of the label in
    "From u1, you can move to X.") must lie BEFORE the first differing token, at the SAME token index in both members,
    with identical token ids up to that index -> their contextual keys are identical (causal attention, deterministic
    forward; bitwise determinism of the pipeline was verified on 20 replays per capture and ANS 3,480/3,480 vs D3-DEV-R1);
  * which of the 12 candidates' arrival keys are computed before / after the first differing token;
  * position slots planned for the capture (ANS, 12 arrival keys, rule-line ends, reward-line ends, Q_END).
Prompt-only controls, all members of all pairs:
  * reward-free symmetry: the descendant chains of the two binary candidates are isomorphic once the reward lines are
    ignored (only the reward lines distinguish the continuing branch);
  * control ladder: 1-hop view / occurrence counts / reward-free k-hop graph -> identical across members (deterministic
    controls give paired 0); modified-lines-only control -> chance; full-structure (with rewards) -> exact.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_h1c as G  # noqa: E402

N_FAM, SEED, FID = 200, 777, 90000
LABEL_END = r"(?<= ){lab}(?=[.,])"


def gemma():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from v2_common import MODELS, render
    from v2_token_audit import encoder
    m = MODELS["gemma4_31b"]
    rend, _, _, _ = encoder(m["repo"], m["revision"], m["template_kwargs"])
    tk = Tokenizer.from_file(hf_hub_download(m["repo"], "tokenizer.json", revision=m["revision"]))

    def go(prompt):
        text = render(rend, "gemma4_31b", prompt)
        e = tk.encode(text, add_special_tokens=False)
        return text, e.ids, e.offsets
    return go


def arrival_token(prompt, text, offsets, label):
    base = text.index(prompt)
    for i, line in enumerate(prompt.splitlines(keepends=True)):
        pass
    m = re.search(rf"(?:leads to|you can move to) {re.escape(label)}\.", prompt)
    end = base + m.end() - 1                          # char index just after the label
    hits = [i for i, (a, b) in enumerate(offsets) if a < end <= b]
    assert offsets[hits[-1]][1] == end
    return hits[-1]


def reward_free_signature(mv, node):
    """Out-degree sequence along the descendant tree, ignoring reward lines (labels and rewards not used)."""
    nxt = {}
    for s, t in mv:
        nxt.setdefault(s, []).append(t)

    def sig(n):
        return tuple(sorted(sig(t) for t in nxt.get(n, [])))
    return sig(node)


def main():
    go = gemma()
    rows = G.build(N_FAM, SEED, FID)
    fam = {}
    for w in rows:
        fam.setdefault(w["family_id"], {})[w["variant"]] = w
    agg = {"pairs": 0, "binary_keys_before_first_diff": 0, "binary_keys_same_index": 0, "binary_prefix_ids_identical": 0,
           "reward_free_symmetric": 0, "ladder_identical_inputs": 0, "full_structure_exact": 0, "same_total_tokens": 0}
    before_after = {"before": 0, "after": 0}
    for v in fam.values():
        a, b = v["base_L"], v["dswap_L"]
        opt = a["oracle"]["optimal_action"]
        non = "B" if opt == "A" else "A"
        ta, ia, oa = go(a["prompt"])
        tb, ib, ob = go(b["prompt"])
        first = next(k for k, (x, y) in enumerate(zip(ia, ib)) if x != y)
        pa = G.parse(a["prompt"])
        u1 = pa["act"][non]
        binary = sorted(t for s, t in pa["mv"] if s == u1)
        pos_a = [arrival_token(a["prompt"], ta, oa, x) for x in binary]
        pos_b = [arrival_token(b["prompt"], tb, ob, x) for x in binary]
        agg["pairs"] += 1
        agg["binary_keys_before_first_diff"] += all(p < first for p in pos_a)
        agg["binary_keys_same_index"] += pos_a == pos_b
        agg["binary_prefix_ids_identical"] += ia[:max(pos_a) + 1] == ib[:max(pos_b) + 1]
        agg["same_total_tokens"] += len(ia) == len(ib)
        for c in pa["cands"]:
            before_after["before" if arrival_token(a["prompt"], ta, oa, c) < first else "after"] += 1
        # prompt-only control ladder
        pb = G.parse(b["prompt"])
        agg["reward_free_symmetric"] += all(reward_free_signature(p["mv"], binary[0]) == reward_free_signature(p["mv"], binary[1]) for p in (pa, pb))
        rf = lambda p: sorted((reward_free_signature(p["mv"], x), x in binary) for x in p["cands"])
        same = (G.one_hop_view(a["prompt"], opt) == G.one_hop_view(b["prompt"], opt)
                and G.counts_view(a["prompt"]) == G.counts_view(b["prompt"]) and rf(pa) == rf(pb))
        agg["ladder_identical_inputs"] += same
        agg["full_structure_exact"] += (G.full_structure_u2(a["prompt"], opt) == a["oracle"]["paths"][non][2]
                                        and G.full_structure_u2(b["prompt"], opt) == b["oracle"]["paths"][non][2])
    out = {"n_pairs": agg["pairs"], "checks": {k: f"{v}/{agg['pairs']}" for k, v in agg.items() if k != "pairs"},
           "all_candidates_arrival_vs_first_diff": before_after,
           "capture_slots_proposed": ["ANS", "12 arrival keys (canonical)", "14 rule-line ends", "2 reward-line ends", "Q_END"],
           "notes": {"same_total_tokens": "the two swapped labels may tokenise to different lengths; positions AFTER the first "
                                          "differing token can shift; all slots are located per world from offsets",
                     "modified_lines_only_control": "the two changed lines pair each binary candidate with one depth-3 label in "
                                                    "each member; without the rest of the graph and the reward lines the "
                                                    "continuing one is undetermined (chance 0.5 per world, 0.25 paired)"}}
    (ROOT / "outputs" / "h1c_design").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "h1c_design" / "token_checks.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
