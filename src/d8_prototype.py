"""D-8 DESIGN PROTOTYPE (CPU only). Builds a small DESIGN SAMPLE that no model will ever see
(throwaway seed, family ids F90000+) to audit the proposed decoy controls prompt-only.

Three matched decoy variants per family, identical except for ONE digit (the decoy reward):
  decoy_max  unreachable chain, terminal reward 9  (> both real rewards)  -- like v1 decoy, in-range
  decoy_min  unreachable chain, terminal reward 1  (< both real rewards)  -- D-8 proper
  decoy_mid  unreachable chain, terminal reward strictly between the real rewards   -- presence/length control
Real rewards are drawn from 2..8 with gap >= 2 so all three decoys are single-digit:
prompt length, token count, rule positions and reward-line order are identical across the three.
Writes outputs/v2_audit/d8_prototype_audit.json.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from build_dataset import Labeler, build_family  # noqa: E402
from independent_check import audit_world  # noqa: E402
from oracle import format_prompt, solve  # noqa: E402
from prompt_only_control import strategies  # noqa: E402

SEED = 909090  # DESIGN SAMPLE ONLY -- not a pre-registered data seed
VARIANTS = ("decoy_max", "decoy_min", "decoy_mid")


def make(fid, depth, opt, rng, lab):
    base = next(w for w in build_family(fid, depth, opt, rng, lab) if w["variant"] == "base")
    low = rng.randint(2, 6)
    high = rng.randint(low + 2, 8)
    o = base["oracle"]
    nonopt = "B" if o["optimal_action"] == "A" else "A"
    t_opt, t_non = o["paths"][o["optimal_action"]][-1], o["paths"][nonopt][-1]
    chain = [lab.new() for _ in range(depth + 1)]
    pos = sorted(rng.sample(range(3 * depth), depth))
    mid = rng.randint(low + 1, high - 1)
    out = {}
    for v, rd in (("decoy_max", 9), ("decoy_min", 1), ("decoy_mid", mid)):
        edges = [dict(e) for e in base["transitions"]]
        dec = [{"source": chain[i], "target": chain[i + 1]} for i in range(depth)]
        for p, de in zip(pos, dec):
            edges.insert(p, de)
        w = {k: base[k] for k in ("start", "depth", "family_id")}
        w.update(variant=v, world_id=f"{base['family_id']}-{v}", transitions=edges,
                 terminal_rewards={t_opt: high, t_non: low, chain[-1]: rd})
        w["oracle"] = solve(w)
        w["oracle"]["decoy_chain"] = chain
        w["prompt"] = format_prompt(w)
        out[v] = w
    return out


def main() -> None:
    rng = random.Random(SEED)
    lab = Labeler(random.Random(SEED + 1))
    fams = []
    fid = 90000
    for depth in (3, 4):
        for k in range(40):
            fams.append(make(fid, depth, "AB"[k % 2], rng, lab))
            fid += 1
    rep = {"seed": SEED, "status": "DESIGN SAMPLE (never shown to any model)", "n_families": len(fams)}
    # integrity
    rep["oracle_violations"] = sum(bool(audit_world(w)) for f in fams for w in f.values())
    rep["optimal_unchanged_across_variants"] = all(len({w["oracle"]["optimal_action"] for w in f.values()}) == 1 for f in fams)
    rep["same_prompt_length_across_variants"] = all(len({len(w["prompt"]) for w in f.values()}) == 1 for f in fams)
    rep["prompts_differ_in_exactly_one_char"] = all(
        sum(a != b for a, b in zip(f["decoy_max"]["prompt"], f["decoy_min"]["prompt"])) == 1 for f in fams)
    rep["decoy_reward_order"] = all(f["decoy_min"]["terminal_rewards"][f["decoy_min"]["oracle"]["decoy_chain"][-1]]
                                    < min(f["decoy_min"]["oracle"]["rewards"].values()) for f in fams)
    try:
        from v2_token_audit import encoder
        from v2_common import MODELS
        m = MODELS["gemma4_31b"]
        rend, enc, _, _ = encoder(m["repo"], m["revision"], m["template_kwargs"])
        rep["gemma_token_length_equal_across_variants"] = all(
            len({len(enc(rend(w["prompt"]))) for w in f.values()}) == 1 for f in fams)
    except Exception as e:  # tokenizer optional for this audit
        rep["gemma_token_length_equal_across_variants"] = f"not checked ({type(e).__name__})"
    # prompt-only strategy audit (P0 strategies; budget k = depth, i.e. generous)
    tab = defaultdict(lambda: [0, 0])
    for f in fams:
        for v, w in f.items():
            o, d = w["oracle"], w["depth"]
            opt = o["optimal_action"]
            non = "B" if opt == "A" else "A"
            truth = {("non", 2): o["paths"][non][2], ("opt", 2): o["paths"][opt][2]}
            for k in (1, 2, d):
                pred = strategies(w["prompt"], d, k)
                for (role, dd), lab_ in truth.items():
                    for s in ("FWD", "BWD-", "BWD+", "ELIM"):
                        key = f"{v}|d{d}|{role}-d2|{s}|k{k}"
                        tab[key][1] += 1
                        tab[key][0] += int(pred.get(s, {}).get((role, dd)) == lab_)
    rep["strategy_recovery"] = {k: round(h / n, 3) for k, (h, n) in sorted(tab.items())}
    out = ROOT / "outputs" / "v2_audit" / "d8_prototype_audit.json"
    out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "strategy_recovery"}, indent=1))
    for v in VARIANTS:
        for d in (3, 4):
            row = {s: rep["strategy_recovery"][f"{v}|d{d}|non-d2|{s}|k{d}"] for s in ("FWD", "BWD-", "BWD+", "ELIM")}
            row2 = {s: rep["strategy_recovery"][f"{v}|d{d}|opt-d2|{s}|k{d}"] for s in ("FWD", "BWD-", "BWD+", "ELIM")}
            print(f"{v} d{d}  unchosen(non-opt) d2: {row}   chosen(opt) d2: {row2}")


if __name__ == "__main__":
    main()
