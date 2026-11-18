"""D1: controlled, group-split dataset (dataset_version = "v1").

pilot_v0 (generate_worlds.py) is left untouched and regenerated separately.
v1 reuses the v0 prompt template (oracle.format_prompt) for comparability and
adds: exact depth / optimal-action balance, globally unique state labels,
matched control variants inside each graph family, and family-level splits.

Variants per family (all share one underlying graph):
  base, cf            primary minimal pair (deep final-transition swap, v0 rule)
  alias_base/_cf      all non-start labels renamed, identical line order
  swap_base/_cf       action letters exchanged (A<->B); graph identical,
                      correct letter flips -> dissociates letter from path
  shuffle_base/_cf    identical graph, rule lines re-permuted
  decoy_base/_cf      + an UNREACHABLE chain of the same depth ending in a
                      terminal whose reward exceeds both real terminals:
                      breaks the "find max reward, trace back" shortcut
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oracle import format_prompt, solve  # noqa: E402

VARIANTS = ["base", "cf", "alias_base", "alias_cf", "swap_base", "swap_cf",
            "shuffle_base", "shuffle_cf", "decoy_base", "decoy_cf"]
PRIMARY = {"base", "cf"}
DATASET_VERSION = "v1"


class Labeler:
    def __init__(self, rng: random.Random):
        self.rng, self.used = rng, {"S"}

    def new(self) -> str:
        while True:
            c = "".join(self.rng.choices(string.ascii_uppercase + string.digits, k=4))
            if c not in self.used:
                self.used.add(c)
                return c


def _edges(branches: dict[str, list[str]]) -> list[dict]:
    out = []
    for act, path in branches.items():
        for i in range(len(path) - 1):
            e = {"source": path[i], "target": path[i + 1]}
            if i == 0:
                e["action"] = act
            out.append(e)
    return out


def _deep_swap(edges: list[dict], pen_a: str, pen_b: str) -> list[dict]:
    new = [dict(e) for e in edges]
    ea = next(e for e in new if e["source"] == pen_a and "action" not in e)
    eb = next(e for e in new if e["source"] == pen_b and "action" not in e)
    ea["target"], eb["target"] = eb["target"], ea["target"]
    return new


def _finish(w: dict) -> dict:
    w["oracle"] = solve(w)
    o = w["oracle"]
    o["states_by_depth"] = {a: {str(d): s for d, s in enumerate(p)} for a, p in o["paths"].items()}
    o["chosen_if_optimal"] = o["optimal_action"]
    o["unchosen_if_optimal"] = "B" if o["optimal_action"] == "A" else "A"
    w["prompt"] = format_prompt(w)
    w["prompt_sha256"] = hashlib.sha256(w["prompt"].encode("utf-8")).hexdigest()
    return w


def build_family(fid: int, depth: int, optimal_base: str, rng: random.Random, lab: Labeler) -> list[dict]:
    # abstract branch "X" is reached by A in base; reward assignment fixes optimal_base
    bx = ["S"] + [lab.new() for _ in range(depth)]
    by = ["S"] + [lab.new() for _ in range(depth)]
    low, high = sorted(rng.sample(range(1, 10), 2))
    rx, ry = (high, low) if optimal_base == "A" else (low, high)
    order = list(range(2 * depth))
    rng.shuffle(order)
    order2 = order[:]
    while order2 == order:
        rng.shuffle(order2)

    def make(branch_x, branch_y, line_order, swap_letters, deep_cf, decoy):
        acts = ("B", "A") if swap_letters else ("A", "B")
        br = {acts[0]: branch_x, acts[1]: branch_y}
        # canonical index: X-branch edges then Y-branch edges, so swap_* differs
        # from base ONLY in the two action letters
        raw = _edges({acts[0]: branch_x}) + _edges({acts[1]: branch_y})
        edges = [raw[i] for i in line_order]
        if deep_cf:
            edges = _deep_swap(edges, br["A"][-2], br["B"][-2])
        rewards = {branch_x[-1]: rx, branch_y[-1]: ry}
        extra = {}
        if decoy:
            edges = [dict(e) for e in edges]
            chain, pos, dec_reward = decoy
            dec_edges = [{"source": chain[i], "target": chain[i + 1]} for i in range(len(chain) - 1)]
            for p, de in zip(pos, dec_edges):
                edges.insert(p, de)
            rewards[chain[-1]] = dec_reward  # strictly above every real reward
            extra["decoy_chain"] = chain
        return edges, rewards, extra

    # alias labels: fresh names, same structure
    amap = {s: (s if s == "S" else lab.new()) for s in bx + by}
    ax, ay = [amap[s] for s in bx], [amap[s] for s in by]
    chain = [lab.new() for _ in range(depth + 1)]  # depth edges, root has no incoming edge
    # drawn ONCE per family so decoy_base / decoy_cf stay a minimal pair
    decoy_chain = (chain, sorted(rng.sample(range(3 * depth), depth)), 10 + rng.randrange(3))

    specs = {
        "base": (bx, by, order, False, False, None),
        "cf": (bx, by, order, False, True, None),
        "alias_base": (ax, ay, order, False, False, None),
        "alias_cf": (ax, ay, order, False, True, None),
        "swap_base": (bx, by, order, True, False, None),
        "swap_cf": (bx, by, order, True, True, None),
        "shuffle_base": (bx, by, order2, False, False, None),
        "shuffle_cf": (bx, by, order2, False, True, None),
        "decoy_base": (bx, by, order, False, False, decoy_chain),
        "decoy_cf": (bx, by, order, False, True, decoy_chain),
    }
    worlds = []
    for v in VARIANTS:
        x, y, lo, sw, cf, dec = specs[v]
        edges, rewards, extra = make(x, y, lo, sw, cf, dec)
        w = {
            "dataset_version": DATASET_VERSION,
            "world_id": f"F{fid:04d}-{v}",
            "family_id": f"F{fid:04d}",
            "pair_id": f"F{fid:04d}-{v.rsplit('_', 1)[0] if '_' in v else 'primary'}",
            "variant": v,
            "is_counterfactual": v.endswith("cf"),
            "is_primary": v in PRIMARY,
            "depth": depth,
            "reward_low": low, "reward_high": high, "reward_gap": high - low,
            "action_labels": ["A", "B"],
            "start": "S",
            "transitions": edges,
            "terminal_rewards": rewards,
        }
        w = _finish(w)
        w["oracle"].update(extra)
        worlds.append(w)
    return worlds


def build(n_per_depth: int, seed: int, depths=(2, 3, 4), split=(0.6, 0.2, 0.2)) -> list[dict]:
    rng = random.Random(seed)
    lab = Labeler(random.Random(seed + 1))
    out, fid = [], 0
    for depth in depths:
        opts = ["A", "B"] * (n_per_depth // 2) + (["A"] if n_per_depth % 2 else [])
        rng.shuffle(opts)
        fams = []
        for opt in opts:
            fams.append(build_family(fid, depth, opt, rng, lab))
            fid += 1
        # family-level split, stratified by depth
        idx = list(range(len(fams)))
        rng.shuffle(idx)
        n_tr = round(split[0] * len(idx))
        n_va = round(split[1] * len(idx))
        for rank, i in enumerate(idx):
            s = "train" if rank < n_tr else "val" if rank < n_tr + n_va else "test"
            for w in fams[i]:
                w["split"] = s
                out.append(w)
    out.sort(key=lambda w: (w["family_id"], VARIANTS.index(w["variant"])))
    return out


def write_jsonl(rows, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    path.write_bytes(data.encode("utf-8"))  # LF endings -> platform-independent hash
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families-per-depth", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data")
    a = ap.parse_args()
    rows = build(a.families_per_depth, a.seed)
    hashes = {"all": write_jsonl(rows, a.out_dir / "generated" / "pilot_v1.jsonl")}
    for s in ("train", "val", "test"):
        hashes[s] = write_jsonl([r for r in rows if r["split"] == s], a.out_dir / "splits" / f"{s}.jsonl")
    # model-visible view only (no oracle): what inference code must read
    vis = [{k: r[k] for k in ("world_id", "family_id", "variant", "split", "prompt", "prompt_sha256")} for r in rows]
    hashes["model_inputs"] = write_jsonl(vis, a.out_dir / "generated" / "pilot_v1_model_inputs.jsonl")
    man = {"dataset_version": DATASET_VERSION, "seed": a.seed, "families_per_depth": a.families_per_depth,
           "n_worlds": len(rows), "n_families": len({r["family_id"] for r in rows}), "sha256": hashes}
    (a.out_dir / "generated" / "pilot_v1_manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
