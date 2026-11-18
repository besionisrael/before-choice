"""Choice-preserving path counterfactuals ("pilot_v1_pathcf"), derived from pilot_v1.

pilot_v1 is read, never modified. For each family the new worlds keep the SAME
labels, terminal rewards, line positions, prompt length and optimal action as
their source world; only the ordered route of one branch changes.

  upath_<src>  reorder two intermediate states of the NON-OPTIMAL branch
  cpath_<src>  same manipulation on the OPTIMAL branch (symmetric reference)
  xswap_<src>  exchange the route-depth-2 state between the two branches
               (depth-1 state for depth-2 families): changes branch
               MEMBERSHIP of two labels, terminals untouched
with <src> in {base, cf}.  Swap positions (route depths, 0 = S):
  depth 3: intermediates 1<->2      depth 4: intermediates 2<->3
  (depth 4 keeps the depth-1 root line unchanged -> multi-hop only)
The pre-registered target, the label at route depth 2 of the non-optimal
branch, changes inside every upath pair (depth 3 and 4). Depth-2 families get
xswap only (one intermediate state -> no reordering possible).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from oracle import format_prompt, solve  # noqa: E402
from build_dataset import write_jsonl  # noqa: E402

SWAP = {3: (1, 2), 4: (2, 3)}


def relabel(world: dict, mapping: dict[str, str]) -> list[dict]:
    """Rename labels inside rule lines in place (line positions unchanged)."""
    out = []
    for e in world["transitions"]:
        n = dict(e)
        n["source"] = mapping.get(e["source"], e["source"])
        n["target"] = mapping.get(e["target"], e["target"])
        out.append(n)
    return out


def derive(src: dict, kind: str) -> dict | None:
    o, depth = src["oracle"], src["depth"]
    opt = o["optimal_action"]
    non = "B" if opt == "A" else "A"
    if kind in ("upath", "cpath"):
        if depth not in SWAP:
            return None
        branch = o["paths"][non if kind == "upath" else opt]
        i, j = SWAP[depth]
        m = {branch[i]: branch[j], branch[j]: branch[i]}
        manipulated = {"branch": "non_optimal" if kind == "upath" else "optimal",
                       "swapped_depths": [i, j], "labels": [branch[i], branch[j]]}
    elif kind == "xswap":
        d = 2 if depth >= 3 else 1
        a, b = o["paths"][opt][d], o["paths"][non][d]
        m = {a: b, b: a}
        manipulated = {"branch": "both", "swapped_depths": [d, d], "labels": [a, b]}
    else:
        raise ValueError(kind)
    w = {k: v for k, v in src.items() if k not in ("oracle", "prompt", "prompt_sha256")}
    w["transitions"] = relabel(src, m)
    w["variant"] = f"{kind}_{src['variant']}"
    w["world_id"] = f"{src['family_id']}-{w['variant']}"
    w["pair_id"] = f"{src['family_id']}-{kind}-{src['variant']}"
    w["source_world_id"] = src["world_id"]
    w["is_primary"] = False
    w["dataset_version"] = "v1_pathcf"
    w["oracle"] = solve(w)
    nw = w["oracle"]
    nw["states_by_depth"] = {a: {str(k): s for k, s in enumerate(p)} for a, p in nw["paths"].items()}
    nw["manipulation"] = manipulated
    w["prompt"] = format_prompt(w)
    w["prompt_sha256"] = hashlib.sha256(w["prompt"].encode("utf-8")).hexdigest()
    return w


def build(rows: list[dict]) -> list[dict]:
    out = []
    for src in rows:
        if src["variant"] not in ("base", "cf"):
            continue
        for kind in ("upath", "cpath", "xswap"):
            w = derive(src, kind)
            if w is not None:
                out.append(w)
    return out


def main() -> None:
    src_path = ROOT / "data" / "generated" / "pilot_v1.jsonl"
    rows = [json.loads(x) for x in src_path.read_text(encoding="utf-8").splitlines()]
    new = build(rows)
    h = {"source_pilot_v1": hashlib.sha256(src_path.read_bytes()).hexdigest(),
         "all": write_jsonl(new, ROOT / "data" / "generated" / "pilot_v1_pathcf.jsonl")}
    for s in ("train", "val", "test"):
        h[s] = write_jsonl([r for r in new if r["split"] == s], ROOT / "data" / "splits" / f"pathcf_{s}.jsonl")
    vis = [{k: r[k] for k in ("world_id", "family_id", "variant", "split", "prompt", "prompt_sha256")} for r in new]
    h["model_inputs"] = write_jsonl(vis, ROOT / "data" / "generated" / "pilot_v1_pathcf_model_inputs.jsonl")
    man = {"dataset_version": "v1_pathcf", "n_worlds": len(new), "sha256": h}
    (ROOT / "data" / "generated" / "pilot_v1_pathcf_manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
