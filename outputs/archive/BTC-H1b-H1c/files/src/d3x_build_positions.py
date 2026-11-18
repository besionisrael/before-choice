"""BTC-H1b X3+X4 (CPU): select the PROBE-DEV training-rule worlds and compute/audit their capture positions.

    python src/d3x_build_positions.py     -> outputs/d3x/plan/{positions.jsonl, audit.json}

World selection = the H1 v0.3 training rule on PROBE-DEV (non-decoy; forced choice from the D3-DEV-R1 capture equals
the oracle optimum; no tie): exactly the 3,480 worlds used by H1-DEV-TRAIN and H1b. The oracle is used only for this
selection (as in H1), never as a decoder input. Positions use the prompt and the pinned tokenizer only."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import d3x_positions as XP  # noqa: E402

OUT = ROOT / "outputs" / "d3x" / "plan"


def training_rule_ids() -> list[str]:
    cap, dev = ROOT / "outputs" / "d3" / "probe_dev", ROOT / "data" / "d5" / "probe_dev"
    man = json.loads((cap / "MANIFEST.json").read_text(encoding="utf-8"))
    idx = [json.loads(x) for s in man["shards"] for x in (cap / f"{s['name']}_index.jsonl").read_text(encoding="utf-8").splitlines()]
    worlds = {r["world_id"]: r for r in map(json.loads, (dev / "worlds.jsonl").read_text(encoding="utf-8").splitlines())}
    keep = []
    for r in idx:
        w = worlds[r["world_id"]]
        ch = "A" if r["logp_A"] > r["logp_B"] else "B" if r["logp_A"] < r["logp_B"] else "TIE"
        if not w["variant"].startswith("decoy") and ch != "TIE" and ch == w["oracle"]["optimal_action"]:
            keep.append(r["world_id"])
    return keep


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keep = set(training_rule_ids())
    inputs = [r for r in map(json.loads, (ROOT / "data" / "d5" / "probe_dev" / "model_inputs.jsonl").read_text(encoding="utf-8").splitlines())
              if r["world_id"] in keep]
    go = XP.gemma_tokenise()
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from v2_common import MODELS
    m = MODELS["gemma4_31b"]
    tk = Tokenizer.from_file(hf_hub_download(m["repo"], "tokenizer.json", revision=m["revision"]))
    recs, agg = [], {"n_worlds": 0, "labels_ok": 0, "line_ends_ok": 0, "ans_is_channel": 0, "depth4": 0, "depth3": 0}
    for r in inputs:
        text, ids, off = go(r["prompt"])
        rec = XP.positions_for(r["prompt"], text, ids, off)
        a = XP.audit_world(r["prompt"], text, ids, off, rec, tk.id_to_token)
        agg["n_worlds"] += 1
        for k in ("labels_ok", "line_ends_ok", "ans_is_channel"):
            agg[k] += int(a[k])
        agg["depth4" if len(rec["cands"]) == 6 else "depth3"] += 1
        recs.append({"world_id": r["world_id"], "prompt_sha256": r["prompt_sha256"], **rec})
    data = "".join(json.dumps(x) + "\n" for x in recs).encode()
    (OUT / "positions.jsonl").write_bytes(data)
    agg["positions_sha256"] = hashlib.sha256(data).hexdigest()
    agg["n_positions_captured"] = sum(sum(x["mask"]) for x in recs)
    agg["layers"] = list(XP.LAYERS)
    agg["slot_names"] = XP.SLOT_NAMES
    (OUT / "audit.json").write_text(json.dumps(agg, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in agg.items() if k != "slot_names"}, indent=1))


if __name__ == "__main__":
    main()
