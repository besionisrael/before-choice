"""BTC-H1c DEV: compute and audit the 30 capture positions for the 2,000 DEV worlds (CPU; tokenizer only).

    .venv-gemma4\Scripts\python src/h1c_build_positions.py   -> outputs/h1c_dev/plan/{positions.jsonl, audit.json}
Audit: every slot decodes to the expected token (label end / "." / `<channel|>`), and for every late pair the two binary
candidates' arrival tokens precede the first differing token with an identical token prefix (structural precondition of
the bitwise key-identity gate)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import h1c_positions as HP  # noqa: E402
import h1c_token_checks as TC  # noqa: E402

DEV = ROOT / "data" / "h1c" / "dev"
OUT = ROOT / "outputs" / "h1c_dev" / "plan"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    go = TC.gemma()
    W = {r["world_id"]: r for r in map(json.loads, (DEV / "worlds.jsonl").read_text(encoding="utf-8").splitlines())}
    inputs = [json.loads(x) for x in (DEV / "model_inputs.jsonl").read_text(encoding="utf-8").splitlines()]
    recs, ok = [], {"slot_tokens_ok": 0, "worlds": 0}
    toks = {}
    for r in inputs:
        text, ids, off = go(r["prompt"])
        q = HP.positions_for(r["prompt"], text, ids, off)
        good = text[off[q["tok"][0]][0]:off[q["tok"][0]][1]] == "<channel|>"
        good &= all(q["cands"][j].endswith(text[off[q["tok"][1 + j]][0]:off[q["tok"][1 + j]][1]].strip()) for j in range(12))
        good &= all(text[off[t][0]:off[t][1]].endswith(".") for t in q["tok"][13:27] + [q["tok"][29]])
        good &= all(text[off[t][0]:off[t][1]].strip().endswith("points") for t in q["tok"][27:29])
        ok["slot_tokens_ok"] += good
        ok["worlds"] += 1
        toks[r["world_id"]] = ids
        recs.append({"world_id": r["world_id"], "prompt_sha256": r["prompt_sha256"], **q})
    pos = {x["world_id"]: x for x in recs}
    pairs = {"late_pairs": 0, "binary_before_first_diff_same_prefix": 0}
    for f in sorted({w["family_id"] for w in W.values()}):
        a, b = f"{f}-base_L", f"{f}-dswap_L"
        opt = W[a]["oracle"]["optimal_action"]
        bs = HP.binary_slots(W[a]["prompt"], opt)
        assert bs == HP.binary_slots(W[b]["prompt"], opt)
        first = next(k for k, (x, y) in enumerate(zip(toks[a], toks[b])) if x != y)
        ta, tb = [pos[a]["tok"][s] for s in bs], [pos[b]["tok"][s] for s in bs]
        pairs["late_pairs"] += 1
        pairs["binary_before_first_diff_same_prefix"] += ta == tb and max(ta) < first and toks[a][:max(ta) + 1] == toks[b][:max(tb) + 1]
    data = "".join(json.dumps(x) + "\n" for x in recs).encode()
    (OUT / "positions.jsonl").write_bytes(data)
    audit = {**ok, **pairs, "positions_sha256": hashlib.sha256(data).hexdigest(), "slots": HP.SLOT_NAMES, "dev_layers": list(HP.DEV_LAYERS),
             "n_tokens": {"min": min(x["n_tokens"] for x in recs), "max": max(x["n_tokens"] for x in recs)}}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "slots"}, indent=1))


if __name__ == "__main__":
    main()
