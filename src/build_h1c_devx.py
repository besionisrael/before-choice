"""BTC-H1c DEV EXTENSION (CPU; owner decision "BTC-H1c ... augmentation unique de DEV"). Not a TEST pool, not sealed.

    python src/build_h1c_devx.py   -> data/h1c/devx/{worlds.jsonl, model_inputs.jsonl, manifest.json}
                                     outputs/h1c_devx/plan/SPLITS.json  (frozen three-way family split)

* 600 NEW families (300 optimal-A / 300 optimal-B) x 5 variants = 3,000 worlds, seed 2026111406, ids F51000-F51599,
  task revision 2 (same generator as the H1c DEV pool); labels disjoint from v1, v2, PROBE-DEV, LOCKED-TEST, the
  H1b design and the H1c DEV pool (all regenerated in memory / read from the non-sealed H1c DEV file)
* FROZEN family roles over the combined DEV (400 old + 600 new = 1,000 families), stratified by optimal action:
    TRAIN   = all 400 old H1c-DEV families (already analysed -> never used for selection or evaluation)
              + 300 new families                                                    -> decoder fitting
    SELECT  = 100 new families  -> early stopping and the PC1-based layer choice (u2 is never used for selection)
    EVAL    = 200 new families  -> exploratory evaluation and the pre-declared stop criteria (never seen before)
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "sealed_stage"))
import build_h1c as G  # noqa: E402
import build_h1c_dev as D  # noqa: E402
from build_dataset import write_jsonl  # noqa: E402

SEED, FID, N_FAM = 2026111406, 51000, 600
OUT = ROOT / "data" / "h1c" / "devx"
PLAN = ROOT / "outputs" / "h1c_devx" / "plan"
N_SELECT, N_EVAL = 100, 200


def main():
    if (OUT / "manifest.json").exists():
        raise SystemExit("H1c DEV extension exists: generated once")
    old = [json.loads(x) for x in (ROOT / "data" / "h1c" / "dev" / "worlds.jsonl").read_text(encoding="utf-8").splitlines()]
    ex = D.exclusion_labels() | D.labels(old)
    rows = G.build(N_FAM, SEED, FID, exclude=ex)
    assert not D.labels(rows) & ex
    assert not {r["family_id"] for r in rows} & {r["family_id"] for r in old}
    assert not {r["prompt_sha256"] for r in rows} & {r["prompt_sha256"] for r in old}
    rng = random.Random(SEED + 7)
    opt = {r["family_id"]: r["oracle"]["optimal_action"] for r in rows}
    role = {f: "TRAIN" for f in {r["family_id"] for r in old}}
    for o in ("A", "B"):
        fams = sorted(f for f, v in opt.items() if v == o)
        rng.shuffle(fams)
        for k, f in enumerate(fams):
            role[f] = "EVAL" if k < N_EVAL // 2 else "SELECT" if k < (N_EVAL + N_SELECT) // 2 else "TRAIN"
    for r in rows:
        r["split"], r["pool"], r["dataset_version"] = role[r["family_id"]], "H1C-DEVX", "h1c-r2"
    OUT.mkdir(parents=True, exist_ok=True)
    man = {"pool": "H1C-DEVX", "seed": SEED, "n_worlds": len(rows), "n_families": len(opt), "sha256": {}}
    man["sha256"]["worlds"] = write_jsonl(rows, OUT / "worlds.jsonl")
    vis = [{k: r[k] for k in ("world_id", "family_id", "variant", "pool", "prompt", "prompt_sha256")} for r in rows]
    man["sha256"]["model_inputs"] = write_jsonl(vis, OUT / "model_inputs.jsonl")
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    PLAN.mkdir(parents=True, exist_ok=True)
    counts = {s: sum(1 for v in role.values() if v == s) for s in ("TRAIN", "SELECT", "EVAL")}
    splits = {"roles": role, "counts_families": counts, "rule": __doc__.strip().splitlines()[-4:],
              "old_dev_families_role": "TRAIN only (already analysed)", "seed": SEED + 7}
    data = json.dumps(splits, indent=1, sort_keys=True).encode()
    (PLAN / "SPLITS.json").write_bytes(data)
    print(json.dumps({**man, "split_counts": counts, "splits_sha256": hashlib.sha256(data).hexdigest()}, indent=1))


if __name__ == "__main__":
    main()
