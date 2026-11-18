"""D5 pools for the H1 confirmatory study (BTC-D5_DATA_DESIGN, owner-validated sizes and seeds).

    python src/build_d5.py --pool PROBE-DEV                      # authorised 2026-11-13
    python src/build_d5.py --pool LOCKED-TEST --approval <file>   # REFUSED until the revised H1 freeze is approved

Common rules (pre-registered):
  * generator unchanged: build_dataset.build_family (10 v1 variants) + build_path_cf.derive (upath/cpath/xswap
    from base and cf) = 16 variants per family; template oracle.format_prompt; rewards 1-9; decoy rewards 10-12;
  * depth 4: 200 families (100 optimal-A / 100 optimal-B); depth 3: 100 families (50/50); no depth 2, 5, 6; no D-8;
  * labels disjoint from v1 (seed 20261113), v2 (seed 2026111302) and -- for LOCKED-TEST -- PROBE-DEV; all
    exclusion sets regenerated in memory from their seeds (no data file is read);
  * PROBE-DEV: seed 2026111401, family ids F20000-F20299, family-level split stratified by depth x optimal action,
    train 75 % / val 25 %;  LOCKED-TEST: seed 2026111402, ids F30000-F30299, written to data/d5/locked_test/ and sealed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from build_dataset import Labeler, build_family, write_jsonl  # noqa: E402
from build_path_cf import derive  # noqa: E402

POOLS = {
    "PROBE-DEV": {"seed": 2026111401, "fid_base": 20000, "out": ROOT / "data" / "d5" / "probe_dev"},
    "LOCKED-TEST": {"seed": 2026111402, "fid_base": 30000, "out": ROOT / "data" / "d5" / "locked_test"},
}
N_FAMILIES = {4: 200, 3: 100}
TRAIN_FRAC = 0.75
VERSION = "d5"


def labels_of(rows) -> set[str]:
    s = set()
    for r in rows:
        for e in r["transitions"]:
            s.update((e["source"], e["target"]))
    s.discard("S")
    return s


def exclusion_labels(pool: str) -> set[str]:
    from build_v2 import build as build_v2, v1_labels
    ex = set(v1_labels())
    v2 = build_v2()
    ex |= labels_of(v2["SCREEN"] + v2["COMPETENCE"])
    if pool == "LOCKED-TEST":
        ex |= labels_of(build("PROBE-DEV"))
    return ex


def build(pool: str) -> list[dict]:
    cfg = POOLS[pool]
    rng = random.Random(cfg["seed"])
    lab = Labeler(random.Random(cfg["seed"] + 1))
    lab.used |= exclusion_labels(pool)
    rows, fid = [], cfg["fid_base"]
    for depth in (4, 3):
        n = N_FAMILIES[depth]
        opts = ["A", "B"] * (n // 2)
        rng.shuffle(opts)
        fams = []
        for opt in opts:
            worlds = build_family(fid, depth, opt, rng, lab)
            extra = [d for w in worlds if w["variant"] in ("base", "cf")
                     for d in (derive(w, k) for k in ("upath", "cpath", "xswap")) if d is not None]
            fams.append((opt, worlds + extra))
            fid += 1
        for opt in ("A", "B"):  # family-level split stratified by depth x optimal action
            idx = [i for i, (o, _) in enumerate(fams) if o == opt]
            rng.shuffle(idx)
            n_train = round(TRAIN_FRAC * len(idx))
            for rank, i in enumerate(idx):
                part = "train" if rank < n_train else "val"
                for w in fams[i][1]:
                    w["dataset_version"] = VERSION
                    w["pool"] = pool
                    w["split"] = part if pool == "PROBE-DEV" else "locked_test"
                    rows.append(w)
    rows.sort(key=lambda w: w["world_id"])
    return rows


def write(pool: str, rows: list[dict]) -> dict:
    out = POOLS[pool]["out"]
    out.mkdir(parents=True, exist_ok=True)
    man = {"pool": pool, "seed": POOLS[pool]["seed"], "n_worlds": len(rows),
           "n_families": len({r["family_id"] for r in rows}), "sha256": {}}
    man["sha256"]["all"] = write_jsonl(rows, out / "worlds.jsonl")
    vis = [{k: r[k] for k in ("world_id", "family_id", "variant", "pool", "prompt", "prompt_sha256")} for r in rows]
    man["sha256"]["model_inputs"] = write_jsonl(vis, out / "model_inputs.jsonl")
    if pool == "PROBE-DEV":
        for part in ("train", "val"):
            man["sha256"][part] = write_jsonl([r for r in rows if r["split"] == part], out / f"{part}.jsonl")
    (out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    return man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, choices=list(POOLS))
    ap.add_argument("--approval", default="")
    a = ap.parse_args()
    if a.pool == "LOCKED-TEST":
        appr = Path(a.approval) if a.approval else None
        if not appr or not appr.exists():
            raise SystemExit("LOCKED-TEST generation REFUSED: requires an owner approval file "
                             "(revised H1 freeze must be validated first)")
    rows = build(a.pool)
    man = write(a.pool, rows)
    if a.pool == "LOCKED-TEST":
        from sealed import seal_dir
        seal_dir(POOLS[a.pool]["out"], "LOCKED-TEST-D5")
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
