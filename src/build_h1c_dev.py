"""BTC-H1c DEV pool (CPU; owner decision "BTC-H1c ... GO pour préparer l'exécution DEV"). Not a TEST pool, not sealed.

    python src/build_h1c_dev.py        -> data/h1c/dev/{worlds.jsonl, model_inputs.jsonl, manifest.json}

* 400 families (200 optimal-A / 200 optimal-B) x 5 variants (base_L, dswap_L, base_E, dswap_E, cswap_L) = 2,000 worlds,
  seed 2026111404, family ids F50000-F50399, task revision 2 (dead-end chains of route length, 12 candidates)
* family-level split stratified by optimal action: train 75 % / val 25 % (selection uses DISTINCT families)
* labels disjoint from v1, v2, PROBE-DEV, the consumed LOCKED-TEST and the (never generated) LOCKED-TEST-H1b design
  (all regenerated in memory from their seeds; no sealed file is read)
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
from build_dataset import write_jsonl  # noqa: E402

SEED, FID, N_FAM, TRAIN_FRAC = 2026111404, 50000, 400, 0.75
OUT = ROOT / "data" / "h1c" / "dev"


def exclusion_labels() -> set[str]:
    import build_d5
    import build_h1b_pool
    ex = build_d5.exclusion_labels("LOCKED-TEST") | build_d5.labels_of(build_d5.build("LOCKED-TEST"))
    ex |= build_d5.labels_of(build_h1b_pool.build(exclude=ex))
    return ex


def labels(rows):
    s = set()
    for r in rows:
        for e in r["transitions"]:
            s.update((e["source"], e["target"]))
    return s - {"S"}


def build():
    ex = exclusion_labels()
    rows = G.build(N_FAM, SEED, FID, exclude=ex)
    assert not labels(rows) & ex
    rng = random.Random(SEED + 7)
    fam_opt = {}
    for r in rows:
        fam_opt[r["family_id"]] = r["oracle"]["optimal_action"]
    split = {}
    for opt in ("A", "B"):
        fams = sorted(f for f, o in fam_opt.items() if o == opt)
        rng.shuffle(fams)
        n_tr = round(TRAIN_FRAC * len(fams))
        for k, f in enumerate(fams):
            split[f] = "train" if k < n_tr else "val"
    for r in rows:
        r["split"], r["pool"], r["dataset_version"] = split[r["family_id"]], "H1C-DEV", "h1c-r2"
    return rows


def main():
    if (OUT / "manifest.json").exists():
        raise SystemExit("H1c DEV pool exists: generated once")
    rows = build()
    OUT.mkdir(parents=True, exist_ok=True)
    man = {"pool": "H1C-DEV", "seed": SEED, "n_worlds": len(rows), "n_families": len({r["family_id"] for r in rows}),
           "variants": list(G.VARIANTS), "split_families": {s: len({r["family_id"] for r in rows if r["split"] == s}) for s in ("train", "val")},
           "sha256": {}}
    man["sha256"]["worlds"] = write_jsonl(rows, OUT / "worlds.jsonl")
    vis = [{k: r[k] for k in ("world_id", "family_id", "variant", "pool", "prompt", "prompt_sha256")} for r in rows]
    man["sha256"]["model_inputs"] = write_jsonl(vis, OUT / "model_inputs.jsonl")
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
