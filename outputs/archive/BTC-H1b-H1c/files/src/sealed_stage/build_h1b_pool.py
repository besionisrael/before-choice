"""LOCKED-TEST-H1b generator (protocol BTC-H1b v1.0). PREPARED, NOT RUN: refused without an owner approval file.

    python src/sealed_stage/build_h1b_pool.py --approval <file>

* 1,000 depth-4 families (500 optimal-A / 500 optimal-B), seed 2026111403, family ids F40000-F40999
* generator unchanged (build_dataset.build_family + build_path_cf.derive), template oracle.format_prompt;
  KEPT variants: base, cf, upath_base, upath_cf (4,000 worlds); the other generated variants are discarded
* labels disjoint from v1, v2, PROBE-DEV and the consumed LOCKED-TEST (all regenerated in memory from their seeds;
  no sealed file is read)
* written to data/d5/locked_test_h1b/ and sealed immediately (seal LOCKED-TEST-H1B, read-only); only counts and hashes
  are printed
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "sealed_stage"))
from build_dataset import Labeler, build_family, write_jsonl  # noqa: E402
from build_path_cf import derive  # noqa: E402

SEED, FID_BASE, N_FAMILIES = 2026111403, 40000, 1000
KEEP = ("base", "cf", "upath_base", "upath_cf")
VERSION = "d5-h1b"


def exclusion_labels() -> set[str]:
    import build_d5
    ex = build_d5.exclusion_labels("LOCKED-TEST")              # v1 + v2 + PROBE-DEV
    ex |= build_d5.labels_of(build_d5.build("LOCKED-TEST"))    # consumed LOCKED-TEST (in-memory regeneration)
    return ex


def build(n_families: int = N_FAMILIES, seed: int = SEED, fid_base: int = FID_BASE, exclude: set[str] | None = None) -> list[dict]:
    rng = random.Random(seed)
    lab = Labeler(random.Random(seed + 1))
    lab.used |= exclusion_labels() if exclude is None else exclude
    opts = ["A", "B"] * (n_families // 2)
    rng.shuffle(opts)
    rows = []
    for k, opt in enumerate(opts):
        worlds = build_family(fid_base + k, 4, opt, rng, lab)
        derived = [d for w in worlds if w["variant"] in ("base", "cf") for d in (derive(w, "upath"),) if d is not None]
        for w in worlds + derived:
            if w["variant"] in KEEP:
                w["dataset_version"], w["pool"], w["split"] = VERSION, "LOCKED-TEST-H1b", "locked_test_h1b"
                rows.append(w)
    rows.sort(key=lambda w: w["world_id"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--approval", default="")
    a = ap.parse_args()
    if not a.approval or not Path(a.approval).exists():
        raise SystemExit("LOCKED-TEST-H1b generation REFUSED: requires an owner approval file")
    import sealed
    import sealed_access_h1b as SA
    out = SA.POOL_DIR
    if out.exists():
        raise SystemExit("pool directory exists: generation is done once")
    rows = build()
    assert len(rows) == 4 * N_FAMILIES and {r["variant"] for r in rows} == set(KEEP)
    import build_d5
    dev = build_d5.build("PROBE-DEV")
    old = build_d5.build("LOCKED-TEST")                         # in memory; the sealed file is not read
    for other in (dev, old):                                     # independence (families, worlds, labels, prompts)
        assert not {r["family_id"] for r in rows} & {r["family_id"] for r in other}
        assert not {r["world_id"] for r in rows} & {r["world_id"] for r in other}
        assert not build_d5.labels_of(rows) & build_d5.labels_of(other)
        assert not {r["prompt_sha256"] for r in rows} & {r["prompt_sha256"] for r in other}
    out.mkdir(parents=True)
    man = {"pool": "LOCKED-TEST-H1b", "seed": SEED, "n_worlds": len(rows), "n_families": len({r["family_id"] for r in rows}), "sha256": {}}
    man["sha256"]["worlds"] = write_jsonl(rows, out / "worlds.jsonl")
    vis = [{k: r[k] for k in ("world_id", "family_id", "variant", "pool", "prompt", "prompt_sha256")} for r in rows]
    man["sha256"]["model_inputs"] = write_jsonl(vis, out / "model_inputs.jsonl")
    (out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    seal = sealed.seal_dir(out, SA.POOL_SEAL)
    print(json.dumps({"n_worlds": man["n_worlds"], "n_families": man["n_families"], "sha256": man["sha256"], "seal_id": seal["seal_id"]}, indent=1))


if __name__ == "__main__":
    main()
