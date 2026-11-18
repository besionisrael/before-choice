"""BTC-V2 datasets: SCREEN and COMPETENCE pools (CPU). PROBE-DEV / LOCKED-TEST are NOT built here.

Pre-registered in reports/BTC-V2_DESIGN_REVIEW.md §4 (owner-approved V2-3, V2-7):
  * same generator (build_dataset.build_family, build_path_cf.derive) and prompt template as v1
  * master seed 2026111302
  * every state label disjoint from every v1 label (pilot_v1 + pathcf, all splits; v1 is regenerated
    in memory from its own seed for this exclusion only -- no v1 file, answer or result is read)
  * depths 2-4; 50 families per depth, family ids F10000.. (no collision with v1 F0000..F0029)
  * per depth and per optimal action (A/B), families are randomly assigned to
      SCREEN      10 A + 10 B   (20 / depth, 60 total)  variants {base, cf, swap_base, swap_cf}
      COMPETENCE  15 A + 15 B   (30 / depth, 90 total)  all 10 v1 variants + pathcf (upath/cpath/xswap)
  * SCREEN families' other variants are generated (to keep the RNG stream identical) but not saved
Reserved for later (NOT generated): PROBE-DEV and LOCKED-TEST, family ids >= F20000, separate seed.
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
from build_dataset import Labeler, build as build_v1, build_family, write_jsonl  # noqa: E402
from build_path_cf import derive  # noqa: E402

MASTER_SEED = 2026111302
V1_SEED, V1_FPD = 20261113, 10
FID_BASE = 10000
PER_DEPTH = {"SCREEN": 10, "COMPETENCE": 15}  # per depth AND per optimal action
SCREEN_VARIANTS = ("base", "cf", "swap_base", "swap_cf")
DATASET_VERSION = "v2"


def v1_labels() -> set[str]:
    rows = build_v1(V1_FPD, V1_SEED)
    out = set()
    for r in rows:
        for e in r["transitions"]:
            out.update((e["source"], e["target"]))
    for r in rows:  # pathcf only permutes existing labels, but include for completeness
        if r["variant"] in ("base", "cf"):
            for k in ("upath", "cpath", "xswap"):
                w = derive(r, k)
                if w:
                    for e in w["transitions"]:
                        out.update((e["source"], e["target"]))
    out.discard("S")
    return out


def build() -> dict[str, list[dict]]:
    rng = random.Random(MASTER_SEED)
    lab = Labeler(random.Random(MASTER_SEED + 1))
    excluded = v1_labels()
    lab.used |= excluded
    n_fam = 2 * (PER_DEPTH["SCREEN"] + PER_DEPTH["COMPETENCE"])  # 50 per depth
    pools: dict[str, list[dict]] = {"SCREEN": [], "COMPETENCE": []}
    fid = FID_BASE
    for depth in (2, 3, 4):
        opts = ["A", "B"] * (n_fam // 2)
        rng.shuffle(opts)
        fams = []
        for opt in opts:
            fams.append((opt, build_family(fid, depth, opt, rng, lab)))
            fid += 1
        for opt in ("A", "B"):
            idx = [i for i, (o, _) in enumerate(fams) if o == opt]
            rng.shuffle(idx)
            for rank, i in enumerate(idx):
                pool = "SCREEN" if rank < PER_DEPTH["SCREEN"] else "COMPETENCE"
                worlds = fams[i][1]
                if pool == "SCREEN":
                    keep = [w for w in worlds if w["variant"] in SCREEN_VARIANTS]
                else:
                    keep = list(worlds)
                    for w in worlds:
                        if w["variant"] in ("base", "cf"):
                            for k in ("upath", "cpath", "xswap"):
                                d = derive(w, k)
                                if d is not None:
                                    keep.append(d)
                for w in keep:
                    w["dataset_version"] = DATASET_VERSION
                    w["pool"] = pool
                    w["split"] = pool.lower()
                    pools[pool].append(w)
    for p in pools.values():
        p.sort(key=lambda w: w["world_id"])
    return pools


def main() -> None:
    pools = build()
    out_dir = ROOT / "data" / "v2"
    man = {"dataset_version": DATASET_VERSION, "master_seed": MASTER_SEED, "fid_base": FID_BASE,
           "excluded_v1_labels": None, "pools": {}}
    for name, rows in pools.items():
        h = write_jsonl(rows, out_dir / f"{name.lower()}.jsonl")
        vis = [{k: r[k] for k in ("world_id", "family_id", "variant", "pool", "prompt", "prompt_sha256")} for r in rows]
        hv = write_jsonl(vis, out_dir / f"{name.lower()}_model_inputs.jsonl")
        man["pools"][name] = {"n_worlds": len(rows), "n_families": len({r["family_id"] for r in rows}),
                              "sha256": h, "model_inputs_sha256": hv}
    man["excluded_v1_labels"] = len(v1_labels())
    man["not_generated"] = ["PROBE-DEV", "LOCKED-TEST"]
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
