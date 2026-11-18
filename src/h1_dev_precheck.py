"""Pre-training checks on PROBE-DEV captured activations (NO probe training; nothing here feeds back into the
frozen grid). Writes outputs/h1_dev/precheck.json.

1. integrity: 12 shards, candidate embeddings, mean_emb, 4,800 ids in input order, MANIFEST sha = freeze manifest
2. non-decoy format violations by depth x variant x split
3. exact training / validation counts under H1 v0.3 §3 (non-decoy, forced choice correct; ties not correct)
4. train / val family disjointness
5. dimensions, finiteness, descriptive distributions per vector index (norms, mean, sd, min, max)
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from d3_capture_core import u16_to_bf16_f32, verify_shards  # noqa: E402

D = ROOT / "outputs" / "d3" / "probe_dev"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def choice(r):
    return "A" if r["logp_A"] > r["logp_B"] else "B" if r["logp_A"] < r["logp_B"] else "TIE"


def main() -> None:
    out: dict = {}
    man = json.loads((D / "MANIFEST.json").read_text(encoding="utf-8"))
    frz = json.loads((ROOT / "reports" / "BTC-H1_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    art = frz["artifacts"]["D3-DEV-R1_activations"]
    out["integrity"] = {"verify_shards": verify_shards(D), "n_shards": len(man["shards"]),
                        "MANIFEST_sha_matches_freeze": sha(D / "MANIFEST.json") == art["MANIFEST.json_sha256"],
                        "candidate_embeddings_sha_matches_freeze": sha(D / "candidate_embeddings.npy") == art["candidate_embeddings_sha256"]}
    idx = []
    for s in man["shards"]:
        idx += [json.loads(x) for x in (D / f"{s['name']}_index.jsonl").read_text(encoding="utf-8").splitlines()]
    inputs = [json.loads(x) for x in (ROOT / "data/d5/probe_dev/model_inputs.jsonl").read_text(encoding="utf-8").splitlines()]
    worlds = {r["world_id"]: r for r in (json.loads(x) for x in (ROOT / "data/d5/probe_dev/worlds.jsonl").read_text(encoding="utf-8").splitlines())}
    out["integrity"]["ids_in_input_order"] = [r["world_id"] for r in idx] == [r["world_id"] for r in inputs]
    out["integrity"]["n_ids_unique"] = len({r["world_id"] for r in idx})

    # 2. non-decoy format violations
    viol = [r for r in idx if not worlds[r["world_id"]]["variant"].startswith("decoy")
            and (not r["top1_in_AB"] or np.exp(r["logp_A"]) + np.exp(r["logp_B"]) < 0.5)]
    out["format_violations_non_decoy"] = {
        "n": len(viol),
        "by_depth": dict(Counter(worlds[r["world_id"]]["depth"] for r in viol)),
        "by_variant": dict(Counter(worlds[r["world_id"]]["variant"] for r in viol)),
        "by_split": dict(Counter(worlds[r["world_id"]]["split"] for r in viol)),
        "by_depth_variant_split": {f"d{k[0]}|{k[1]}|{k[2]}": v for k, v in sorted(Counter(
            (worlds[r["world_id"]]["depth"], worlds[r["world_id"]]["variant"], worlds[r["world_id"]]["split"]) for r in viol).items())},
        "top1_tokens": dict(Counter(r["top5"][0][1] for r in viol).most_common(10)),
        "forced_choice_correct_among_them": sum(choice(r) == worlds[r["world_id"]]["oracle"]["optimal_action"] for r in viol),
        "world_ids": sorted(r["world_id"] for r in viol)}

    # 3. H1 v0.3 §3 training-rule counts
    rule = defaultdict(Counter)
    for r in idx:
        w = worlds[r["world_id"]]
        if w["variant"].startswith("decoy"):
            rule["decoy_excluded"][w["split"]] += 1
            continue
        c = choice(r)
        key = "tie_excluded" if c == "TIE" else ("included" if c == w["oracle"]["optimal_action"] else "wrong_choice_excluded")
        rule[key][(w["split"], w["depth"])] += 1
    out["training_rule_counts"] = {k: {f"{s}|d{d}" if isinstance(t, tuple) else t: n for t, n in v.items()
                                       for s, d in ([t] if isinstance(t, tuple) else [(t, "")])} for k, v in rule.items()}
    inc = rule["included"]
    out["training_rule_totals"] = {"train": sum(n for (s, d), n in inc.items() if s == "train"),
                                   "val": sum(n for (s, d), n in inc.items() if s == "val")}
    # selection population: DEV-val admissible depth-4 upath pairs (both members forced-correct, no tie)
    byid = {r["world_id"]: r for r in idx}
    fam = defaultdict(dict)
    for wid, w in worlds.items():
        fam[w["family_id"]][w["variant"]] = w
    adm = Counter()
    for f, v in fam.items():
        if v["base"]["depth"] != 4:
            continue
        for src in ("base", "cf"):
            a, b = v[src], v[f"upath_{src}"]
            ok = all(choice(byid[x["world_id"]]) == x["oracle"]["optimal_action"] for x in (a, b))
            adm[(a["split"], "admissible" if ok else "not_admissible")] += 1
    out["depth4_upath_pairs"] = {f"{s}|{k}": n for (s, k), n in sorted(adm.items())}
    out["ab_ties_total"] = sum(choice(r) == "TIE" for r in idx)

    # 4. family disjointness
    fs = defaultdict(set)
    for w in worlds.values():
        fs[w["split"]].add(w["family_id"])
    out["family_split"] = {"train_families": len(fs["train"]), "val_families": len(fs["val"]),
                           "overlap": len(fs["train"] & fs["val"])}

    # 5. representation descriptives (no training; for the record only)
    stats = {"n": 0, "sum": np.zeros(62), "sumsq": np.zeros(62), "norm_sum": np.zeros(62),
             "min": np.full(62, np.inf), "max": np.full(62, -np.inf), "nonfinite": 0}
    emb_nonfinite, emb_norms = 0, []
    for s in man["shards"]:
        a = np.load(D / f"{s['name']}_acts.npy", mmap_mode="r")
        assert a.shape[1:] == (62, 5376) and a.dtype == np.uint16
        for i in range(0, a.shape[0], 50):
            x = u16_to_bf16_f32(np.asarray(a[i:i + 50])).astype(np.float64)  # (b, 62, d)
            stats["nonfinite"] += int((~np.isfinite(x)).sum())
            stats["n"] += x.shape[0]
            stats["sum"] += x.mean(-1).sum(0)
            stats["sumsq"] += (x ** 2).mean(-1).sum(0)
            stats["norm_sum"] += np.linalg.norm(x, axis=-1).sum(0)
            stats["min"] = np.minimum(stats["min"], x.min(axis=(0, 2)))
            stats["max"] = np.maximum(stats["max"], x.max(axis=(0, 2)))
        e = np.load(D / f"{s['name']}_mean_emb.npy")
        assert e.shape[1:] == (5376,) and e.dtype == np.float32
        emb_nonfinite += int((~np.isfinite(e)).sum())
        emb_norms += list(np.linalg.norm(e, axis=1))
    n = stats["n"]
    mean = stats["sum"] / n
    sd = np.sqrt(np.maximum(stats["sumsq"] / n - mean ** 2, 0))
    out["representations"] = {
        "n_worlds": n, "shape_per_world": [62, 5376], "nonfinite_acts": stats["nonfinite"], "nonfinite_mean_emb": emb_nonfinite,
        "per_vector": {int(k): {"mean_l2_norm": round(float(stats["norm_sum"][k] / n), 3), "elem_mean": round(float(mean[k]), 4),
                                "elem_sd": round(float(sd[k]), 4), "min": round(float(stats["min"][k]), 3), "max": round(float(stats["max"][k]), 3)}
                       for k in range(62)},
        "mean_emb_l2_norm": {"mean": round(float(np.mean(emb_norms)), 3), "min": round(float(np.min(emb_norms)), 3), "max": round(float(np.max(emb_norms)), 3)}}
    ce = np.load(D / "candidate_embeddings.npy")
    out["candidate_embeddings"] = {"shape": list(ce.shape), "finite": bool(np.isfinite(ce).all()),
                                   "l2_norm_mean": round(float(np.linalg.norm(ce, axis=1).mean()), 3)}
    (ROOT / "outputs" / "h1_dev").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "h1_dev" / "precheck.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "representations"}, indent=1)[:6000])
    rep = out["representations"]
    print("nonfinite acts/mean_emb:", rep["nonfinite_acts"], rep["nonfinite_mean_emb"])
    for k in (0, 1, 15, 30, 45, 59, 60, 61):
        print("vector", k, rep["per_vector"][k])


if __name__ == "__main__":
    main()
