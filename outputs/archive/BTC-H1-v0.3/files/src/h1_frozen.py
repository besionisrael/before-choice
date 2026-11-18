"""Frozen H1 predictors: loading, hash verification, frozen standardisation (BTC-D3-TEST preparation, CPU).

DEV-side module: it never references the sealed pool. It
  * loads the 16 final predictors listed in FINAL.json and refuses any sha256 mismatch (FINAL.json AND the
    freeze manifest `artifacts.H1-DEV-TRAIN`);
  * states which input each predictor consumes (vector index, mean_emb, none, prompt text) from SELECTIONS;
  * recomputes the DEV-train standardisation statistics used by the final refits (h1_train_dev.standardise:
    mean/std over DEV-train training-rule worlds, std < 1e-6 -> 1; candidate table over DEV-train labels) and
    writes them ONCE to a hashed file, so the TEST evaluation applies exactly the DEV transform;
  * offers `apply_bilinear` / `apply_logit`, the only scoring paths used by the TEST evaluator.
Nothing here trains, tunes or rewrites a predictor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import h1_probe as H

ANCHORS = ("c1", "c3", "u1", "u3")
EXPECTED = ({"primary_u2", "order_u1", "order_u3", "B1", "B5", "B3", "B2_perm0", "B2_perm1", "B2_perm2"}
            | {f"anchor_{h}" for h in ANCHORS} | {f"B3o_{h}" for h in ("u1", "u2", "u3")})
STD_FILE = "STANDARDISATION.npz"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def input_spec(final: dict) -> dict:
    """Which input each frozen predictor consumes (from the frozen selections)."""
    lS = final["primary_selection"]["config"][2]
    spec = {"primary_u2": lS, "order_u1": lS, "order_u3": lS, "B2_perm0": lS, "B2_perm1": lS, "B2_perm2": lS,
            "B1": "mean_emb", "B5": "none", "B3": "text", "B3o_u1": "text", "B3o_u2": "text", "B3o_u3": "text"}
    for h in ANCHORS:
        spec[f"anchor_{h}"] = final["anchor_selection"][h]["config"][2]
    return spec


def load_final(out_dir: Path, freeze_predictors: dict | None = None) -> dict:
    """Load and hash-verify the 16 final predictors. Returns {name: {...}} and the FINAL status."""
    final = json.loads((out_dir / "FINAL.json").read_text(encoding="utf-8"))
    preds = final["final_refits"]["predictors"]
    if set(preds) != EXPECTED:
        raise ValueError(f"predictor set differs from the frozen 16: {sorted(set(preds) ^ EXPECTED)}")
    spec = input_spec(final)
    out = {}
    for name, info in preds.items():
        p = out_dir / info["path"]
        h = sha(p)
        if h != info["sha256"]:
            raise ValueError(f"{name}: sha256 mismatch vs FINAL.json")
        if freeze_predictors is not None and h != freeze_predictors[name]:
            raise ValueError(f"{name}: sha256 mismatch vs freeze manifest")
        obj = torch.load(p, weights_only=False)
        rec = {"sha256": h, "input": spec[name], "slot": obj["slot"], "wd": obj["wd"]}
        if "w" in obj:
            rec.update(kind="logit", w=obj["w"])
        else:
            rec.update(kind="bilinear", r=obj["r"], E=obj["E"],
                       fits=[H.Fit(U, V, obj["E"], float("nan")) for U, V in zip(obj["U"], obj["V"])])
            if "permutation" in obj:
                rec["permutation"] = obj["permutation"]
        out[name] = rec
    return {"predictors": out, "final": final}


def freeze_standardisation(dev, final: dict, path: Path) -> str:
    """Recompute the DEV-train statistics of every input used by a frozen predictor; write once; return sha256."""
    if path.exists():
        raise SystemExit(f"{path.name} exists: frozen statistics are never rewritten")
    tr = dev.by_split["train"]
    arrays = {}
    for ell in sorted({v for v in input_spec(final).values() if isinstance(v, int)}):
        a = dev.layer(ell, tr)
        arrays[f"z{ell}_mu"], arrays[f"z{ell}_sd"] = _musd(a)
    arrays["mean_emb_mu"], arrays["mean_emb_sd"] = _musd(dev.mean_emb(tr))
    train_labels = sorted({dev.label_index[l] for i in tr for l in dev.rows[i]["cands"]})
    arrays["T_mu"], arrays["T_sd"] = _musd(dev.T_raw[train_labels])
    np.savez(path, **arrays)
    return sha(path)


def _musd(a: np.ndarray):
    mu, sd = a.mean(0), a.std(0)
    return mu, np.where(sd < 1e-6, 1.0, sd)


def load_standardisation(path: Path, expected_sha: str) -> dict:
    if sha(path) != expected_sha:
        raise ValueError("standardisation file sha256 mismatch")
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def standardise_with(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(((x - mu) / sd).astype(np.float32))


def apply_bilinear(pred: dict, z: torch.Tensor, cand: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    assert pred["kind"] == "bilinear"
    return H.predict_proba(pred["fits"], z, cand, T)


def apply_logit(pred: dict, X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    assert pred["kind"] == "logit"
    return torch.softmax((X @ pred["w"]).masked_fill(M, float("-inf")), -1)
