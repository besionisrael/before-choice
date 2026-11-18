"""BTC-H1c DEV-extension analysis (CPU; EXPLORATORY). PREPARED, NOT RUN on real data.
Requires both GATES files (H1c DEV and DEV extension) to say analysis_allowed.

    python src/h1c_devx_analysis.py --run
    python src/h1c_devx_analysis.py --synthetic-test <dir>

Data: H1c DEV capture (slots 0-12 used) + DEV-extension capture (13 slots), layers {32, 40, 48}; admissible worlds only.
Family roles frozen in outputs/h1c_devx/plan/SPLITS.json: TRAIN (fit), SELECT (early stopping + layer choice), EVAL
(exploratory evaluation; 200 never-seen families).
Pre-declared procedure:
  1  LAYER = argmax over {32, 40, 48} of PC1 (12-way c1 pointer: ANS query x 12 arrival keys) accuracy on SELECT
     (fit on TRAIN; ties -> lower layer). u2 is NEVER used for selection.
  2  at LAYER, on EVAL: PC1; PC2 = binary CHOSEN continuation (successors of the chosen depth-1 label); P = binary
     UNCHOSEN continuation (robust sign-flip rule on late pairs, A1); controls key-only / choice x keys / mu_choice
     query (0.5 by construction on late pairs), P_perm; early pairs.
Pre-declared stop criteria (instrument adequacy; evaluated on EVAL only):
  SC1  PC1 >= 0.80                                   (sensitivity)
  SC2  |PC1(SELECT) - PC1(EVAL)| <= 0.10              (generalisation)
  SC3  PC2 > 0.5 with one-sided sign-flip p < 0.025  (the instrument can decode a continuation)
  any failure -> INSTRUMENT INSUFFICIENT: close H1c, no TEST.   all pass -> INSTRUMENT ADEQUATE: re-examine a frozen
  confirmatory protocol (the unchosen-continuation result is reported, exploratory, and does not enter SC1-SC3).
Decoders: pointer r = 8, lambda = 0.1, seeds 0, 1, 2, early stopping on SELECT (no hyper-parameter search).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import h1_probe as H  # noqa: E402
import h1c_dev_analysis as A  # noqa: E402
import h1c_positions as HP  # noqa: E402
from h1b_x3x4_analysis import _train_two_tables  # noqa: E402

LAYERS = (32, 40, 48)
N_SLOTS = 13
CRIT = {"SC1_PC1_min": 0.80, "SC2_select_eval_max_gap": 0.10, "SC3_PC2_signflip_alpha": 0.025}


class CapX:
    """Combined admissible rows of several captures; roles from SPLITS.json; layer() -> {world: (13, d)}."""

    def __init__(self, caps, splits_path: Path):
        roles = json.loads(splits_path.read_text(encoding="utf-8"))["roles"]
        self.parts = [A.Cap(c, d, None) for c, d in caps]
        self.layers = [l for l in LAYERS if all(l in p.layers for p in self.parts)]
        self.rows, self.W = [], {}
        for p in self.parts:
            self.rows += p.rows
            self.W.update(p.W)
        for r in self.rows:
            r["role"] = roles[r["family_id"]]
        self.itr = [k for k, r in enumerate(self.rows) if r["role"] == "TRAIN"]
        self.iva = [k for k, r in enumerate(self.rows) if r["role"] == "SELECT"]
        self.va = [k for k, r in enumerate(self.rows) if r["role"] == "EVAL"]
        self._cache = {}

    def layer(self, ell):
        if ell not in self._cache:
            self._cache.clear()
            out = {}
            for p in self.parts:
                out.update({w: a[:N_SLOTS] for w, a in p.layer(ell).items()})
                p._cache.clear()
            self._cache[ell] = out
        return self._cache[ell]


def c1_inputs(cap, ell, ids):
    L = cap.layer(ell)
    z = np.stack([L[cap.rows[k]["world_id"]][0] for k in ids])
    keys = np.concatenate([L[cap.rows[k]["world_id"]][1:13] for k in ids])
    return z, keys, torch.arange(len(ids) * 12).view(len(ids), 12), torch.tensor([cap.rows[k]["c1_slot"] for k in ids])


def fit(inp, cap, ell, seeds, max_epochs):
    a, b = inp(cap, ell, cap.itr), inp(cap, ell, cap.iva)
    (Za, Zb), (Ka, Kb) = A.std(a[0], b[0]), A.std(a[1], b[1])
    fits = [_train_two_tables(Za, a[2], a[3], Ka, Zb, b[2], b[3], Kb, 8, 0.1, s, max_epochs) for s in seeds]

    def score(ids):
        o = inp(cap, ell, ids)
        Zo, Ko = A.std(a[0], o[0])[1], A.std(a[1], o[1])[1]
        return (H.argmax_canonical(H.predict_proba(fits, Zo, o[2], Ko)) == o[3]).float().numpy()
    return score


def signflip_all(cap, ids, ok, R=10_000, seed=0):
    fam = {}
    for j, k in enumerate(ids):
        fam.setdefault(cap.rows[k]["family_id"], []).append(ok[j])
    d = np.array([np.mean(v) - 0.5 for v in fam.values()])
    null = (np.random.default_rng(seed).choice((-1.0, 1.0), size=(R, len(d))) * d).mean(1)
    return {"n_families": int(len(d)), "acc": float(d.mean() + 0.5), "signflip_p_one_sided": float((np.sum(null >= d.mean()) + 1) / (R + 1))}


def fam_ci(cap, ids, ok, B=10_000, seed=0, subset=None):
    """Reporting only: 95 % family-cluster percentile bootstrap CI of the mean accuracy over `ids` (optionally a subset)."""
    fam = {}
    for j, k in enumerate(ids):
        if subset is None or subset(cap.rows[k]):
            fam.setdefault(cap.rows[k]["family_id"], []).append(ok[j])
    keys = sorted(fam)
    if not keys:
        return {"n_families": 0}
    S = np.array([sum(fam[f]) for f in keys])
    N = np.array([len(fam[f]) for f in keys], dtype=float)
    W = np.random.default_rng(seed).multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=B).astype(float)
    m = (W @ S) / (W @ N)
    return {"n_families": len(keys), "n_worlds": int(N.sum()), "acc": float(S.sum() / N.sum()),
            "ci95": [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]}


def run(cap: CapX, out: Path, seeds=(0, 1, 2), max_epochs=H.MAX_EPOCHS) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DEVX_ANALYSIS.json").exists():
        raise SystemExit("single analysis only")
    H.set_threads(4)
    meter = H.BudgetMeter(2.0)
    res = {"status": "EXPLORATORY", "n": {"TRAIN": len(cap.itr), "SELECT": len(cap.iva), "EVAL": len(cap.va)}, "layer_selection": {}}
    sc = {}
    for ell in cap.layers:
        sc[ell] = fit(c1_inputs, cap, ell, seeds, max_epochs)
        res["layer_selection"][str(ell)] = {"PC1_SELECT": float(sc[ell](cap.iva).mean())}
        meter.tick(f"PC1|{ell}")
    L = max(cap.layers, key=lambda l: (res["layer_selection"][str(l)]["PC1_SELECT"], -l))
    res["selected_layer"] = L
    pc1_sel, pc1_eval, pc1_tr = float(sc[L](cap.iva).mean()), float(sc[L](cap.va).mean()), float(sc[L](cap.itr).mean())
    sc2 = fit(lambda c, l, ids: A.binary_inputs(c, l, ids, chosen=True), cap, L, seeds, max_epochs)
    ok2 = sc2(cap.va)
    pc2 = signflip_all(cap, cap.va, ok2)
    scu = fit(lambda c, l, ids: A.binary_inputs(c, l, ids), cap, L, seeds, max_epochs)
    oku = scu(cap.va)
    meter.tick("binary")
    res["PC1"] = {"TRAIN": pc1_tr, "SELECT": pc1_sel, "EVAL": pc1_eval,
                  "EVAL_ci": fam_ci(cap, cap.va, sc[L](cap.va)), "SELECT_ci": fam_ci(cap, cap.iva, sc[L](cap.iva))}
    late = lambda r: r["variant"] in ("base_L", "dswap_L")
    res["uncertainty_EVAL"] = {"PC2_all": fam_ci(cap, cap.va, ok2), "P_unchosen_all": fam_ci(cap, cap.va, oku),
                               "P_unchosen_late_worlds": fam_ci(cap, cap.va, oku, subset=late),
                               "admissible": {"EVAL_worlds": len(cap.va), "EVAL_families": len({cap.rows[k]["family_id"] for k in cap.va})}}
    res["PC2_chosen_continuation_EVAL"] = pc2
    res["P_unchosen_EVAL"] = {"single_all": float(oku.mean()), "robust_late": A.signflip(cap, cap.va, oku),
                              "robust_early": A.signflip(cap, cap.va, oku, variants=("base_E", "dswap_E")),
                              "paired_late_descriptive": A.paired(cap, cap.va, oku)}
    crit = {"SC1": pc1_eval >= CRIT["SC1_PC1_min"], "SC2": abs(pc1_sel - pc1_eval) <= CRIT["SC2_select_eval_max_gap"],
            "SC3": pc2["acc"] > 0.5 and pc2["signflip_p_one_sided"] < CRIT["SC3_PC2_signflip_alpha"]}
    res["stop_criteria"] = {"thresholds": CRIT, "results": crit,
                            "decision": "INSTRUMENT ADEQUATE: re-examine a frozen confirmatory protocol" if all(crit.values())
                            else "INSTRUMENT INSUFFICIENT: close H1c, no TEST"}
    res["cpu_hours"] = meter.used_s() / 3600
    (out / "DEVX_ANALYSIS.json").write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--synthetic-test", default="")
    a = ap.parse_args()
    if a.synthetic_test:
        b = Path(a.synthetic_test)
        c1, d1 = A.make_synthetic(b / "p1", n_fam=60, seed=1)
        c2, d2 = A.make_synthetic(b / "p2", n_fam=60, seed=2)
        W = [json.loads(x) for x in (d1 / "worlds.jsonl").read_text(encoding="utf-8").splitlines()]
        fams = sorted({w["family_id"] for w in W})
        roles = {f: ("TRAIN" if i % 3 == 0 else "SELECT" if i % 3 == 1 else "EVAL") for i, f in enumerate(fams)}
        (b / "SPLITS.json").write_text(json.dumps({"roles": roles}), encoding="utf-8")
        print(json.dumps(run(CapX([(c1, d1)], b / "SPLITS.json"), b / "out", seeds=(0,), max_epochs=30), indent=1, default=float))
        return
    if not a.run:
        raise SystemExit("use --run (after an authorised capture whose gates passed) or --synthetic-test")
    for g in (ROOT / "outputs" / "h1c_dev" / "capture" / "GATES.json", ROOT / "outputs" / "h1c_devx" / "capture" / "GATES.json"):
        if not json.loads(g.read_text(encoding="utf-8"))["analysis_allowed"]:
            raise SystemExit(f"ANALYSIS REFUSED: gates failed in {g}")
    cap = CapX([(ROOT / "outputs" / "h1c_dev" / "capture", ROOT / "data" / "h1c" / "dev"),
                (ROOT / "outputs" / "h1c_devx" / "capture", ROOT / "data" / "h1c" / "devx")], ROOT / "outputs" / "h1c_devx" / "plan" / "SPLITS.json")
    print(json.dumps(run(cap, ROOT / "outputs" / "h1c_devx" / "analysis"), indent=1, default=float))


if __name__ == "__main__":
    main()
