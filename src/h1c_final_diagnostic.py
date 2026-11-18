"""BTC-H1c final diagnostic (CPU; POST HOC / EXPLORATORY; existing DEV + DEVX activations; layer 40; EVAL families).

    python src/h1c_final_diagnostic.py   -> outputs/h1c_devx/final_diagnostic/FINAL_DIAGNOSTIC.json  (1 CPU-h cap)

Same frozen roles (TRAIN fit / SELECT early stopping / EVAL report), same layer 40, same decoder family (pointer r = 8,
lambda 0.1, seeds 0-2; linear controls L-BFGS lambda 0.01 as in the DEV analysis), same robust statistic (per-family
mean correctness over the worlds of complete pairs - 0.5; one-sided sign-flip; family-bootstrap 95 % CI). No search.
  E1  EARLY pairs (base_E, dswap_E), UNCHOSEN continuation: pointer P vs key-only, choice x keys (C_lin) and mu_choice
      query (C_bil); difference P - best control per family (sign-flip, CI). Arrival-key identity within early pairs is
      measured (expected: NOT identical, because the departure lines precede the arrivals).
  E2  CSWAP pairs (base_L, cswap_L), CHOSEN continuation: chosen pointer (PC2 decoder) vs the same controls; on all cswap
      pairs and on the subset whose chosen binary arrival keys are bitwise identical within the pair (the condition
      that makes them comparable to the late unchosen pairs).
  E3  branch comparison on EVAL: chosen (E2, identical-key subset) vs unchosen (late pairs, base_L/dswap_L) robust
      statistics, per-family difference where both are available, with CI; admissibility and population sizes listed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import h1_probe as H  # noqa: E402
import h1c_dev_analysis as A  # noqa: E402
import h1c_devx_analysis as X  # noqa: E402
from h1b_x3x4_analysis import _train_two_tables  # noqa: E402

LAYER, SEEDS, R_FLIP, B = 40, (0, 1, 2), 10_000, 10_000


def pair_index(cap, ids, variants):
    pos = {cap.rows[k]["world_id"]: j for j, k in enumerate(ids)}
    out = {}
    for j, k in enumerate(ids):
        r = cap.rows[k]
        if r["variant"] == variants[0]:
            b = f"{r['family_id']}-{variants[1]}"
            if b in pos:
                out.setdefault(r["family_id"], []).append((j, pos[b]))
    return out


def robust(pairs, ok, keep=None, seed=0):
    fam = {f: [x for p in ps if keep is None or keep(p) for x in (ok[p[0]], ok[p[1]])] for f, ps in pairs.items()}
    fam = {f: v for f, v in fam.items() if v}
    if not fam:
        return {"n_families": 0}
    d = np.array([np.mean(v) - 0.5 for v in fam.values()])
    rng = np.random.default_rng(seed)
    null = (rng.choice((-1.0, 1.0), size=(R_FLIP, len(d))) * d).mean(1)
    S = np.array([sum(v) for v in fam.values()])
    N = np.array([len(v) for v in fam.values()], dtype=float)
    W = rng.multinomial(len(d), np.full(len(d), 1 / len(d)), size=B).astype(float)
    m = (W @ S) / (W @ N)
    return {"n_families": int(len(d)), "n_worlds": int(N.sum()), "acc": float(S.sum() / N.sum()),
            "ci95": [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))],
            "signflip_p_one_sided": float((np.sum(null >= d.mean()) + 1) / (R_FLIP + 1))}


def diff(pairs_a, ok_a, pairs_b, ok_b, keep_a=None, keep_b=None, seed=0):
    """Per-family difference of mean correctness (a - b) over families present in both; two-sided sign-flip p; CI."""
    fa = {f: np.mean([x for p in ps if keep_a is None or keep_a(p) for x in (ok_a[p[0]], ok_a[p[1]])] or [np.nan]) for f, ps in pairs_a.items()}
    fb = {f: np.mean([x for p in ps if keep_b is None or keep_b(p) for x in (ok_b[p[0]], ok_b[p[1]])] or [np.nan]) for f, ps in pairs_b.items()}
    fams = [f for f in fa if f in fb and not np.isnan(fa[f]) and not np.isnan(fb[f])]
    if not fams:
        return {"n_families": 0}
    d = np.array([fa[f] - fb[f] for f in fams])
    rng = np.random.default_rng(seed)
    null = np.abs((rng.choice((-1.0, 1.0), size=(R_FLIP, len(d))) * d).mean(1))
    boots = d[rng.integers(0, len(d), size=(B, len(d)))].mean(1)
    return {"n_families": len(fams), "mean_diff": float(d.mean()), "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "signflip_p_two_sided": float((np.sum(null >= abs(d.mean())) + 1) / (R_FLIP + 1))}


def controls(cap, chosen, meter):
    """Key-only and choice x keys (linear) and mu_choice-query bilinear, trained on TRAIN, correctness on EVAL."""
    zi, ki, ci, yi = A.binary_inputs(cap, LAYER, cap.itr, chosen=chosen)
    zv, kv, cv, yv = A.binary_inputs(cap, LAYER, cap.iva, chosen=chosen)
    zo, ko, co, yo = A.binary_inputs(cap, LAYER, cap.va, chosen=chosen)
    Ki, Kv, Ko = A.std(ki, kv, ko)
    ch = lambda ids: torch.tensor([float(cap.rows[k]["choice"] == "A") for k in ids])
    out = {}
    for name, use_ch in (("key_only", False), ("C_lin", True)):
        Xi, Mi = A.lin_features(Ki, ci, ch(cap.itr) if use_ch else None)
        Xo, Mo = A.lin_features(Ko, co, ch(cap.va) if use_ch else None)
        w = H.train_conditional_logit(Xi, Mi, yi, 0.01, meter, name)
        out[name] = (H.argmax_canonical(torch.softmax(Xo @ w, -1)) == yo).float().numpy()
    mus = {c: zi[[cap.rows[k]["choice"] == c for k in cap.itr]].mean(0) for c in ("A", "B")}
    Q = lambda ids: np.stack([mus[cap.rows[k]["choice"]] for k in ids])
    QI = A.std(zi, Q(cap.itr))[1]
    QV, QO = A.std(zi, Q(cap.iva), Q(cap.va))[1:]
    KI, KV = A.std(ki, kv)
    fits = [_train_two_tables(QI, ci, yi, KI, QV, cv, yv, KV, 8, 0.1, s, H.MAX_EPOCHS) for s in SEEDS]
    meter.tick("C_bil")
    out["C_bil"] = (H.argmax_canonical(H.predict_proba(fits, QO, co, Ko)) == yo).float().numpy()
    return out


def main():
    H.set_threads(4)
    meter = H.BudgetMeter(1.0)
    out_dir = ROOT / "outputs" / "h1c_devx" / "final_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "FINAL_DIAGNOSTIC.json").exists():
        raise SystemExit("single run only")
    cap = X.CapX([(ROOT / "outputs" / "h1c_dev" / "capture", ROOT / "data" / "h1c" / "dev"),
                  (ROOT / "outputs" / "h1c_devx" / "capture", ROOT / "data" / "h1c" / "devx")], ROOT / "outputs" / "h1c_devx" / "plan" / "SPLITS.json")
    res = {"status": "POST HOC / EXPLORATORY", "layer": LAYER, "n": {"TRAIN": len(cap.itr), "SELECT": len(cap.iva), "EVAL": len(cap.va)}}
    okP = X.fit(lambda c, l, ids: A.binary_inputs(c, l, ids), cap, LAYER, SEEDS, H.MAX_EPOCHS)(cap.va)
    okC = X.fit(lambda c, l, ids: A.binary_inputs(c, l, ids, chosen=True), cap, LAYER, SEEDS, H.MAX_EPOCHS)(cap.va)
    meter.tick("pointers")
    ctl_u, ctl_c = controls(cap, False, meter), controls(cap, True, meter)
    L = cap.layer(LAYER)

    def ident(pairs, chosen):
        same = {}
        for f, ps in pairs.items():
            for p in ps:
                a, b = cap.rows[cap.va[p[0]]], cap.rows[cap.va[p[1]]]
                slots = a["bsc" if chosen else "bs"]
                same[p] = all(np.array_equal(L[a["world_id"]][s], L[b["world_id"]][s]) for s in slots)
        return same
    early = pair_index(cap, cap.va, ("base_E", "dswap_E"))
    late = pair_index(cap, cap.va, ("base_L", "dswap_L"))
    csw = pair_index(cap, cap.va, ("base_L", "cswap_L"))
    id_e, id_l, id_c = ident(early, False), ident(late, False), ident(csw, True)
    res["key_identity_within_pairs"] = {"early_unchosen": f"{sum(id_e.values())}/{len(id_e)}", "late_unchosen": f"{sum(id_l.values())}/{len(id_l)}",
                                        "cswap_chosen": f"{sum(id_c.values())}/{len(id_c)}"}
    # E1 early pairs (unchosen)
    e1 = {"P": robust(early, okP)}
    for n, okc in ctl_u.items():
        e1[n] = robust(early, okc)
    best = max(("key_only", "C_lin", "C_bil"), key=lambda n: e1[n].get("acc", 0))
    e1["best_control"] = best
    e1["P_minus_best_control"] = diff(early, okP, early, ctl_u[best])
    res["E1_early_unchosen"] = e1
    # E2 cswap pairs (chosen)
    keep_id = lambda p: id_c[p]
    e2 = {"all_pairs": {"PC2": robust(csw, okC), **{n: robust(csw, o) for n, o in ctl_c.items()}},
          "identical_key_subset": {"PC2": robust(csw, okC, keep_id), **{n: robust(csw, o, keep_id) for n, o in ctl_c.items()}}}
    res["E2_cswap_chosen"] = e2
    # E3 branch comparison (late-comparable): chosen on identical-key cswap pairs vs unchosen on late pairs
    res["E3_branch_comparison"] = {"chosen_cswap_identical": e2["identical_key_subset"]["PC2"], "unchosen_late": robust(late, okP),
                                   "chosen_minus_unchosen_per_family": diff(csw, okC, late, okP, keep_a=keep_id),
                                   "populations": {"late_pairs_contrib_families": len(late), "cswap_pairs_contrib_families": len(csw),
                                                   "cswap_identical_key_pairs": int(sum(id_c.values())),
                                                   "note": "different admissible subsets: a cswap pair needs base_L and cswap_L admissible; "
                                                           "a late pair needs base_L and dswap_L admissible; the chosen keys are identical only "
                                                           "when the chosen arrivals precede the chosen swapped lines (not enforced by the generator)"}}
    res["cpu_hours"] = meter.used_s() / 3600
    (out_dir / "FINAL_DIAGNOSTIC.json").write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
