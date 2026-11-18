"""BTC-D3-TEST preparation, item 1: verify the 16 frozen predictors (CPU, DEV data only; never touches the sealed pool).

    python src/h1_verify_final.py            -> outputs/h1_dev/FINAL_VERIFICATION.json (+ final/STANDARDISATION.npz, once)

Checks
  1. sha256 of every predictor == FINAL.json == freeze manifest artifacts.H1-DEV-TRAIN; FINAL/SELECTIONS hashes;
  2. structure vs H1 v0.3: 16 names; ranks, weight decays, epochs, input vector index per predictor equal the
     frozen selections; U/V shapes; 3 seed fits per bilinear predictor; B2 permutation ids 0..2;
  3. bitwise REPRODUCTION (read-only): the primary u2 refit (3 seeds) and the B3 logit refit are retrained from the
     frozen configuration on the frozen DEV activations and compared with the saved weights (nothing is overwritten);
  4. freezes the DEV-train standardisation statistics into a hashed file (written once) and checks they reproduce
     the transform used by the refits;
  5. pipeline sanity: in-sample DEV accuracy of the loaded primary predictor (IN-SAMPLE, not a result).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import h1_features as F  # noqa: E402
import h1_frozen as Z  # noqa: E402
import h1_probe as H  # noqa: E402
import h1_train_dev as T  # noqa: E402

OUT = ROOT / "outputs" / "h1_dev"


def main() -> None:
    H.set_threads(T.THREADS)
    frz = json.loads((ROOT / "reports" / "BTC-H1_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    art = frz["artifacts"]["H1-DEV-TRAIN"]
    rep = {"hashes": {}, "structure": {}, "reproduction": {}, "standardisation": {}, "pipeline": {}}
    rep["hashes"]["FINAL.json"] = Z.sha(OUT / "FINAL.json") == art["FINAL.json_sha256"]
    rep["hashes"]["SELECTIONS.json"] = Z.sha(OUT / "SELECTIONS.json") == art["SELECTIONS.json_sha256"]
    rep["hashes"]["h1_code_frozen"] = all(Z.sha(ROOT / p) == frz["code_sha256"][p] for p in ("src/h1_probe.py", "src/h1_features.py"))
    rep["hashes"]["runner"] = Z.sha(ROOT / "src" / "h1_train_dev.py") == art["runner_sha256"]
    L = Z.load_final(OUT, art["predictors"])           # raises on any mismatch
    rep["hashes"]["16_predictors_vs_FINAL_and_manifest"] = True
    P, fin = L["predictors"], L["final"]
    sel = json.loads((OUT / "SELECTIONS.json").read_text(encoding="utf-8"))
    lS, rS, wS, ES = (sel["primary_selection"]["config"][2], sel["primary_selection"]["config"][3],
                      sel["primary_selection"]["config"][4], sel["primary_selection"]["E"])
    d = 5376
    want = {"primary_u2": (lS, rS, wS, ES, d), "order_u1": (lS, rS, wS, sel["order_E"]["u1"], d),
            "order_u3": (lS, rS, wS, sel["order_E"]["u3"], d),
            "B1": ("mean_emb", sel["B1_selection"]["config"][3], sel["B1_selection"]["config"][4], sel["B1_selection"]["E"], d),
            "B5": ("none", sel["B5_selection"]["config"][3], sel["B5_selection"]["config"][4], sel["B5_selection"]["E"], 1)}
    for h in Z.ANCHORS:
        c = sel["anchor_selection"][h]
        want[f"anchor_{h}"] = (c["config"][2], c["config"][3], c["config"][4], c["E"], d)
    for k in range(3):
        want[f"B2_perm{k}"] = (lS, rS, wS, ES, d)
    st = {}
    for name, (inp, r, w, E, din) in want.items():
        p = P[name]
        ok = (p["input"] == inp and p["r"] == r and p["wd"] == w and p["E"] == E and len(p["fits"]) == 3
              and all(tuple(f.U.shape) == (din, r) and tuple(f.V.shape) == (d, r) for f in p["fits"]))
        if name.startswith("B2_perm"):
            ok = ok and p["permutation"] == int(name[-1]) and p["slot"] == "u2"
        st[name] = {"ok": bool(ok), "input": inp, "r": r, "wd": w, "E": E}
    for name, slot in [("B3", "u2"), ("B3o_u1", "u1"), ("B3o_u2", "u2"), ("B3o_u3", "u3")]:
        p = P[name]
        selw = sel["B3_selection"] if name == "B3" else sel["B3o_selection"][slot]
        st[name] = {"ok": bool(p["kind"] == "logit" and p["slot"] == slot and p["wd"] == selw["config"][4]
                                and tuple(p["w"].shape) == (len(F.B3_FEATURES),)), "wd": p["wd"]}
    rep["structure"] = {"per_predictor": st, "all_ok": all(v["ok"] for v in st.values()), "n": len(P)}

    # ---- reproduction (read-only) and frozen standardisation
    dev = T.Dev(T.REAL["capture"], T.REAL["dev"])
    tr, va = dev.by_split["train"], dev.by_split["val"]
    allids = tr + va
    std_path = OUT / "final" / Z.STD_FILE
    if not std_path.exists():
        rep["standardisation"]["written_now"] = True
        rep["standardisation"]["sha256"] = Z.freeze_standardisation(dev, fin, std_path)
    else:
        rep["standardisation"]["written_now"] = False
        rep["standardisation"]["sha256"] = Z.sha(std_path)
    S = Z.load_standardisation(std_path, rep["standardisation"]["sha256"])
    a_tr, a_all = dev.layer(lS, tr), dev.layer(lS, allids)
    z_runner = T.standardise(a_tr, a_all)[1]
    z_frozen = Z.standardise_with(a_all, S[f"z{lS}_mu"], S[f"z{lS}_sd"])
    rep["standardisation"]["z_identical_to_runner_transform"] = bool(torch.equal(z_runner, z_frozen))
    train_labels = sorted({dev.label_index[l] for i in tr for l in dev.rows[i]["cands"]})
    mu, sd = dev.T_raw[train_labels].mean(0), dev.T_raw[train_labels].std(0)
    Tm = torch.from_numpy((dev.T_raw - mu) / np.where(sd < 1e-6, 1.0, sd)).float()
    Tf = Z.standardise_with(dev.T_raw, S["T_mu"], S["T_sd"])
    rep["standardisation"]["T_identical_to_runner_transform"] = bool(torch.equal(Tm, Tf))
    c_all = dev.cand_tensor(allids)
    keep = [k for k, i in enumerate(allids) if "u2" in dev.rows[i]["tgt"]]
    y = torch.tensor([dev.rows[allids[k]]["tgt"]["u2"] for k in keep])
    same = []
    for s, f_saved in zip(H.SEEDS, P["primary_u2"]["fits"]):
        f = H.train_bilinear(z_runner[keep], c_all[keep], y, z_runner[keep], c_all[keep], y, Tm, rS, wS, s, fixed_epochs=max(ES, 1))
        same.append({"seed": s, "U_equal": bool(torch.equal(f.U, f_saved.U)), "V_equal": bool(torch.equal(f.V, f_saved.V)),
                     "max_abs_diff": float(max((f.U - f_saved.U).abs().max(), (f.V - f_saved.V).abs().max()))})
    rep["reproduction"]["primary_u2"] = same
    Xall = torch.zeros(len(allids), H.K, len(F.B3_FEATURES))
    Mall = torch.ones(len(allids), H.K, dtype=torch.bool)
    for k, i in enumerate(allids):
        row = dev.rows[i]
        Xall[k, :len(row["cands"])] = torch.from_numpy(F.b3_features(row["prompt"], row["cands"], row["choice"]))
        Mall[k, :len(row["cands"])] = False
    wv = H.train_conditional_logit(Xall[keep], Mall[keep], y, P["B3"]["wd"])
    rep["reproduction"]["B3"] = {"w_equal": bool(torch.equal(wv, P["B3"]["w"])), "max_abs_diff": float((wv - P["B3"]["w"]).abs().max())}

    # ---- pipeline sanity (IN-SAMPLE on DEV; not an evaluation)
    prob = Z.apply_bilinear(P["primary_u2"], z_frozen, c_all, Tf)
    pred = H.argmax_canonical(prob).numpy()
    acc = float(np.mean([pred[k] == dev.rows[allids[k]]["tgt"]["u2"] for k in keep]))
    rep["pipeline"] = {"primary_u2_in_sample_DEV_world_acc": acc, "n_worlds": len(keep),
                       "note": "in-sample (the final refit saw these worlds); checks the loading/scoring path only"}
    rep["all_pass"] = bool(all(rep["hashes"].values()) and rep["structure"]["all_ok"]
                           and rep["standardisation"]["z_identical_to_runner_transform"]
                           and rep["standardisation"]["T_identical_to_runner_transform"]
                           and all(x["U_equal"] and x["V_equal"] for x in same) and rep["reproduction"]["B3"]["w_equal"])
    (OUT / "FINAL_VERIFICATION.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
