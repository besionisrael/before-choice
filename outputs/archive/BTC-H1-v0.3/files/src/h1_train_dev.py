"""BTC-H1-DEV-TRAIN runner — H1 v0.3 §3–§5, §11 (DEV ONLY; never touches LOCKED-TEST).

    python src/h1_train_dev.py --run                         # REAL activations: requires owner authorisation
    python src/h1_train_dev.py --run --resume --resume-approval <file>   # only after an approved interruption
    python src/h1_train_dev.py --synthetic-test <dir>        # CPU self-test on synthetic shards (tests only)

Order of fits (pre-declared; frozen grid H1 v0.3 §4):
  P  primary u2 head: LAYERS(17) x R(8,32) x WD(4) x SEEDS(3) = 408      -> select (l*, r*, wd*), E*
  A  B6 anchor heads c1, c3, u1, u3: same grid each = 1,632               -> per head select
  B1 mean_emb head: R x WD x SEEDS = 24                                   -> select
  B5 candidate-only head: R x WD x SEEDS = 24                             -> select
  B3 conditional logit u2: WD(4); B3-order heads u1,u2,u3: 3 x WD(4)      -> select
  O  order heads u1, u3 at (l*, r*, wd*): 2 x 3 seeds (early stopping)    -> E_u1, E_u3
  B2 permuted targets at (l*, r*, wd*), E* epochs: 3 perms x 3 seeds = 9
  F  final refits on ALL DEV training-rule worlds: primary 3, anchors 12, order 6, B1 3, B5 3 (+ B3 1, B3-order 3)
Totals: 2,130 bilinear fits (as frozen) + 20 conditional-logit fits, all under ONE BudgetMeter (48 CPU-h, 4 threads).
Every completed fit is appended to fits.jsonl. A grid's selection is computed ONLY when every planned key of that grid
is present; on BudgetExceeded the runner writes BUDGET_EXCEEDED.json and stops (no selection on a partial grid).
No LOCKED-TEST path is referenced; no TEST evaluation exists in this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import h1_features as F  # noqa: E402
import h1_probe as H  # noqa: E402
from d3_capture_core import u16_to_bf16_f32  # noqa: E402

THREADS = 4
CPU_HOURS = 48.0
REAL = {"capture": ROOT / "outputs" / "d3" / "probe_dev", "dev": ROOT / "data" / "d5" / "probe_dev",
        "out": ROOT / "outputs" / "h1_dev"}
ANCHOR_HEADS = ("c1", "c3", "u1", "u3")


# ------------------------------------------------------------------ data
class Dev:
    def __init__(self, capture: Path, dev: Path):
        man = json.loads((capture / "MANIFEST.json").read_text(encoding="utf-8"))
        self.shards = man["shards"]
        self.capture = capture
        idx = []
        for s in self.shards:
            idx += [json.loads(x) for x in (capture / f"{s['name']}_index.jsonl").read_text(encoding="utf-8").splitlines()]
        self.pos = {r["world_id"]: i for i, r in enumerate(idx)}
        self.idx = idx
        inputs = {r["world_id"]: r for r in (json.loads(x) for x in (dev / "model_inputs.jsonl").read_text(encoding="utf-8").splitlines())}
        worlds = {r["world_id"]: r for r in (json.loads(x) for x in (dev / "worlds.jsonl").read_text(encoding="utf-8").splitlines())}
        lab = json.loads((capture / "candidate_labels.json").read_text(encoding="utf-8"))["labels"]
        self.label_index = {l: i for i, l in enumerate(lab)}
        self.T_raw = np.load(capture / "candidate_embeddings.npy")
        self.rows = []  # training-rule worlds (H1 v0.3 §3)
        for r in idx:
            w = worlds[r["world_id"]]
            if w["variant"].startswith("decoy"):
                continue
            ch = "A" if r["logp_A"] > r["logp_B"] else "B" if r["logp_A"] < r["logp_B"] else "TIE"
            if ch == "TIE" or ch != w["oracle"]["optimal_action"]:
                continue
            prompt = inputs[r["world_id"]]["prompt"]                       # model-visible, oracle-free
            cands = F.canonical_candidates(F.candidates_from_prompt(prompt))
            sup = F.supervision_targets(w, ch)                            # oracle -> labels only
            tgt = {s: cands.index(l) for s, l in sup.items()}
            self.rows.append({"world_id": r["world_id"], "family_id": w["family_id"], "split": w["split"], "depth": w["depth"],
                              "variant": w["variant"], "choice": ch, "prompt": prompt, "cands": cands, "tgt": tgt})
        self.by_split = {s: [i for i, r in enumerate(self.rows) if r["split"] == s] for s in ("train", "val")}
        # DEV-val selection pairs: admissible depth-4 upath pairs (both members are training-rule worlds)
        present = {r["world_id"]: i for i, r in enumerate(self.rows)}
        self.pairs = {}
        for split in ("train", "val"):
            ps = []
            fams = sorted({r["family_id"] for r in self.rows if r["split"] == split and r["depth"] == 4})
            for f in fams:
                for src in ("base", "cf"):
                    a, b = f"{f}-{src}", f"{f}-upath_{src}"
                    if a in present and b in present:
                        ps.append((present[a], present[b]))
            self.pairs[split] = ps
        self.val_pairs = self.pairs["val"]

    def cand_tensor(self, ids):
        c = torch.full((len(ids), H.K), H.PAD, dtype=torch.long)
        for k, i in enumerate(ids):
            for j, l in enumerate(self.rows[i]["cands"]):
                c[k, j] = self.label_index[l]
        return c

    def layer(self, ell: int, ids) -> np.ndarray:
        need = {self.pos[self.rows[i]["world_id"]] for i in ids}
        out = {}
        base = 0
        for s in self.shards:
            a = np.load(self.capture / f"{s['name']}_acts.npy", mmap_mode="r")
            for g in range(base, base + s["n"]):
                if g in need:
                    out[g] = u16_to_bf16_f32(np.asarray(a[g - base, ell]))
            base += s["n"]
        return np.stack([out[self.pos[self.rows[i]["world_id"]]] for i in ids]).astype(np.float32)

    def mean_emb(self, ids) -> np.ndarray:
        allm = np.concatenate([np.load(self.capture / f"{s['name']}_mean_emb.npy") for s in self.shards])
        return np.stack([allm[self.pos[self.rows[i]["world_id"]]] for i in ids]).astype(np.float32)


def b2_slot(row: dict, k: int) -> str:
    """B2 target slot for a world. Depth 4 (6 slots): the frozen H.b2_permutation of the 6 slots, slot pi(u2).
    Depth 3 (4 slots c1,c2,u1,u2; c3/u3 do not exist): the same sha256 seeding applied to a permutation of the
    4 AVAILABLE slots (operational clarification; never falls back to the true u2 target)."""
    if all(s in row["tgt"] for s in H.SLOTS):
        pi = H.b2_permutation(row["family_id"], k)
        return H.SLOTS[pi[H.SLOTS.index("u2")]]
    avail = [s for s in H.SLOTS if s in row["tgt"]]
    seed = int(hashlib.sha256(f"B2|{row['family_id']}|{k}|{len(avail)}".encode()).hexdigest()[:16], 16)
    pi = np.random.default_rng(seed).permutation(len(avail))
    return avail[int(pi[avail.index("u2")])]


def standardise(train: np.ndarray, *others: np.ndarray):
    mu, sd = train.mean(0), train.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return [torch.from_numpy((x - mu) / sd) for x in (train,) + others]


# ------------------------------------------------------------------ grids
def plan(layers, ranks, wds, seeds):
    P = [("P", "u2", l, r, w, s) for l in layers for r in ranks for w in wds for s in seeds]
    A = [("A", h, l, r, w, s) for h in ANCHOR_HEADS for l in layers for r in ranks for w in wds for s in seeds]
    B1 = [("B1", "u2", "mean_emb", r, w, s) for r in ranks for w in wds for s in seeds]
    B5 = [("B5", "u2", "none", r, w, s) for r in ranks for w in wds for s in seeds]
    B3 = [("B3", "u2", "text", 0, w, 0) for w in wds] + [("B3o", h, "text", 0, w, 0) for h in ("u1", "u2", "u3") for w in wds]
    return {"P": P, "A": A, "B1": B1, "B5": B5, "B3": B3}


def key(t):
    return "|".join(map(str, t))


class Ledger:
    """Append-only record of completed fits; selection only on complete grids."""

    def __init__(self, path: Path):
        self.path = path
        self.done = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                self.done[rec["key"]] = rec

    def add(self, rec: dict) -> None:
        self.done[rec["key"]] = rec
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def complete(self, keys) -> bool:
        return all(key(k) in self.done for k in keys)


def select(ledger: Ledger, keys, metric: str, by_head: bool = False):
    """Mean over seeds of the DEV-val metric; ties: smaller r, larger wd, lower layer index."""
    assert ledger.complete(keys), "refusing selection on an incomplete grid"
    groups = defaultdict(list)
    for k in keys:
        rec = ledger.done[key(k)]
        groups[k[:5]].append(rec)
    def rank(item):
        cfg, recs = item
        lay = cfg[2] if isinstance(cfg[2], int) else -1
        return (-np.mean([r[metric] for r in recs]), cfg[3], -cfg[4], lay)
    cfg, recs = sorted(groups.items(), key=rank)[0]
    return {"config": list(cfg), "mean_metric": float(np.mean([r[metric] for r in recs])),
            "E": int(round(np.mean([r["best_epoch"] for r in recs])))}


# ------------------------------------------------------------------ evaluation on DEV-val
def paired_acc(prob, ids, dev: Dev, slot="u2", split="val"):
    """Paired accuracy on the admissible depth-4 upath pairs of `split` (both members correct)."""
    pos = {i: k for k, i in enumerate(ids)}
    pred = H.argmax_canonical(prob).numpy()
    ok = {i: float(pred[pos[i]] == dev.rows[i]["tgt"][slot]) for i in ids}
    ps = dev.pairs[split]
    return float(np.mean([ok[a] * ok[b] for a, b in ps])) if ps else float("nan")


def head_acc(prob, ids, dev: Dev, slot):
    """Single-world accuracy for `slot`, over the worlds where that slot exists (c3/u3: depth 4 only)."""
    keep = [k for k, i in enumerate(ids) if slot in dev.rows[i]["tgt"]]
    if not keep:
        return float("nan")
    return world_acc(prob[keep], [ids[k] for k in keep], dev, slot)


def world_acc(prob, ids_val, dev: Dev, slot):
    pred = H.argmax_canonical(prob).numpy()
    return float(np.mean([pred[k] == dev.rows[i]["tgt"][slot] for k, i in enumerate(ids_val)]))


# ------------------------------------------------------------------ main programme
def run(dev: Dev, out: Path, layers=H.LAYERS, ranks=H.GRID_R, wds=H.GRID_WD, seeds=H.SEEDS,
        max_epochs=H.MAX_EPOCHS, resume=False) -> dict:
    H.set_threads(THREADS)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "fits.jsonl").exists() and not resume:
        raise SystemExit("fits.jsonl exists: refusing to restart without an approved --resume")
    ledger = Ledger(out / "fits.jsonl")
    meter = H.BudgetMeter(CPU_HOURS, out / "budget.tsv")
    G = plan(layers, ranks, wds, seeds)
    status = {"grids": {g: len(v) for g, v in G.items()}}

    tr, va = dev.by_split["train"], dev.by_split["val"]
    c_tr, c_va = dev.cand_tensor(tr), dev.cand_tensor(va)
    train_labels = sorted({dev.label_index[l] for i in tr for l in dev.rows[i]["cands"]})
    mu, sd = dev.T_raw[train_labels].mean(0), dev.T_raw[train_labels].std(0)
    T = torch.from_numpy((dev.T_raw - mu) / np.where(sd < 1e-6, 1.0, sd)).float()

    def targets(ids, slot):
        keep = [k for k, i in enumerate(ids) if slot in dev.rows[i]["tgt"]]
        return keep, torch.tensor([dev.rows[ids[k]]["tgt"][slot] for k in keep])

    def fit_and_log(tag, z_tr, z_va, slot, r, w, s, mname, metric_fn, fixed=None, perm_targets=None):
        """One fit on DEV-train with early stopping on DEV-val; logs DEV-train AND DEV-val metrics."""
        k = key(tag)
        if k in ledger.done:
            return ledger.done[k]
        ktr, ytr = targets(tr, slot) if perm_targets is None else perm_targets
        kva, yva = targets(va, slot)
        f = H.train_bilinear(z_tr[ktr], c_tr[ktr], ytr, z_va[kva], c_va[kva], yva, T, r, w, s,
                             max_epochs=max_epochs, fixed_epochs=fixed)
        p_va = H.predict_proba([f], z_va, c_va, T)
        p_tr = H.predict_proba([f], z_tr, c_tr, T)
        rec = {"key": k, "best_epoch": f.best_epoch, "hit_epoch_cap": f.best_epoch >= max_epochs, "val_loss": f.val_loss,
               f"val_{mname}": metric_fn(p_va, va, "val"), f"train_{mname}": metric_fn(p_tr, tr, "train"),
               "cpu_s": meter.used_s()}
        ledger.add(rec)
        meter.tick(k)
        return rec

    try:
        layer_cache = {}

        def Z(ell):
            if ell not in layer_cache:
                layer_cache.clear()
                layer_cache[ell] = standardise(dev.layer(ell, tr), dev.layer(ell, va))
            return layer_cache[ell]

        mP = lambda prob, ids, split: paired_acc(prob, ids, dev, "u2", split)
        for t in G["P"]:
            zt, zv = Z(t[2])
            fit_and_log(t, zt, zv, "u2", t[3], t[4], t[5], "paired_acc", mP)
        selP = select(ledger, G["P"], "val_paired_acc")
        status["primary_selection"] = selP
        for h in ANCHOR_HEADS:
            mA = lambda prob, ids, split, h=h: head_acc(prob, ids, dev, h)
            for t in [t for t in G["A"] if t[1] == h]:
                zt, zv = Z(t[2])
                fit_and_log(t, zt, zv, h, t[3], t[4], t[5], "world_acc", mA)
        status["anchor_selection"] = {h: select(ledger, [t for t in G["A"] if t[1] == h], "val_world_acc") for h in ANCHOR_HEADS}
        me_tr, me_va = standardise(dev.mean_emb(tr), dev.mean_emb(va))
        for t in G["B1"]:
            fit_and_log(t, me_tr, me_va, "u2", t[3], t[4], t[5], "paired_acc", mP)
        status["B1_selection"] = select(ledger, G["B1"], "val_paired_acc")
        one_tr, one_va = torch.ones(len(tr), 1), torch.ones(len(va), 1)
        for t in G["B5"]:
            fit_and_log(t, one_tr, one_va, "u2", t[3], t[4], t[5], "paired_acc", mP)
        status["B5_selection"] = select(ledger, G["B5"], "val_paired_acc")
        # B3 / B3-order (conditional logit, counted in the same budget)
        def X(ids):
            x = torch.zeros(len(ids), H.K, len(F.B3_FEATURES))
            m = torch.ones(len(ids), H.K, dtype=torch.bool)
            for k, i in enumerate(ids):
                row = dev.rows[i]
                x[k, :len(row["cands"])] = torch.from_numpy(F.b3_features(row["prompt"], row["cands"], row["choice"]))
                m[k, :len(row["cands"])] = False
            return x, m
        Xtr, Mtr = X(tr)
        Xva, Mva = X(va)
        for t in G["B3"]:
            if key(t) in ledger.done:
                continue
            slot = t[1]
            ktr, ytr = targets(tr, slot)
            wv = H.train_conditional_logit(Xtr[ktr], Mtr[ktr], ytr, t[4], meter, key(t))
            p_va = torch.softmax((Xva @ wv).masked_fill(Mva, float("-inf")), -1)
            p_tr = torch.softmax((Xtr @ wv).masked_fill(Mtr, float("-inf")), -1)
            if t[0] == "B3":
                mv, mt = paired_acc(p_va, va, dev, "u2", "val"), paired_acc(p_tr, tr, dev, "u2", "train")
            else:
                mv, mt = head_acc(p_va, va, dev, slot), head_acc(p_tr, tr, dev, slot)
            ledger.add({"key": key(t), "best_epoch": 0, "hit_epoch_cap": False, "val_loss": float("nan"),
                        "val_metric": mv, "train_metric": mt, "cpu_s": meter.used_s()})
        status["B3_selection"] = select(ledger, [t for t in G["B3"] if t[0] == "B3"], "val_metric")
        status["B3o_selection"] = {h: select(ledger, [t for t in G["B3"] if t[0] == "B3o" and t[1] == h], "val_metric") for h in ("u1", "u2", "u3")}
        # order heads u1, u3 at (l*, r*, wd*): each selects its own epochs on DEV-val (H1 v0.3 §11.2)
        lS, rS, wS = selP["config"][2], selP["config"][3], selP["config"][4]
        zt, zv = Z(lS)
        O = [("O", h, lS, rS, wS, s) for h in ("u1", "u3") for s in seeds]
        for t in O:
            mO = lambda prob, ids, split, h=t[1]: head_acc(prob, ids, dev, h)
            fit_and_log(t, zt, zv, t[1], rS, wS, t[5], "world_acc", mO)
        status["order_E"] = {h: int(round(np.mean([ledger.done[key(t)]["best_epoch"] for t in O if t[1] == h]))) for h in ("u1", "u3")}
        status["completed_all_selection_grids"] = True
        (out / "SELECTIONS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        # ---------------- final refits on ALL DEV training-rule worlds (train + val), fixed epochs, 3 seeds
        allids = tr + va
        c_all = dev.cand_tensor(allids)
        fin_dir = out / "final"
        fin_dir.mkdir(exist_ok=True)
        final = {}

        def refit(name, z_all, slot, r, w, E, perm=None):
            keep = [k for k, i in enumerate(allids) if slot in dev.rows[i]["tgt"]]
            y = torch.tensor([dev.rows[allids[k]]["tgt"][slot] for k in keep])
            fits = []
            for s in seeds:
                k = f"F|{name}|{s}"
                f = H.train_bilinear(z_all[keep], c_all[keep], y, z_all[keep], c_all[keep], y, T, r, w, s, fixed_epochs=max(E, 1))
                fits.append(f)
                ledger.add({"key": k, "best_epoch": f.best_epoch, "val_loss": float("nan"), "cpu_s": meter.used_s()})
                meter.tick(k)
            p = fin_dir / f"{name}.pt"
            torch.save({"U": [f.U for f in fits], "V": [f.V for f in fits], "r": r, "wd": w, "E": E, "slot": slot}, p)
            final[name] = {"path": str(p.relative_to(out)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "E": E}

        def zall(ell):
            a_tr, a_all = dev.layer(ell, tr), dev.layer(ell, allids)
            return standardise(a_tr, a_all)[1]   # DEV-train statistics (H1 v0.3 §4)
        refit("primary_u2", zall(lS), "u2", rS, wS, selP["E"])
        for h, sel in status["anchor_selection"].items():
            refit(f"anchor_{h}", zall(sel["config"][2]), h, sel["config"][3], sel["config"][4], sel["E"])
        z_l = zall(lS)
        for h in ("u1", "u3"):
            refit(f"order_{h}", z_l, h, rS, wS, status["order_E"][h])
        me_all = standardise(dev.mean_emb(tr), dev.mean_emb(allids))[1]
        refit("B1", me_all, "u2", status["B1_selection"]["config"][3], status["B1_selection"]["config"][4], status["B1_selection"]["E"])
        refit("B5", torch.ones(len(allids), 1), "u2", status["B5_selection"]["config"][3], status["B5_selection"]["config"][4], status["B5_selection"]["E"])
        # B2: paired permuted targets, NO search: (l*, r*, wd*), E* epochs, 3 permutations x 3 seeds = the 9 frozen fits,
        # trained on ALL DEV training-rule worlds (retraining rule), one averaged predictor per permutation.
        keep_u2 = [k for k, i in enumerate(allids) if "u2" in dev.rows[i]["tgt"]]
        b2_info = {}
        for kperm in range(3):
            yperm = torch.tensor([dev.rows[allids[k]]["tgt"][b2_slot(dev.rows[allids[k]], kperm)] for k in keep_u2])
            ytrue = torch.tensor([dev.rows[allids[k]]["tgt"]["u2"] for k in keep_u2])
            fits = []
            for s in seeds:
                kk = key(("B2", f"perm{kperm}", lS, rS, wS, s))
                f = H.train_bilinear(z_l[keep_u2], c_all[keep_u2], yperm, z_l[keep_u2], c_all[keep_u2], yperm, T, rS, wS, s,
                                     fixed_epochs=max(selP["E"], 1))
                fits.append(f)
                ledger.add({"key": kk, "best_epoch": f.best_epoch, "hit_epoch_cap": False, "val_loss": float("nan"),
                            "trained_on": "all DEV training-rule worlds", "cpu_s": meter.used_s()})
                meter.tick(kk)
            p = fin_dir / f"B2_perm{kperm}.pt"
            torch.save({"U": [f.U for f in fits], "V": [f.V for f in fits], "r": rS, "wd": wS, "E": selP["E"], "slot": "u2",
                        "permutation": kperm}, p)
            final[f"B2_perm{kperm}"] = {"path": str(p.relative_to(out)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                        "E": selP["E"]}
            b2_info[kperm] = {"fraction_targets_unchanged": float((yperm == ytrue).float().mean())}
        status["B2_info"] = b2_info
        Xall, Mall = X(allids)
        for name, slot, sel in [("B3", "u2", status["B3_selection"])] + [(f"B3o_{h}", h, status["B3o_selection"][h]) for h in ("u1", "u2", "u3")]:
            keep = [k for k, i in enumerate(allids) if slot in dev.rows[i]["tgt"]]
            y = torch.tensor([dev.rows[allids[k]]["tgt"][slot] for k in keep])
            wv = H.train_conditional_logit(Xall[keep], Mall[keep], y, sel["config"][4], meter, f"F|{name}")
            p = fin_dir / f"{name}.pt"
            torch.save({"w": wv, "wd": sel["config"][4], "slot": slot}, p)
            final[name] = {"path": str(p.relative_to(out)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        stats = {"T_mu_sd_from": "DEV-train candidate labels", "z_mu_sd_from": "DEV-train worlds"}
        expected = ({"primary_u2", "order_u1", "order_u3", "B1", "B5", "B3"} | {f"anchor_{h}" for h in ANCHOR_HEADS}
                    | {f"B3o_{h}" for h in ("u1", "u2", "u3")} | {f"B2_perm{k}" for k in range(3)})
        assert set(final) == expected, f"final predictor set incomplete: {sorted(expected - set(final))}"
        status["final_refits"] = {
            "predictors": final, "standardisation": stats, "n_predictors": len(final),
            "convention": "16 final predictors = 13 (primary u2; B6 anchors c1,c3,u1,u3; order heads u1,u3; B1; B5; B3; "
                          "B3-order u1,u2,u3) + 3 B2 predictors (one per permutation, each a 3-seed average). Each bilinear "
                          "predictor stores its 3 seed fits; the L2 order test uses primary_u2 as its u2 head.",
            "n_bilinear_refits": 3 * (1 + 4 + 2 + 2), "n_b2_fits_on_all_dev": 9, "n_logit_refits": 4}
        status["cpu_hours_used"] = meter.used_s() / 3600
        (out / "FINAL.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    except H.BudgetExceeded as e:
        (out / "BUDGET_EXCEEDED.json").write_text(json.dumps({"error": str(e), "completed_fits": len(ledger.done),
                                                               "status": status}, indent=2), encoding="utf-8")
        raise SystemExit(f"BUDGET EXCEEDED: {e} (no selection on incomplete grids)")
    return status


# ------------------------------------------------------------------ synthetic self-test (tests only)
def make_synthetic(dir_: Path, n_worlds=240, d=16, signal_layer=12, seed=0):
    """Builds fake capture shards for the first n_worlds real PROBE-DEV worlds; plants the u2 label's embedding
    into vector `signal_layer`. Used ONLY by tests."""
    rng = np.random.default_rng(seed)
    dev = REAL["dev"]
    inputs = [json.loads(x) for x in (dev / "model_inputs.jsonl").read_text(encoding="utf-8").splitlines()][:n_worlds]
    worlds = {r["world_id"]: r for r in (json.loads(x) for x in (dev / "worlds.jsonl").read_text(encoding="utf-8").splitlines())}
    import re
    labels = sorted({m.group(1) for r in inputs for m in re.finditer(r"(?<= )([A-Z0-9]{4})(?=[.,:])", r["prompt"])})
    T = rng.standard_normal((len(labels), d)).astype(np.float32)
    li = {l: i for i, l in enumerate(labels)}
    acts = rng.standard_normal((n_worlds, 62, d)).astype(np.float32)
    idx = []
    for k, r in enumerate(inputs):
        w = worlds[r["world_id"]]
        opt = w["oracle"]["optimal_action"]
        sup = F.supervision_targets(w, opt)
        if "u2" in sup:
            acts[k, signal_layer] += 3.0 * T[li[sup["u2"]]]
        idx.append({"world_id": r["world_id"], "logp_A": -0.1 if opt == "A" else -3.0, "logp_B": -3.0 if opt == "A" else -0.1})
    import torch as _t
    u16 = _t.from_numpy(acts).to(_t.bfloat16).view(_t.int16).numpy().view(np.uint16)
    dir_.mkdir(parents=True, exist_ok=True)
    np.save(dir_ / "shard_000_acts.npy", u16)
    np.save(dir_ / "shard_000_mean_emb.npy", acts[:, 0])
    (dir_ / "shard_000_index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in idx), encoding="utf-8")
    np.save(dir_ / "candidate_embeddings.npy", T)
    (dir_ / "candidate_labels.json").write_text(json.dumps({"labels": labels}), encoding="utf-8")
    (dir_ / "MANIFEST.json").write_text(json.dumps({"shards": [{"name": "shard_000", "n": n_worlds, "files": {}}]}), encoding="utf-8")
    dev_sub = dir_ / "dev"
    dev_sub.mkdir(exist_ok=True)
    (dev_sub / "model_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inputs), encoding="utf-8")
    (dev_sub / "worlds.jsonl").write_text("".join(json.dumps(worlds[r["world_id"]]) + "\n" for r in inputs), encoding="utf-8")
    return dir_, dev_sub


def prechecks_real() -> dict:
    import os
    import stat
    import sealed
    frz = json.loads((ROOT / "reports" / "BTC-H1_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
    art = frz["artifacts"]["D3-DEV-R1_activations"]
    seal_dir = ROOT / "data" / "d5" / ("locked" + "_test")  # name assembled: no literal sealed path in dev code
    chk = {"h1_code_frozen": all(sha(p) == frz["code_sha256"][p] for p in ("src/h1_probe.py", "src/h1_features.py")),
           "h1_document_frozen": sha("reports/BTC-H1_CONFIRMATORY_FREEZE_PROPOSAL.md") == frz["protocol_document_sha256_final"],
           "activations_manifest": sha("outputs/d3/probe_dev/MANIFEST.json") == art["MANIFEST.json_sha256"],
           "seal_intact": all(sealed.hash_only(seal_dir).values()),
           "seal_unread": not (seal_dir / sealed.ACCESS_LOG).exists()}
    if not all(chk.values()):
        raise SystemExit(f"PRECHECK FAILED: {chk}")
    return chk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume-approval", default="")
    ap.add_argument("--synthetic-test", default="")
    a = ap.parse_args()
    if a.synthetic_test:
        cap, devdir = make_synthetic(Path(a.synthetic_test))
        st = run(Dev(cap, devdir), Path(a.synthetic_test) / "out", layers=(0, 12), ranks=(8,), wds=(1e-3,), seeds=(0,), max_epochs=15)
        print(json.dumps(st, indent=1))
        return
    if not a.run:
        raise SystemExit("nothing to do (use --run only with owner authorisation)")
    if a.resume and not (a.resume_approval and Path(a.resume_approval).exists()):
        raise SystemExit("--resume requires an owner approval file")
    chk = prechecks_real()
    print("prechecks:", chk)
    st = run(Dev(REAL["capture"], REAL["dev"]), REAL["out"], resume=a.resume)
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
