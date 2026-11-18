"""CPU benchmark of the H1 training grid on SYNTHETIC arrays with the real shapes (no activations).

Shapes: d = 5,376; label table 5,800 x 5,376; train worlds 2,850, val 930 (≈ 90 % correct-choice of the
3,164 / 1,036 DEV non-decoy worlds); 6 candidates. Measures process CPU time (user+sys, all threads) and wall
time per epoch for r in {8, 32, 128} and thread counts {1, 4, 16}; extrapolates full-grid cost at the
worst case (60 epochs per fit) and at a typical early-stopped length. Writes outputs/metrics/bench_h1_grid.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h1_probe as H  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D, NL, NTR, NVA = 5376, 5800, 2850, 930
N_EPOCH_BENCH = 3
BENCH_R = (8, 32, 128)  # ranks benchmarked (v0.2 grid); v0.3 uses (8, 32)


def data(seed=0):
    g = torch.Generator().manual_seed(seed)
    T = torch.randn(NL, D, generator=g)
    mk = lambda n: (torch.randn(n, D, generator=g), torch.stack([torch.randperm(NL, generator=g)[:6] for _ in range(n)]),
                    torch.randint(0, 6, (n,), generator=g))
    return T, mk(NTR), mk(NVA)


def bench(threads, r, T, tr, va):
    H.set_threads(threads)
    c0, w0 = time.process_time(), time.time()
    f = H.train_bilinear(*tr, *va, T, r, 1e-3, 0, max_epochs=N_EPOCH_BENCH)
    # force N_EPOCH_BENCH epochs regardless of early stopping: use fixed_epochs path for timing
    c1, w1 = time.process_time(), time.time()
    f = H.train_bilinear(*tr, *va, T, r, 1e-3, 0, fixed_epochs=N_EPOCH_BENCH)
    c2, w2 = time.process_time(), time.time()
    # one validation pass cost (early stopping evaluates val every epoch)
    with torch.no_grad():
        H._scores(va[0], va[1], T, f.U, f.V)
    c3, w3 = time.process_time(), time.time()
    return {"cpu_s_per_epoch": (c2 - c1) / N_EPOCH_BENCH + (c3 - c2), "wall_s_per_epoch": (w2 - w1) / N_EPOCH_BENCH + (w3 - w2)}


def main():
    T, tr, va = data()
    res = {"shapes": {"d": D, "labels": NL, "train": NTR, "val": NVA}, "by_threads": {}}
    for th in (1, 4, 16):
        res["by_threads"][th] = {}
        for r in BENCH_R:
            res["by_threads"][th][r] = bench(th, r, T, tr, va)
            print(th, r, res["by_threads"][th][r], flush=True)
    # full-grid accounting (fits)
    n_layers = 62
    grid_layer = n_layers * len(BENCH_R) * len(H.GRID_WD) * len(H.SEEDS)   # per head with layer search
    grid_nolayer = len(BENCH_R) * len(H.GRID_WD) * len(H.SEEDS)
    fits = {"primary_u2 (layer x r x wd x seed)": grid_layer,
            "B6 anchors c1,c3,u1,u3 (each full grid)": 4 * grid_layer,
            "order heads u1,u3 at (l*, r*, wd*) x 3 seeds": 6,
            "B1 mean_emb (r x wd x seed)": grid_nolayer,
            "B2 3 permutations x 3 seeds at (l*, r*, wd*)": 9,
            "B5 candidate-only (r x wd x seed)": grid_nolayer,
            "final refits (primary 3 + anchors 12 + order 6 + B1 3 + B5 3)": 27}
    res["fits"] = fits
    # cost per fit: worst case 60 epochs; per-r share of grid is equal (1/3 each)
    out = {}
    for th in (1, 4, 16):
        per_r = res["by_threads"][th]
        mean_cpu_epoch = sum(per_r[r]["cpu_s_per_epoch"] for r in BENCH_R) / 3
        mean_wall_epoch = sum(per_r[r]["wall_s_per_epoch"] for r in BENCH_R) / 3
        for label, ep in (("worst_60_epochs", 60), ("typical_20_epochs", 20)):
            tot_fits = sum(fits.values())
            out[f"threads{th}_{label}"] = {
                "cpu_hours_all": tot_fits * ep * mean_cpu_epoch / 3600,
                "wall_hours_all": tot_fits * ep * mean_wall_epoch / 3600,
                "cpu_hours_without_B6_layer_search": (tot_fits - 4 * grid_layer) * ep * mean_cpu_epoch / 3600}
    res["extrapolation"] = out
    p = ROOT / "outputs" / "metrics" / "bench_h1_grid.json"
    p.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
