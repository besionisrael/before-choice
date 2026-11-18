"""H1 probe machinery (BTC-H1 freeze v0.3). CPU, torch. Exercised ONLY on synthetic arrays until the
owner authorises training on real activations.

Input layout (per world i):
  z[i]      (d,)  activation at ONE vector index
  cand[i]   (K=6,) indices into a frozen label-embedding table T, in CANONICAL order = candidate labels sorted
            by their label string (h1_features.canonical_candidates); depth-3 worlds have 4 candidates and are
            padded with -1 (masked). The order therefore carries no information about slots or targets.
  y[i]      index (in canonical order) of the target label for the head being trained.
Slots c1..c3 / u1..u3 (chosen / unchosen branch, defined from the MODEL'S forced choice) are a SUPERVISION
convention only: they are used to compute y, never as an input. All ties are broken by canonical index,
i.e. by label string, never by slot.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

SLOTS = ("c1", "c2", "c3", "u1", "u2", "u3")
LAYERS = tuple(range(0, 61, 4)) + (61,)  # 17 vector indices (v0.3 grid)
GRID_R = (8, 32)
GRID_WD = (1e-4, 1e-3, 1e-2, 1e-1)
SEEDS = (0, 1, 2)
LR, BATCH, MAX_EPOCHS, PATIENCE = 1e-3, 256, 60, 5
K = 6
PAD = -1


@dataclass
class Fit:
    U: torch.Tensor
    V: torch.Tensor
    best_epoch: int
    val_loss: float


def _scores(z, cand, T, U, V):
    mask = cand == PAD
    c = cand.clamp(min=0)
    uniq, inv = torch.unique(c, return_inverse=True)  # project only the labels present in this batch
    s = (z @ U).unsqueeze(1).mul((T[uniq] @ V)[inv]).sum(-1)
    return s.masked_fill(mask, float("-inf"))


def train_bilinear(z_tr, cand_tr, y_tr, z_va, cand_va, y_va, T, r, wd, seed, max_epochs=MAX_EPOCHS, fixed_epochs=None):
    """AdamW; early stopping on val loss (patience 5) unless fixed_epochs is given (final refit)."""
    g = torch.Generator().manual_seed(seed)
    d = z_tr.shape[1]
    U = (torch.randn(d, r, generator=g) / math.sqrt(d)).requires_grad_()
    V = (torch.randn(T.shape[1], r, generator=g) / math.sqrt(T.shape[1])).requires_grad_()
    opt = torch.optim.AdamW([U, V], lr=LR, weight_decay=wd)
    best, best_ep, bad, state = math.inf, 0, 0, (U.detach().clone(), V.detach().clone())
    n = z_tr.shape[0]
    epochs = fixed_epochs if fixed_epochs is not None else max_epochs
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            loss = torch.nn.functional.cross_entropy(_scores(z_tr[b], cand_tr[b], T, U, V), y_tr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if fixed_epochs is not None:
            continue
        with torch.no_grad():
            vl = float(torch.nn.functional.cross_entropy(_scores(z_va, cand_va, T, U, V), y_va))
        if vl < best - 1e-6:
            best, best_ep, bad, state = vl, ep, 0, (U.detach().clone(), V.detach().clone())
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if fixed_epochs is not None:
        return Fit(U.detach(), V.detach(), fixed_epochs, float("nan"))
    return Fit(state[0], state[1], best_ep, best)


def predict_proba(fits, z, cand, T):
    """Single deterministic predictor = average of the seeds' softmax probabilities (masked slots = 0)."""
    with torch.no_grad():
        return torch.stack([torch.softmax(_scores(z, cand, T, f.U, f.V), -1) for f in fits]).mean(0)


def argmax_canonical(prob: torch.Tensor) -> torch.Tensor:
    """argmax with ties broken by LOWEST canonical index (= label order), never by slot."""
    m = prob.max(-1, keepdim=True).values
    idx = torch.arange(prob.shape[-1]).expand_as(prob)
    return torch.where(prob == m, idx, torch.full_like(idx, 10 ** 6)).min(-1).values


def joint_assignment(logp: np.ndarray, k: int, n_valid: int | None = None) -> tuple[int, ...]:
    """Assign k heads to k DISTINCT candidates (canonical indices < n_valid) maximising the summed log-prob.
    Ties: lexicographically smallest tuple of canonical indices (label order; slot-independent)."""
    n = logp.shape[1] if n_valid is None else n_valid
    best, arg = -math.inf, None
    for a in itertools.permutations(range(n), k):  # lexicographic enumeration
        s = sum(logp[h, a[h]] for h in range(k))
        if s > best + 1e-12:
            best, arg = s, a
    return arg


def b6_expectation(anchor_logp: np.ndarray, target_idx: int, n_valid: int = K) -> float:
    """B6 per-world probability of a correct primary answer. anchor_logp: (4, n) log-probs of heads
    (c1, c3, u1, u3). The 4 anchors take 4 DISTINCT candidates (no duplicates possible); exactly n-4 = 2
    candidates remain (depth 4) and share probability uniformly. Returns 1/2 if the target remains, else 0."""
    a = joint_assignment(anchor_logp, 4, n_valid)
    remaining = [s for s in range(n_valid) if s not in a]
    assert len(remaining) == n_valid - 4 and remaining
    return 1.0 / len(remaining) if target_idx in remaining else 0.0


def order_correct(order_logp: np.ndarray, true_idx: tuple[int, int, int], n_valid: int = K) -> float:
    """Exact match of the jointly assigned ordered triple (u1, u2, u3) with the true canonical indices."""
    return float(joint_assignment(order_logp, 3, n_valid) == tuple(true_idx))


def paired_accuracy(p_world: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Paired success of (i, j) = p_i * p_j (0/1 decoders: both correct; B6: product of expectations)."""
    return np.array([p_world[i] * p_world[j] for i, j in pairs])


def b2_permutation(family_id: str, k: int) -> tuple[int, ...]:
    """Paired permutation of the 6 SLOTS for family f, index k in {0,1,2} (sha256-seeded)."""
    seed = int(hashlib.sha256(f"B2|{family_id}|{k}".encode()).hexdigest()[:16], 16)
    return tuple(int(x) for x in np.random.default_rng(seed).permutation(6))


def family_bootstrap(values_by_family, B=10_000, seed=0, stat=None):
    fams = sorted(values_by_family)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        pick = [fams[i] for i in rng.integers(0, len(fams), len(fams))]
        out.append(stat(pick) if stat else np.mean([v for f in pick for v in values_by_family[f]]))
    out = np.sort(np.array(out))
    return float(out[int(0.025 * B)]), float(out[int(0.975 * B) - 1])


def l2_order_predict(heads, head_layers, layer, z_layer, cand, T, true_idx) -> np.ndarray:
    """L2 order test. ALL three heads (u1, u2, u3) must have been fit on DEV at the SAME vector index `layer`
    (= l*); z_layer is that single layer's activations; cand in canonical order; true_idx (N, 3) canonical
    indices of the true (u1, u2, u3) labels (supervision only). No oracle answer is an input."""
    assert set(heads) == {"u1", "u2", "u3"}
    assert all(head_layers[h] == layer for h in heads), "order heads must share the H1 layer l*"
    logp = np.stack([torch.log(predict_proba(heads[h], z_layer, cand, T)).numpy() for h in ("u1", "u2", "u3")], 1)
    nv = (cand != PAD).sum(-1).numpy()
    return np.array([order_correct(logp[i], tuple(true_idx[i]), int(nv[i])) for i in range(logp.shape[0])])


class BudgetExceeded(RuntimeError):
    pass


class BudgetMeter:
    """GLOBAL CPU budget for the whole H1 programme (bilinear heads AND B3 / B3-order logistic fits).
    1 CPU-hour = 3600 s of process CPU time (user + system, all threads) of the single training process,
    as returned by time.process_time(). tick() after every fit; on overrun raises BudgetExceeded (the caller
    writes BUDGET_EXCEEDED with the completed fits and stops; no selection on a partial grid)."""

    def __init__(self, cpu_hours: float = 48.0, log_path=None):
        self.limit_s = cpu_hours * 3600
        self.t0 = time.process_time()
        self.wall0 = time.time()
        self.n_fits = 0
        self.log_path = log_path

    def used_s(self) -> float:
        return time.process_time() - self.t0

    def tick(self, label: str) -> None:
        self.n_fits += 1
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{self.n_fits}\t{label}\t{self.used_s():.1f}\t{time.time() - self.wall0:.1f}\n")
        if self.used_s() > self.limit_s:
            raise BudgetExceeded(f"CPU budget {self.limit_s / 3600:.1f} h exceeded after {self.n_fits} fits")


def set_threads(n: int = 4) -> None:
    torch.set_num_threads(n)
    os.environ["OMP_NUM_THREADS"] = str(n)


def train_conditional_logit(X: torch.Tensor, cand_mask: torch.Tensor, y: torch.Tensor, wd: float,
                            meter: "BudgetMeter | None" = None, label: str = "B3") -> torch.Tensor:
    """B3 / B3-order: conditional logit over candidates, score = X[i, k] . w ; L2 penalty wd; L-BFGS
    (deterministic, no seed). X: (N, K, F) prompt-text features (h1_features.b3_features), cand_mask True
    where a candidate is padding. Counted in the GLOBAL CPU budget via `meter`."""
    w = torch.zeros(X.shape[-1], requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1.0, max_iter=200, tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        s = (X @ w).masked_fill(cand_mask, float("-inf"))
        loss = torch.nn.functional.cross_entropy(s, y) + wd * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    if meter is not None:
        meter.tick(label)
    return w.detach()
