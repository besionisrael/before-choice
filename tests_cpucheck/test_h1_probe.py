"""H1 machinery tests on SYNTHETIC arrays + oracle-free feature checks on PROBE-DEV prompts.
Run with:  .venv-cpucheck\\Scripts\\python -m unittest discover -s tests_cpucheck   (no activations involved)"""

import inspect
import json
import pathlib
import sys
import unittest

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import h1_features as F  # noqa: E402
import h1_probe as H  # noqa: E402

H.set_threads(4)
TARGET_SLOT_FOR_SYNTH = 4


def synth(n, d, n_labels, signal, seed, k=6):
    """Activation carries (signal x) the embedding of one designated candidate (the 'target');
    candidates are presented in an arbitrary order; y = position of the target in that order."""
    g = torch.Generator().manual_seed(seed)
    T = torch.randn(n_labels, d, generator=g)
    cand = torch.stack([torch.randperm(n_labels, generator=g)[:k] for _ in range(n)])
    tgt = cand[:, TARGET_SLOT_FOR_SYNTH % k]
    z = torch.randn(n, d, generator=g) + signal * T[tgt]
    perm = torch.stack([torch.randperm(k, generator=g) for _ in range(n)])
    cand = torch.gather(cand, 1, perm)
    y = (cand == tgt.unsqueeze(1)).float().argmax(1)
    return z, cand, y, T


class Trainer(unittest.TestCase):
    def test_signal_and_chance(self):
        for signal, lo, hi in ((1.0, 0.50, 1.01), (0.0, 0.05, 0.30)):  # chance = 1/6
            z, c, y, T = synth(1200, 64, 400, signal, 0)
            f = H.train_bilinear(z[:900], c[:900], y[:900], z[900:], c[900:], y[900:], T, 8, 1e-3, 0, max_epochs=40)
            acc = float((H.argmax_canonical(H.predict_proba([f], z[900:], c[900:], T)) == y[900:]).float().mean())
            self.assertTrue(lo <= acc <= hi, (signal, acc))

    def test_determinism_fixed_epochs_and_padding(self):
        z, c, y, T = synth(300, 16, 100, 1.0, 2)
        f1 = H.train_bilinear(z, c, y, z, c, y, T, 8, 1e-3, 0, fixed_epochs=3)
        f2 = H.train_bilinear(z, c, y, z, c, y, T, 8, 1e-3, 0, fixed_epochs=3)
        self.assertTrue(torch.equal(f1.U, f2.U) and f1.best_epoch == 3)
        c4 = c.clone()
        c4[:, 4:] = H.PAD  # depth-3 style: 4 valid candidates
        p = H.predict_proba([f1], z, c4, T)
        self.assertTrue(torch.all(p[:, 4:] == 0) and torch.allclose(p.sum(-1), torch.ones(len(z))))


class PermutationInvariance(unittest.TestCase):
    """Permuting the candidates and mapping back must not change predictions in LABEL space."""

    def setUp(self):
        z, c, y, T = synth(400, 32, 200, 1.0, 5)
        self.z, self.c, self.T = z, c, T
        self.fits = [H.train_bilinear(z, c, y, z, c, y, T, 8, 1e-3, s, fixed_epochs=4) for s in (0, 1)]
        g = torch.Generator().manual_seed(9)
        self.perm = torch.stack([torch.randperm(6, generator=g) for _ in range(len(z))])
        self.cp = torch.gather(c, 1, self.perm)

    def test_probe_scores_equivariant_and_labels_invariant(self):
        p = H.predict_proba(self.fits, self.z, self.c, self.T)
        pp = H.predict_proba(self.fits, self.z, self.cp, self.T)
        torch.testing.assert_close(torch.gather(p, 1, self.perm), pp)  # equivariance
        lab = torch.gather(self.c, 1, p.argmax(-1, keepdim=True))[:, 0]
        labp = torch.gather(self.cp, 1, pp.argmax(-1, keepdim=True))[:, 0]
        self.assertTrue(torch.equal(lab, labp))

    def test_canonical_order_removes_input_order(self):
        self.assertTrue(torch.equal(torch.sort(self.c, 1).values, torch.sort(self.cp, 1).values))
        self.assertEqual(F.canonical_candidates(["ZB12", "A9Q0", "K7K7"]), F.canonical_candidates(["K7K7", "ZB12", "A9Q0"]))

    def test_joint_assignment_b6_and_order_label_invariant(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            lp = np.log(rng.dirichlet(np.ones(6), size=4))
            perm = rng.permutation(6)  # new position j holds old candidate perm[j]
            a = H.joint_assignment(lp, 4)
            ap = H.joint_assignment(lp[:, perm], 4)
            self.assertEqual([int(x) for x in a], [int(perm[j]) for j in ap])  # same LABELS assigned
            tgt = int(rng.integers(6))
            self.assertEqual(H.b6_expectation(lp, tgt), H.b6_expectation(lp[:, perm], int(np.where(perm == tgt)[0][0])))
            lo = lp[:3]
            true = tuple(int(x) for x in rng.permutation(6)[:3])
            truep = tuple(int(np.where(perm == t)[0][0]) for t in true)
            self.assertEqual(H.order_correct(lo, true), H.order_correct(lo[:, perm], truep))

    def test_ties_broken_by_canonical_index_not_slot(self):
        prob = torch.tensor([[0.3, 0.3, 0.1, 0.1, 0.1, 0.1]])
        self.assertEqual(int(H.argmax_canonical(prob)[0]), 0)
        self.assertEqual(H.joint_assignment(np.zeros((3, 6)), 3), (0, 1, 2))

    def test_b6_duplicates_and_product(self):
        lp = np.full((4, 6), -10.0)
        lp[:, 0] = 0.0
        self.assertEqual(len(set(H.joint_assignment(lp, 4))), 4)
        self.assertIn(H.b6_expectation(lp, 5), (0.0, 0.5))
        self.assertEqual(list(H.paired_accuracy(np.array([0.5, 0.5]), [(0, 1)])), [0.25])


class OracleFreeInputs(unittest.TestCase):
    """Prediction inputs are built from model-input records (no oracle field) + the model's choice."""

    @classmethod
    def setUpClass(cls):
        d = ROOT / "data" / "d5" / "probe_dev"
        cls.inputs = [json.loads(x) for x in (d / "model_inputs.jsonl").read_text(encoding="utf-8").splitlines()]
        cls.worlds = {r["world_id"]: r for r in (json.loads(x) for x in (d / "worlds.jsonl").read_text(encoding="utf-8").splitlines())}

    def test_inputs_have_no_oracle_and_builders_take_no_world(self):
        self.assertTrue(all("oracle" not in r for r in self.inputs))
        for fn in (F.candidates_from_prompt, F.b3_features, F.canonical_candidates):
            params = inspect.signature(fn).parameters
            self.assertNotIn("world", params)
            self.assertNotIn("oracle", params)

    def test_prompt_candidates_equal_oracle_intermediates(self):
        n = 0
        for r in self.inputs:
            if r["variant"].startswith("decoy"):
                continue
            w = self.worlds[r["world_id"]]
            truth = sorted({s for p in w["oracle"]["paths"].values() for s in p[1:-1]})
            self.assertEqual(F.candidates_from_prompt(r["prompt"]), truth)
            self.assertEqual(len(truth), 2 * (w["depth"] - 1))
            n += 1
        self.assertEqual(n, 4200)

    def test_b3_features_permutation_equivariant(self):
        r = next(x for x in self.inputs if x["variant"] == "base")
        c = F.candidates_from_prompt(r["prompt"])
        f1 = F.b3_features(r["prompt"], c, "A")
        perm = [3, 0, 5, 1, 4, 2][:len(c)]
        f2 = F.b3_features(r["prompt"], [c[i] for i in perm], "A")
        np.testing.assert_array_equal(f1[perm], f2)

    def test_supervision_uses_model_choice(self):
        w = next(x for x in self.worlds.values() if x["variant"] == "base" and x["depth"] == 4)
        a, b = F.supervision_targets(w, "A"), F.supervision_targets(w, "B")
        self.assertEqual(a["u2"], b["c2"])
        self.assertEqual(set(a), {"c1", "c2", "c3", "u1", "u2", "u3"})


class Budget(unittest.TestCase):
    def test_overrun_raises(self):
        import time
        m = H.BudgetMeter(cpu_hours=1e-9)
        t = time.process_time()
        while time.process_time() - t < 0.1:
            pass
        with self.assertRaises(H.BudgetExceeded):
            m.tick("fit")

    def test_grid_is_v03(self):
        self.assertEqual(H.LAYERS, (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 61))
        self.assertEqual((H.GRID_R, len(H.GRID_WD), H.SEEDS), ((8, 32), 4, (0, 1, 2)))


class OrderSameLayer(unittest.TestCase):
    def test_heads_must_share_layer(self):
        z, c, y, T = synth(60, 16, 80, 1.0, 3)
        f = H.train_bilinear(z, c, y, z, c, y, T, 8, 1e-3, 0, fixed_epochs=1)
        heads = {"u1": [f], "u2": [f], "u3": [f]}
        true = np.tile(np.array([0, 1, 2]), (60, 1))
        self.assertEqual(H.l2_order_predict(heads, {"u1": 12, "u2": 12, "u3": 12}, 12, z, c, T, true).shape, (60,))
        with self.assertRaises(AssertionError):
            H.l2_order_predict(heads, {"u1": 8, "u2": 12, "u3": 12}, 12, z, c, T, true)


if __name__ == "__main__":
    unittest.main()


class ConditionalLogitB3(unittest.TestCase):
    def test_deterministic_and_counted_in_budget(self):
        g = torch.Generator().manual_seed(0)
        X = torch.randn(500, 6, 7, generator=g)
        wtrue = torch.randn(7, generator=g)
        y = (X @ wtrue).argmax(-1)
        mask = torch.zeros(500, 6, dtype=torch.bool)
        m = H.BudgetMeter(cpu_hours=48)
        w1 = H.train_conditional_logit(X, mask, y, 1e-3, m)
        w2 = H.train_conditional_logit(X, mask, y, 1e-3, m)
        self.assertTrue(torch.equal(w1, w2))
        self.assertEqual(m.n_fits, 2)
        self.assertGreater(float(((X @ w1).argmax(-1) == y).float().mean()), 0.8)
