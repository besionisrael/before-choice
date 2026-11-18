"""Runner tests (synthetic shards only; no real activations)."""

import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import h1_probe as H  # noqa: E402
import h1_train_dev as T  # noqa: E402


class Plan(unittest.TestCase):
    def test_frozen_counts(self):
        G = T.plan(H.LAYERS, H.GRID_R, H.GRID_WD, H.SEEDS)
        self.assertEqual({k: len(v) for k, v in G.items()}, {"P": 408, "A": 1632, "B1": 24, "B5": 24, "B3": 16})
        bilinear = len(G["P"]) + len(G["A"]) + len(G["B1"]) + len(G["B5"]) + 6 + 9 + 27  # + order heads + B2 + refits
        self.assertEqual(bilinear, 2130)
        self.assertEqual((T.THREADS, T.CPU_HOURS), (4, 48.0))

    def test_no_sealed_path_literal(self):
        self.assertNotIn("locked_test", (ROOT / "src" / "h1_train_dev.py").read_text(encoding="utf-8"))


class Runner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cap, cls.devdir = T.make_synthetic(pathlib.Path(cls.tmp.name) / "syn", n_worlds=240, d=16)
        cls.dev = T.Dev(cls.cap, cls.devdir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_end_to_end_and_restart_refused(self):
        out = pathlib.Path(self.tmp.name) / "out1"
        st = T.run(self.dev, out, layers=(0, 12), ranks=(8,), wds=(1e-3,), seeds=(0,), max_epochs=3)
        self.assertTrue(st["completed_all_selection_grids"])
        self.assertEqual(len(st["final_refits"]["predictors"]), 16)
        self.assertTrue((out / "FINAL.json").exists() and (out / "SELECTIONS.json").exists())
        with self.assertRaises(SystemExit):
            T.run(self.dev, out, layers=(0, 12), ranks=(8,), wds=(1e-3,), seeds=(0,), max_epochs=3)

    def test_selection_refuses_incomplete_grid(self):
        with tempfile.TemporaryDirectory() as t:
            L = T.Ledger(pathlib.Path(t) / "fits.jsonl")
            keys = [("P", "u2", l, 8, 1e-3, 0) for l in (0, 12)]
            L.add({"key": T.key(keys[0]), "best_epoch": 3, "val_paired_acc": 0.5})
            with self.assertRaises(AssertionError):
                T.select(L, keys, "val_paired_acc")

    def test_budget_exceeded_stops_without_selection(self):
        out = pathlib.Path(self.tmp.name) / "out2"
        old = T.CPU_HOURS
        T.CPU_HOURS = 1e-9
        try:
            with self.assertRaises(SystemExit):
                T.run(self.dev, out, layers=(0, 12), ranks=(8,), wds=(1e-3,), seeds=(0,), max_epochs=2)
        finally:
            T.CPU_HOURS = old
        self.assertTrue((out / "BUDGET_EXCEEDED.json").exists())
        self.assertFalse((out / "SELECTIONS.json").exists())
        rec = json.loads((out / "BUDGET_EXCEEDED.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(rec["completed_fits"], 1)


if __name__ == "__main__":
    unittest.main()


class B2Slots(unittest.TestCase):
    def test_depth3_never_falls_back_and_is_paired(self):
        r6 = {"family_id": "F20001", "tgt": {s: i for i, s in enumerate(H.SLOTS)}}
        r4 = {"family_id": "F20250", "tgt": {"c1": 0, "c2": 1, "u1": 2, "u2": 3}}
        for k in range(3):
            self.assertIn(T.b2_slot(r6, k), H.SLOTS)
            s4 = T.b2_slot(r4, k)
            self.assertIn(s4, r4["tgt"])
            self.assertEqual(s4, T.b2_slot(dict(r4), k))  # same family -> same slot (paired)
        # across many depth-3 families the permuted slot is not always u2 (permutation actually applied)
        slots = {T.b2_slot({"family_id": f"F{20200 + i}", "tgt": r4["tgt"]}, 0) for i in range(40)}
        self.assertGreater(len(slots), 1)


class LedgerFields(unittest.TestCase):
    def test_train_and_val_metrics_and_cap_flag(self):
        with tempfile.TemporaryDirectory() as t:
            cap, devdir = T.make_synthetic(pathlib.Path(t) / "s", n_worlds=240, d=16)
            out = pathlib.Path(t) / "o"
            st = T.run(T.Dev(cap, devdir), out, layers=(0, 12), ranks=(8,), wds=(1e-3,), seeds=(0,), max_epochs=3)
            recs = [json.loads(x) for x in (out / "fits.jsonl").read_text(encoding="utf-8").splitlines()]
            p = [r for r in recs if r["key"].startswith("P|")]
            self.assertTrue(all("train_paired_acc" in r and "val_paired_acc" in r and "hit_epoch_cap" in r for r in p))
            a = [r for r in recs if r["key"].startswith("A|")]
            self.assertTrue(all("train_world_acc" in r and "val_world_acc" in r for r in a))
            self.assertEqual(len([r for r in recs if r["key"].startswith("B2|")]), 3)  # 3 perms x 1 seed here
            self.assertIn("B2_perm0", st["final_refits"]["predictors"])
            self.assertEqual(st["final_refits"]["n_predictors"], 16)
