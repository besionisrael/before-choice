"""BTC-H1b exploratory runner: synthetic shards only (planted u2 signal at vector 12)."""

import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import h1_train_dev as T  # noqa: E402
import h1b_explore as X  # noqa: E402


class H1bRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = tempfile.TemporaryDirectory()
        base = pathlib.Path(cls.t.name)
        cap, devdir = T.make_synthetic(base / "syn", n_worlds=480, d=16)
        cls.data = X.Data(T.Dev(cap, devdir), devdir)
        cls.out = base / "out"
        cls.s = X.run(cls.data, cls.out, layers=(0, 12), x1b_layers=(12,), x2_layers=(12,), seeds=(0,), max_epochs=20, x7_vector=12)

    @classmethod
    def tearDownClass(cls):
        cls.t.cleanup()

    def test_plan_written_before_results_and_splits_family_disjoint(self):
        plan = json.loads((self.out / "PLAN.json").read_text(encoding="utf-8"))
        self.assertIn("X7_thresholds_from_train_margins", plan)
        f = lambda ids: {self.data.dev.rows[i]["family_id"] for i in ids}
        self.assertFalse(f(self.data.itr) & f(self.data.iva))
        self.assertFalse(f(self.data.tr) & f(self.data.va))
        self.assertEqual(sorted(self.data.itr + self.data.iva), sorted(self.data.tr))

    def test_all_blocks_logged_and_planted_signal_found(self):
        recs = [json.loads(x) for x in (self.out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual({r["block"] for r in recs}, {"B3plus", "X1a", "X1b", "X2", "LC", "X7"})
        u2 = self.s["X1b"]["u2"]["12"]
        self.assertGreater(u2["inner_val"], 0.3)          # planted signal recovered (chance ~0.2; ~300 synthetic train worlds)
        self.assertIn("X1b|u2", self.s["X7"])

    def test_single_block_only(self):
        with self.assertRaises(SystemExit):
            X.run(self.data, self.out, layers=(0,), x1b_layers=(), x2_layers=(), seeds=(0,), max_epochs=2)


if __name__ == "__main__":
    unittest.main()
