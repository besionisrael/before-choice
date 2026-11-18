"""BTC-H1b X3+X4 analysis: synthetic multi-position shards (planted pointer signal for c1 at ANS / Q_END)."""

import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import h1b_x3x4_analysis as A  # noqa: E402


class X3X4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = tempfile.TemporaryDirectory()
        b = pathlib.Path(cls.t.name)
        cls.data, xdir, pos = A.make_synthetic_x(b, n_worlds=960)
        cls.out = b / "out"
        cls.s = A.run(cls.data, A.XData(cls.data, xdir, pos), cls.out, layers=[8], queries=("ANS", "ACT_END_A"), seeds=(0,), max_epochs=60)

    @classmethod
    def tearDownClass(cls):
        cls.t.cleanup()

    def cell(self, slot, rep, q):
        return next(r for r in self.s["grid"] if r["slot"] == slot and r["rep"] == rep and r["query"] == q)

    def test_representation_and_position_effects_separated(self):
        hit = self.cell("c1", "ctx_tgt", "ANS")["val"]
        self.assertGreater(hit, 0.45)                                         # pointer finds the planted label
        for rep, q in (("static", "ANS"), ("ctx_src", "ANS"), ("ctx_tgt", "ACT_END_A")):
            self.assertLess(self.cell("c1", rep, q)["val"], hit - 0.2)        # static keys / wrong key / wrong position fail

    def test_decision_uses_predeclared_rule_and_all_logged(self):
        sc = self.s["selected_cell"]
        self.assertEqual((sc["rep"], sc["query"]), ("ctx_tgt", "ANS"))
        self.assertEqual(set(sc["criteria"]), {"S-A", "S-B", "S-C"})
        self.assertEqual(sc["verdict"], "SENSITIVE" if all(sc["criteria"].values()) else "NOT SENSITIVE")
        recs = [json.loads(x) for x in (self.out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual({r["block"] for r in recs}, {"B3plus", "grid", "key_only", "choice_keys", "decision"})
        self.assertTrue((self.out / "PLAN.json").exists())

    def test_owner_amendments_choice_keys_seeds_per_choice(self):
        self.assertTrue(self.s["choice_keys"])                                   # choice + contextual keys control
        self.assertIn("c1_choice_keys_val", self.s["selected_cell"])
        self.assertEqual(len(self.cell("c1", "ctx_tgt", "ANS")["epochs"]), 1)     # the seeds passed (harmonised everywhere)
        c1 = self.s["selected_all_slots"]["c1"]
        self.assertEqual(c1["n_val_choice_A"] + c1["n_val_choice_B"], len([i for i in self.data.va if "c1" in self.data.dev.rows[i]["tgt"]]))
        self.assertIn("val_choice_A", self.s["selected_all_slots"]["A1"])
        plan = json.loads((self.out / "PLAN.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["decoder"]["seeds"], [0, 1, 2])
        self.assertIn("missing_positions", plan)
        self.assertIn("no automatic causal reading", plan["pre_reward_queries"])

    def test_single_analysis_only(self):
        with self.assertRaises(SystemExit):
            A.run(self.data, None, self.out)


if __name__ == "__main__":
    unittest.main()
