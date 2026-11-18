"""BTC-H1c DEV preparation tests (synthetic captures built from the real H1c DEV pool; no model)."""

import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import h1c_dev_analysis as A  # noqa: E402
import h1c_gates as G  # noqa: E402


class Gates(unittest.TestCase):
    def test_bitwise_gate_passes_on_identical_keys_and_blocks_otherwise(self):
        with tempfile.TemporaryDirectory() as t:
            cap, dev = A.make_synthetic(pathlib.Path(t) / "ok", n_fam=30)
            g = G.gates(cap, dev, None)
            self.assertTrue(g["G2_bitwise_identity"]["passed"])
            self.assertTrue(g["G3_competence"]["passed"])
            self.assertTrue(g["analysis_allowed"])
            cap2, dev2 = A.make_synthetic(pathlib.Path(t) / "bad", n_fam=30, break_identity=True)
            g2 = G.gates(cap2, dev2, None)
            self.assertFalse(g2["G2_bitwise_identity"]["passed"])
            self.assertFalse(g2["analysis_allowed"])


class Analysis(unittest.TestCase):
    def test_planted_continuation_found_and_key_controls_zero_by_construction(self):
        with tempfile.TemporaryDirectory() as t:
            cap, dev = A.make_synthetic(pathlib.Path(t), n_fam=120)
            r = A.run(A.Cap(cap, dev, None), pathlib.Path(t) / "out", seeds=(0,), max_epochs=40, layers=[40])
            self.assertGreater(r["P"]["val_paired_late"][0], 0.4)
            for c in ("key_only", "C_lin", "C_bil"):
                self.assertEqual(r[c]["val_paired_late"][0], 0.0)      # identical keys -> identical prediction
            self.assertLess(r["P_perm"]["val_paired_late"], 0.4)
            self.assertIn("PC1_c1_12way", r)
            rp = r["P"]["robust_primary_late"]
            self.assertGreater(rp["single_acc_late"], 0.6)
            self.assertLess(rp["signflip_p_one_sided"], 0.025)
            for c in ("key_only", "C_lin", "C_bil"):
                self.assertAlmostEqual(r[c]["robust_primary_late"]["single_acc_late"], 0.5)   # one right, one wrong per pair
            self.assertEqual(r["selected"]["layer"], 40)

    def test_selection_and_report_use_distinct_families(self):
        with tempfile.TemporaryDirectory() as t:
            cap, dev = A.make_synthetic(pathlib.Path(t), n_fam=60)
            c = A.Cap(cap, dev, None)
            f = lambda ids: {c.rows[k]["family_id"] for k in ids}
            self.assertFalse(f(c.itr) & f(c.iva))
            self.assertFalse((f(c.itr) | f(c.iva)) & f(c.va))


if __name__ == "__main__":
    unittest.main()
