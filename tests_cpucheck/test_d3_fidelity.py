"""D3 technical amendment A2: corrected fidelity rule (incident BTC-D3-DEV). Synthetic logit vectors only."""

import pathlib
import sys
import unittest

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from d3_capture_core import FIDELITY_MAX_ABS, TAU, FidelityError, check_fidelity  # noqa: E402

V, A, B = 1000, 10, 11


def base():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(V, generator=g) * 3
    x[A], x[B] = 20.0, 15.0  # A clearly preferred
    return x


class Fidelity(unittest.TestCase):
    def test_constants_fixed(self):
        self.assertEqual((FIDELITY_MAX_ABS, TAU), (0.25, 0.125))

    def test_identical(self):
        o = base()
        r = check_fidelity(o, o.clone(), A, B, "W0", "post_norm")
        self.assertTrue(r["top1_ok"] and not r["near_tie_top1"] and r["ab_decision_same"])

    def test_two_token_exact_tie(self):
        o = base()
        o[A] = 25.0
        o[500] = 25.0                      # exact tie between A and token 500 (argmax picks A)
        rcn = o.clone()
        rcn[500] += 0.1                    # bf16 rounding reorders the tie
        r = check_fidelity(o, rcn, A, B, "W1", "post_norm")
        self.assertEqual(r["recon_top1"], 500)
        self.assertTrue(r["near_tie_top1"] and r["top1_ok"])

    def test_three_token_near_tie_beyond_top2(self):
        o = base()
        o[A], o[600], o[700] = 25.0, 24.95, 24.90   # three numerically indiscernible tokens
        rcn = o.clone()
        rcn[700] += 0.12                            # third-ranked token becomes reconstructed top-1
        r = check_fidelity(o, rcn, A, B, "W2", "pre_norm")
        self.assertEqual(r["recon_top1"], 700)       # not limited to the top-2
        self.assertAlmostEqual(r["orig_margin_top1_vs_recon_top1"], 0.10, places=5)
        self.assertTrue(r["near_tie_top1"] and r["top1_ok"])

    def test_excessive_logit_divergence_aborts(self):
        o = base()
        rcn = o.clone()
        rcn[123] += 0.30                            # > 0.25 somewhere in the full vocabulary
        with self.assertRaises(FidelityError) as cm:
            check_fidelity(o, rcn, A, B, "W3", "post_norm")
        self.assertIn("[W3]", str(cm.exception))

    def test_ab_inversion_aborts(self):
        o = base()
        o[A], o[B] = 15.10, 15.00                   # A preferred by 0.10
        rcn = o.clone()
        rcn[A] -= 0.08
        rcn[B] += 0.08                              # within 0.25, but B now preferred
        with self.assertRaises(FidelityError) as cm:
            check_fidelity(o, rcn, A, B, "W4", "post_norm")
        self.assertIn("ab_decision_same': False", str(cm.exception))

    def test_genuine_reconstruction_error_aborts(self):
        o = base()
        g = torch.Generator().manual_seed(1)
        rcn = torch.randn(V, generator=g) * 3       # wrong vector entirely
        with self.assertRaises(FidelityError) as cm:
            check_fidelity(o, rcn, A, B, "W5", "pre_norm")
        self.assertIn("[W5]", str(cm.exception))

    def test_original_ab_tie_logged(self):
        o = base()
        o[A] = o[B] = 18.0                          # exact bf16 tie in the ORIGINAL logits
        rcn = o.clone()
        rcn[A] += 0.01
        r = check_fidelity(o, rcn, A, B, "W6", "post_norm")
        self.assertTrue(r["ab_tie_original"] and r["ab_decision_same"])

    def test_top1_rule_is_implied_by_maxdiff(self):
        # with max|diff| = d over the full vocabulary, any reconstructed top-1 satisfies margin <= 2d
        g = torch.Generator().manual_seed(2)
        for _ in range(200):
            o = torch.randn(V, generator=g)
            rcn = o + (torch.rand(V, generator=g) - 0.5) * 0.4  # |diff| <= 0.2
            t1, r1 = int(o.argmax()), int(rcn.argmax())
            d = float((rcn - o).abs().max())
            self.assertLessEqual(float(o[t1] - o[r1]), 2 * d + 1e-6)


if __name__ == "__main__":
    unittest.main()


class NonFinite(unittest.TestCase):
    def test_nan_or_inf_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            o = base()
            rcn = o.clone()
            rcn[42] = bad
            with self.assertRaises(FidelityError) as cm:
                check_fidelity(o, rcn, A, B, "W7", "post_norm")
            self.assertIn("non-finite", str(cm.exception))
            o2 = base()
            o2[7] = bad
            with self.assertRaises(FidelityError):
                check_fidelity(o2, base(), A, B, "W8", "pre_norm")
