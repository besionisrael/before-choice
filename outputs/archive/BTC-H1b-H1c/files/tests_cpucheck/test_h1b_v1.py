"""BTC-H1b v1.0 freeze tests (synthetic / tiny; no LOCKED-TEST-H1b is generated or read)."""

import json
import pathlib
import sys
import unittest

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "sealed_stage"))
import h1_features as F  # noqa: E402
import h1_probe as H  # noqa: E402
import build_h1b_pool as G  # noqa: E402
import h1b_eval_test as E  # noqa: E402
from h1b_freeze_dev import depth2_from_prompt  # noqa: E402

DEV = ROOT / "data" / "d5" / "probe_dev"


class Generator(unittest.TestCase):
    def test_tiny_pool_structure(self):
        rows = G.build(n_families=10, seed=99, fid_base=99000, exclude=set())   # NOT the H1b seed
        self.assertEqual(len(rows), 40)
        self.assertEqual({r["variant"] for r in rows}, set(G.KEEP))
        self.assertTrue(all(r["depth"] == 4 for r in rows))
        for r in rows:
            self.assertEqual(len(F.candidates_from_prompt(r["prompt"])), 6)
            self.assertEqual(len(depth2_from_prompt(r["prompt"])), 2)
            self.assertNotIn("oracle", json.dumps({k: r[k] for k in ("prompt",)}))

    def test_generation_refused_without_approval(self):
        import subprocess
        p = subprocess.run([sys.executable, str(ROOT / "src" / "sealed_stage" / "build_h1b_pool.py")], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("REFUSED", p.stdout + p.stderr)
        self.assertFalse((ROOT / "data" / "d5" / "locked_test_h1b").exists())


def _setup(planted_c1=True, seed=0):
    """Real DEV depth-4 worlds (4 variants), all admissible; keys random 16-d; query = [key(u2)[:8], key(c1)[8:]]."""
    rng = np.random.default_rng(seed)
    W = [json.loads(x) for x in (DEV / "worlds.jsonl").read_text(encoding="utf-8").splitlines()]
    W = [w for w in W if w["depth"] == 4 and w["variant"] in ("base", "cf", "upath_base", "upath_cf")]
    inputs = {w["world_id"]: {"prompt": w["prompt"]} for w in W}
    idx = [{"world_id": w["world_id"], "logp_A": -0.1 if w["oracle"]["optimal_action"] == "A" else -3.0,
            "logp_B": -3.0 if w["oracle"]["optimal_action"] == "A" else -0.1} for w in W]
    rows, audit = E.build_rows(idx, inputs, {w["world_id"]: w for w in W})
    d = 16
    acts = np.zeros((len(idx), 7, d), np.float32)
    for r in rows:
        k = rng.standard_normal((6, d)).astype(np.float32)
        acts[r["g"], 1:7] = k
        acts[r["g"], 0, :8] = 3 * k[r["tgt"]["u2"], :8]
        acts[r["g"], 0, 8:] = 3 * k[r["tgt"]["c1"], 8:] if planted_c1 else 0
    A = {L: acts for L in (32, 40, 48)}
    stats = {}
    for pre in ("L40_u2", "L40_c1", "L40_c2", "L32_u2", "L48_u2"):
        stats.update({f"{pre}_zmu": np.zeros(d), f"{pre}_zsd": np.ones(d), f"{pre}_kmu": np.zeros(d), f"{pre}_ksd": np.ones(d),
                      f"{pre}_mu_A": np.zeros(d), f"{pre}_mu_B": np.zeros(d)})
    first, second = torch.zeros(d, 8), torch.zeros(d, 8)
    first[:8] = torch.eye(8)
    second[8:] = torch.eye(8)
    bil = lambda M: {"U": [M], "V": [M], "E": 1}
    dec = {"P_u2_L40": bil(first), "P_u2_L32": bil(first), "P_u2_L48": bil(first), "C_bil_u2_L40": bil(first),
           "P_c1_L40": bil(second), "P_c2_L40": bil(torch.zeros(d, 8)),
           "C_lin_u2_L40": {"w": [torch.zeros(2 * d)]}, "K_u2_L40": {"w": [torch.zeros(d)]},
           "B3plus_u2": {"w": [torch.zeros(len(F.B3_FEATURES) + 1)]}}
    return rows, A, dec, stats, audit


class EvalCore(unittest.TestCase):
    def test_planted_query_signal_met_and_perm_between_families(self):
        rows, A, dec, stats, audit = _setup()
        self.assertEqual(audit.get("n_not_admissible", 0), 0)
        res = E.evaluate(rows, A, dec, stats, min_families=50, n_boot=500, k_perm=5)
        self.assertGreater(res["P_c1_single"], 0.8)
        self.assertEqual(res["outcome"], "H1b MET")
        self.assertGreaterEqual(res["point_paired"]["P"], 0.85)   # 8-d random keys: a few dot-product collisions
        self.assertLess(res["point_paired"]["P_perm"], 0.3)
        self.assertGreaterEqual(res["perm_partners"]["min"], 1)
        self.assertIn("not necessarily an independent representation", res["interpretation"])
        self.assertEqual(set(res["secondary"]), {"S1_P_minus_B3plus", "S2_P_minus_K", "S3_DeltaC_single", "S4_P_minus_elim"})

    def test_sensitivity_gate_and_insufficient(self):
        rows, A, dec, stats, _ = _setup(planted_c1=False)
        self.assertEqual(E.evaluate(rows, A, dec, stats, min_families=50, n_boot=200, k_perm=2)["outcome"], "SENSITIVITY CHECK FAILED")
        rows, A, dec, stats, _ = _setup()
        self.assertEqual(E.evaluate(rows, A, dec, stats, min_families=10_000, n_boot=200, k_perm=2)["outcome"], "INSUFFICIENT")

    def test_cmax_inside_replicates(self):
        # an equally strong control (C_bil given P's weights AND P's query path is impossible by design, so emulate it by
        # making P_c2/C structure irrelevant): with C_bil = P (same weights, zero query -> uniform) Delta_C still uses max()
        rows, A, dec, stats, _ = _setup()
        res = E.evaluate(rows, A, dec, stats, min_families=50, n_boot=300, k_perm=2)
        self.assertIn(res["Delta_C"]["C_max_point"], ("C_bil", "C_lin"))
        self.assertAlmostEqual(res["Delta_C"]["point"], res["point_paired"]["P"] - max(res["point_paired"]["C_bil"], res["point_paired"]["C_lin"]))

    def test_constants_frozen(self):
        self.assertEqual((E.MIN_FAMILIES, E.C1_MIN, E.B_RESAMPLES, E.BOOT_SEED, E.K_PERM), (875, 0.80, 10_000, 0, 50))


if __name__ == "__main__":
    unittest.main()
