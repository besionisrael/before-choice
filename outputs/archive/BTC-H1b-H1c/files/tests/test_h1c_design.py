"""BTC-H1c design: generator invariants (design seed; no pool)."""

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_h1c as G  # noqa: E402


class H1cDesign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = G.build(40, 4242, 91000)
        cls.fam = {}
        for w in cls.rows:
            cls.fam.setdefault(w["family_id"], {})[w["variant"]] = w

    def test_pairs_keep_choice_branch_counts_and_change_u2(self):
        for v in self.fam.values():
            for o in ("L", "E"):
                a, b = v[f"base_{o}"], v[f"dswap_{o}"]
                opt = a["oracle"]["optimal_action"]
                non = "B" if opt == "A" else "A"
                self.assertEqual(b["oracle"]["optimal_action"], opt)
                self.assertEqual(a["terminal_rewards"], b["terminal_rewards"])
                self.assertEqual(a["oracle"]["paths"][opt], b["oracle"]["paths"][opt])
                self.assertNotEqual(a["oracle"]["paths"][non][2], b["oracle"]["paths"][non][2])
                self.assertEqual(G.one_hop_view(a["prompt"], opt), G.one_hop_view(b["prompt"], opt))
                self.assertEqual(G.counts_view(a["prompt"]), G.counts_view(b["prompt"]))
                self.assertEqual(len(G.parse(a["prompt"])["cands"]), 12)

    def test_full_structure_control_is_exact(self):
        for w in self.rows:
            opt = w["oracle"]["optimal_action"]
            self.assertEqual(G.full_structure_u2(w["prompt"], opt), w["oracle"]["paths"]["B" if opt == "A" else "A"][2])

    def test_no_pool_written_by_default(self):
        import subprocess
        p = subprocess.run([sys.executable, str(ROOT / "src" / "build_h1c.py")], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)


if __name__ == "__main__":
    unittest.main()


class RewardFreeSymmetry(unittest.TestCase):
    def test_binary_candidates_symmetric_without_rewards(self):
        sys.path.insert(0, str(ROOT / "src"))
        import h1c_token_checks as TC
        for w in G.build(20, 99, 92000):
            if w["variant"] not in ("base_L", "dswap_L"):
                continue
            p = G.parse(w["prompt"])
            non = "B" if w["oracle"]["optimal_action"] == "A" else "A"
            b = sorted(t for s, t in p["mv"] if s == p["act"][non])
            self.assertEqual(TC.reward_free_signature(p["mv"], b[0]), TC.reward_free_signature(p["mv"], b[1]))
