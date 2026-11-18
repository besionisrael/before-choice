"""CPU tests for BTC-D2-002 selection and scorer (synthetic outputs, in memory only)."""

import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from score_d2_competence import THRESHOLD, load_oracle, score  # noqa: E402

HAS_MODAL = importlib.util.find_spec("modal") is not None
ORACLE = load_oracle()


def fake(policy, tie_ids=()):
    raw = []
    ids = sorted(ORACLE)
    for rep, wids in ((0, ids), (1, ids[:20])):
        for w in wids:
            c = "TIE" if w in tie_ids else policy(ORACLE[w])
            a, b = {"A": (-0.1, -3.0), "B": (-3.0, -0.1), "TIE": (-0.7, -0.7)}[c]
            raw.append({"world_id": w, "replay": rep, "logp_A": a, "logp_B": b, "top5": [["A", a]]})
    return raw


class Selection(unittest.TestCase):
    @unittest.skipUnless(HAS_MODAL, "modal not installed")
    def test_train_val_only_and_limits(self):
        import d2_competence as d
        sel = d.select_worlds()
        self.assertEqual(len(sel), 352)
        self.assertEqual({r["split"] for r in sel}, {"train", "val"})
        src = (ROOT / "src" / "d2_competence.py").read_text(encoding="utf-8")
        self.assertIn('gpu="L4", image=image, timeout=900, retries=0, max_containers=1', src)
        self.assertNotIn("output_hidden_states", src)
        self.assertNotIn('"test.jsonl"', src)
        self.assertEqual(d.MODEL_REVISION, "cdbee75f17c01a7cc42f958dc650907174af0554")

    def test_oracle_has_no_test_rows(self):
        self.assertEqual(len(ORACLE), 352)
        self.assertTrue(all(r["split"] in ("train", "val") for r in ORACLE.values()))


class Scorer(unittest.TestCase):
    def test_perfect(self):
        m = score(fake(lambda o: o["oracle"]["optimal_action"]), ORACLE)["metrics"]
        c = m["CHECKPOINT_depth2_base_cf"]
        self.assertEqual((c["n"], c["n_families"], c["acc_primary"], c["decision"]), (16, 8, 1.0, "PASS"))
        self.assertEqual(m["cf_sensitivity"]["base_vs_cf"]["both_correct"], 8 * 3)
        self.assertEqual(m["replay"]["identical_choices"], 20)

    def test_constant_policy_fails(self):
        m = score(fake(lambda o: "A"), ORACLE)["metrics"]
        c = m["CHECKPOINT_depth2_base_cf"]
        self.assertEqual((c["acc_primary"], c["decision"]), (0.5, "FAIL"))
        self.assertEqual(m["cf_sensitivity"]["base_vs_cf"]["same_choice"], 24)

    def test_tie_rules(self):
        d2 = sorted(w for w, r in ORACLE.items() if r["depth"] == 2 and r["variant"] in ("base", "cf"))
        # 11 correct, 1 wrong, 4 ties -> primary 13/16=0.8125, strict 11/16=0.6875, lenient 15/16
        ties, wrong = set(d2[:4]), d2[4]
        pol = lambda o: o["oracle"]["optimal_action"] if o["world_id"] != wrong else ("B" if o["oracle"]["optimal_action"] == "A" else "A")
        c = score(fake(pol, ties), ORACLE)["metrics"]["CHECKPOINT_depth2_base_cf"]
        self.assertEqual(c["n_ties"], 4)
        self.assertAlmostEqual(c["acc_primary"], 13 / 16)
        self.assertAlmostEqual(c["acc_strict"], 11 / 16)
        self.assertEqual(c["decision"], "PASS")
        self.assertTrue(c["tie_sensitive"])
        self.assertEqual(THRESHOLD, 0.70)


if __name__ == "__main__":
    unittest.main()
