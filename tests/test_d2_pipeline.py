"""CPU tests of the D2 pipeline. Uses SYNTHETIC model outputs generated here only to
test the scorer's arithmetic; nothing synthetic is ever written to outputs/."""

import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from score_d2 import load_oracle, score  # noqa: E402

HAS_MODAL = importlib.util.find_spec("modal") is not None


@unittest.skipUnless(HAS_MODAL, "modal client not installed")
class SmokeSelection(unittest.TestCase):
    def test_selection_fixed_train_only(self):
        import d2_smoke
        sel = d2_smoke.select_worlds()
        self.assertEqual(len(sel), 20)
        self.assertTrue(all(r["split"] == "train" for r in sel))
        self.assertEqual(len({r["world_id"] for r in sel}), 20)
        self.assertEqual(sum(r["variant"].startswith("upath") for r in sel), 4)
        self.assertEqual(sum(r["variant"].startswith("cpath") for r in sel), 4)

    def test_function_limits(self):
        import d2_smoke
        self.assertEqual(d2_smoke.MODEL_REVISION, "cdbee75f17c01a7cc42f958dc650907174af0554")
        self.assertEqual((d2_smoke.TOKEN_A, d2_smoke.TOKEN_B), (32, 33))
        src = (ROOT / "src" / "d2_smoke.py").read_text(encoding="utf-8")
        self.assertIn('gpu="L4", image=image, timeout=900, retries=0, max_containers=1', src)

    def test_no_torch_objects_cross_the_wire(self):
        # Regression for BTC-D2-001 attempt 1: torch.__version__ (TorchVersion) / torch exceptions
        # made the result undeserialisable locally and the results were lost.
        src = (ROOT / "src" / "d2_smoke.py").read_text(encoding="utf-8")
        self.assertNotIn('"torch": torch.__version__', src)
        self.assertIn('str(torch.__version__)', src)
        self.assertIn('json.dumps(payload["meta"])', src)
        self.assertIn("except BaseException as exc", src)


class Scorer(unittest.TestCase):
    def setUp(self):
        self.oracle = load_oracle()
        if HAS_MODAL:
            import d2_smoke
            self.ids = [r["world_id"] for r in d2_smoke.select_worlds()]
        else:
            self.skipTest("modal client not installed")

    def fake(self, policy):
        raw = []
        for rep in (0, 1):
            for w in self.ids:
                c = policy(self.oracle[w])
                raw.append({"world_id": w, "replay": rep, "n_tokens": 100,
                            "logp_A": -0.1 if c == "A" else -3.0, "logp_B": -3.0 if c == "A" else -0.1,
                            "top5": [[c, -0.1]]})
        return raw

    def test_perfect_model(self):
        m = score(self.fake(lambda o: o["oracle"]["optimal_action"]), self.oracle)["metrics"]
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["base_cf_pairs"], {"n": 3, "both_correct": 3})
        self.assertEqual(m["upath_pairs"]["n_constructed"], 4)
        self.assertEqual(m["upath_pairs"]["n_admissible_optimal_both"], 4)
        self.assertEqual(m["cpath_pairs"]["choice_invariant"], 4)
        self.assertEqual(m["replay_max_abs_logp_diff"], 0.0)

    def test_always_A(self):
        m = score(self.fake(lambda o: "A"), self.oracle)["metrics"]
        self.assertEqual(m["accuracy"], 0.5)
        self.assertEqual(m["base_cf_pairs"]["both_correct"], 0)
        self.assertEqual(m["swap_pairs"]["letter_flipped"], 0)
        self.assertEqual(m["upath_pairs"]["choice_invariant"], 4)


if __name__ == "__main__":
    unittest.main()
