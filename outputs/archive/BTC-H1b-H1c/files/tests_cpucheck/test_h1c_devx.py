"""BTC-H1c DEV extension: frozen splits and analysis pipeline (synthetic captures; no model)."""

import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import h1c_dev_analysis as A  # noqa: E402
import h1c_devx_analysis as X  # noqa: E402


class Splits(unittest.TestCase):
    def test_frozen_roles(self):
        roles = json.loads((ROOT / "outputs" / "h1c_devx" / "plan" / "SPLITS.json").read_text(encoding="utf-8"))["roles"]
        old = {json.loads(x)["family_id"] for x in (ROOT / "data" / "h1c" / "dev" / "worlds.jsonl").read_text(encoding="utf-8").splitlines()}
        new = {json.loads(x)["family_id"] for x in (ROOT / "data" / "h1c" / "devx" / "worlds.jsonl").read_text(encoding="utf-8").splitlines()}
        self.assertFalse(old & new)
        self.assertTrue(all(roles[f] == "TRAIN" for f in old))                 # analysed families never select / evaluate
        self.assertEqual(sum(roles[f] == "EVAL" for f in new), 200)
        self.assertEqual(sum(roles[f] == "SELECT" for f in new), 100)
        self.assertEqual(sum(roles[f] == "TRAIN" for f in new), 300)


class Pipeline(unittest.TestCase):
    def test_runs_and_decision_follows_criteria(self):
        with tempfile.TemporaryDirectory() as t:
            b = pathlib.Path(t)
            c, d = A.make_synthetic(b / "p", n_fam=90, seed=3)
            fams = sorted({json.loads(x)["family_id"] for x in (d / "worlds.jsonl").read_text(encoding="utf-8").splitlines()})
            roles = {f: ("TRAIN" if i % 3 else "EVAL" if i % 6 == 0 else "SELECT") for i, f in enumerate(fams)}
            (b / "S.json").write_text(json.dumps({"roles": roles}), encoding="utf-8")
            cap = X.CapX([(c, d)], b / "S.json")
            f = lambda ids: {cap.rows[k]["family_id"] for k in ids}
            self.assertFalse(f(cap.itr) & f(cap.iva) or f(cap.itr) & f(cap.va) or f(cap.iva) & f(cap.va))
            r = X.run(cap, b / "out", seeds=(0,), max_epochs=20)
            self.assertIn(r["selected_layer"], X.LAYERS)
            crit = r["stop_criteria"]["results"]
            self.assertEqual(r["stop_criteria"]["decision"].startswith("INSTRUMENT ADEQUATE"), all(crit.values()))
            self.assertIn("robust_late", r["P_unchosen_EVAL"])
            with self.assertRaises(SystemExit):
                X.run(cap, b / "out", seeds=(0,), max_epochs=2)


if __name__ == "__main__":
    unittest.main()
