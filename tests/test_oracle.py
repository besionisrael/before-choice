import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_worlds import create_pair
from oracle import solve


class GraphOracleTests(unittest.TestCase):
    def test_counterfactual_pairs(self):
        for depth in (2, 3, 4):
            for seed in range(40):
                base, counter = create_pair(seed, seed + 2000, depth)
                self.assertEqual(set(base["terminal_rewards"]), set(counter["terminal_rewards"]))
                self.assertEqual(base["terminal_rewards"], counter["terminal_rewards"])
                self.assertNotEqual(base["oracle"]["optimal_action"], counter["oracle"]["optimal_action"])
                self.assertEqual(len(base["oracle"]["paths"]["A"]) - 1, depth)
                self.assertEqual(len(base["oracle"]["paths"]["B"]) - 1, depth)
                self.assertEqual(solve(base), base["oracle"])
                self.assertEqual(solve(counter), counter["oracle"])
                for w in (base, counter):
                    self.assertNotIn("optimal_action", w["prompt"])
                    self.assertNotIn("oracle", w["prompt"])
                    self.assertNotIn("paths", w["prompt"])

    def test_command_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "data.jsonl"
            proc = subprocess.run([
                sys.executable, str(ROOT / "generate_worlds.py"),
                "--pairs", "7", "--seed", "101", "--output", str(out),
            ], capture_output=True, text=True, check=True)
            self.assertIn("14 worlds", proc.stdout)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 14)
            self.assertEqual(len({x["pair_id"] for x in rows}), 7)


if __name__ == "__main__":
    unittest.main()
