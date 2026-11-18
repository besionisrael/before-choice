import json
import pathlib
import sys
import tempfile
import unittest
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from build_dataset import VARIANTS, build, write_jsonl  # noqa: E402
from generate_worlds import create_pair  # noqa: E402
from independent_check import audit_world  # noqa: E402

ROWS = build(10, 20261113)
BY = {r["world_id"]: r for r in ROWS}


def fam(fid, v):
    return BY[f"{fid}-{v}"]


class IndependentOracle(unittest.TestCase):
    def test_v1_all_worlds_pass_independent_audit(self):
        bad = {r["world_id"]: audit_world(r) for r in ROWS if audit_world(r)}
        self.assertEqual(bad, {})

    def test_pilot_v0_passes_independent_audit(self):
        for depth in (2, 3, 4):
            for seed in range(100):
                for w in create_pair(seed, seed + 7000, depth):
                    self.assertEqual(audit_world(w), [], w["world_id"])


class Invariants(unittest.TestCase):
    def setUp(self):
        self.fids = sorted({r["family_id"] for r in ROWS})

    def test_every_family_complete(self):
        for f in self.fids:
            self.assertEqual({r["variant"] for r in ROWS if r["family_id"] == f}, set(VARIANTS))

    def test_counterfactual_flips_optimal(self):
        for f in self.fids:
            for p in ("", "alias_", "swap_", "shuffle_", "decoy_"):
                b, c = fam(f, p + "base"), fam(f, p + "cf")
                self.assertNotEqual(b["oracle"]["optimal_action"], c["oracle"]["optimal_action"])
                # same label multiset, same rewards, same number of lines
                lab = lambda w: Counter(x for e in w["transitions"] for x in (e["source"], e["target"]))
                self.assertEqual(lab(b), lab(c))
                self.assertEqual(b["terminal_rewards"], c["terminal_rewards"])
                self.assertEqual(len(b["prompt"]), len(c["prompt"]))

    def test_cf_prompt_differs_in_exactly_two_lines(self):
        for f in self.fids:
            b, c = fam(f, "base")["prompt"].splitlines(), fam(f, "cf")["prompt"].splitlines()
            self.assertEqual(sum(x != y for x, y in zip(b, c)), 2)

    def test_swap_differs_only_in_action_letters(self):
        for f in self.fids:
            b, s = fam(f, "base"), fam(f, "swap_base")
            self.assertEqual(b["prompt"].replace("action A", "#").replace("action B", "action A").replace("#", "action B"), s["prompt"])
            self.assertNotEqual(b["oracle"]["optimal_action"], s["oracle"]["optimal_action"])

    def test_shuffle_same_graph_different_order(self):
        for f in self.fids:
            b, s = fam(f, "base"), fam(f, "shuffle_base")
            key = lambda w: sorted(json.dumps(e, sort_keys=True) for e in w["transitions"])
            self.assertEqual(key(b), key(s))
            self.assertNotEqual(b["prompt"], s["prompt"])
            self.assertEqual(b["oracle"], s["oracle"])

    def test_alias_isomorphic_disjoint_labels(self):
        for f in self.fids:
            b, a = fam(f, "base"), fam(f, "alias_base")
            names = lambda w: {x for e in w["transitions"] for x in (e["source"], e["target"])} - {"S"}
            self.assertFalse(names(b) & names(a))
            self.assertEqual(b["oracle"]["optimal_action"], a["oracle"]["optimal_action"])
            self.assertEqual(b["oracle"]["rewards"], a["oracle"]["rewards"])

    def test_decoy_unreachable_and_max_reward(self):
        for f in self.fids:
            d = fam(f, "decoy_base")
            chain = d["oracle"]["decoy_chain"]
            onpath = set(sum(d["oracle"]["paths"].values(), []))
            self.assertFalse(set(chain) & onpath)
            self.assertGreater(d["terminal_rewards"][chain[-1]], max(d["oracle"]["rewards"].values()))
            self.assertEqual(d["oracle"]["optimal_action"], fam(f, "base")["oracle"]["optimal_action"])


class Balance(unittest.TestCase):
    def test_depth_and_optimal_balance(self):
        prim = [r for r in ROWS if r["variant"] == "base"]
        self.assertEqual(Counter(r["depth"] for r in prim), {2: 10, 3: 10, 4: 10})
        for d in (2, 3, 4):
            c = Counter(r["oracle"]["optimal_action"] for r in prim if r["depth"] == d)
            self.assertEqual(c["A"], c["B"])
        # pooled base+cf is exactly 50/50 by construction
        c = Counter(r["oracle"]["optimal_action"] for r in ROWS if r["is_primary"])
        self.assertEqual(c["A"], c["B"])

    def test_trivial_heuristics_at_chance_on_primary(self):
        prim = [r for r in ROWS if r["is_primary"]]
        def acc(f):
            return sum(f(r) == r["oracle"]["optimal_action"] for r in prim) / len(prim)
        def first_action_line(r):
            for e in r["transitions"]:
                if "action" in e:
                    return e["action"]
        def first_reward_listed_high(r):  # pick action whose root label sorts first?? (label-position heuristic)
            roots = {e["action"]: e["target"] for e in r["transitions"] if "action" in e}
            return min(roots, key=roots.get)
        self.assertEqual(acc(lambda r: "A"), 0.5)
        # paired design: any heuristic blind to deep edges answers identically on base/cf -> exactly 0.5
        for h in (first_action_line, first_reward_listed_high):
            self.assertEqual(acc(h), 0.5)


class Splits(unittest.TestCase):
    def test_no_family_or_label_crosses_split(self):
        fam_split, label_split = {}, {}
        for r in ROWS:
            self.assertEqual(fam_split.setdefault(r["family_id"], r["split"]), r["split"])
            for e in r["transitions"]:
                for x in (e["source"], e["target"]):
                    if x != "S":
                        self.assertEqual(label_split.setdefault(x, r["split"]), r["split"], x)

    def test_split_sizes(self):
        c = Counter(r["split"] for r in ROWS if r["variant"] == "base")
        self.assertEqual(c, {"train": 18, "val": 6, "test": 6})

    def test_deterministic(self):
        with tempfile.TemporaryDirectory() as t:
            h1 = write_jsonl(build(10, 20261113), pathlib.Path(t) / "a.jsonl")
            h2 = write_jsonl(build(10, 20261113), pathlib.Path(t) / "b.jsonl")
            self.assertEqual(h1, h2)
            self.assertNotEqual(h1, write_jsonl(build(10, 1), pathlib.Path(t) / "c.jsonl"))

    def test_committed_files_match_generator(self):
        p = ROOT / "data" / "generated" / "pilot_v1.jsonl"
        if not p.exists():
            self.skipTest("dataset not built")
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows, json.loads(json.dumps(ROWS)))

    def test_model_inputs_have_no_oracle(self):
        p = ROOT / "data" / "generated" / "pilot_v1_model_inputs.jsonl"
        if not p.exists():
            self.skipTest("dataset not built")
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            self.assertNotIn("oracle", r)
            self.assertNotIn("terminal_rewards", r)


if __name__ == "__main__":
    unittest.main()
