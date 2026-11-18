"""Choice-preserving path counterfactuals + formal shortcut checks for D-1 revision."""

import json
import pathlib
import sys
import unittest
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from build_dataset import build as build_v1  # noqa: E402
from build_path_cf import build  # noqa: E402
from independent_check import audit_world  # noqa: E402

V1 = build_v1(10, 20261113)
SRC = {r["world_id"]: r for r in V1}
NEW = build(V1)


def non_opt(w):
    return "B" if w["oracle"]["optimal_action"] == "A" else "A"


def labels(w):
    return Counter(x for e in w["transitions"] for x in (e["source"], e["target"]))


def pair_invariant_features(w):
    """Everything an answer/outcome decoder could know: answer, rewards, label inventory, length."""
    return (w["oracle"]["optimal_action"], tuple(sorted(w["terminal_rewards"].items())),
            tuple(sorted(labels(w).items())), len(w["prompt"]))


class ShortcutDocumentation(unittest.TestCase):
    def test_nonoptimal_terminal_is_lowest_visible_reward_everywhere(self):
        # The original D-1 target is readable from the prompt alone: argmin over visible rewards.
        for w in V1:
            r = w["terminal_rewards"]
            self.assertEqual(w["oracle"]["paths"][non_opt(w)][-1], min(r, key=r.get), w["world_id"])

    def test_depth1_state_is_one_hop_lexical(self):
        # depth-1 state of each action is printed in a single rule line "At S, action X leads to Y."
        for w in V1:
            for a in ("A", "B"):
                self.assertIn(f"At S, action {a} leads to {w['oracle']['paths'][a][1]}.", w["prompt"])


class PathCF(unittest.TestCase):
    def test_counts(self):
        c = Counter((w["variant"].split("_")[0], w["depth"]) for w in NEW)
        self.assertEqual(c, {("xswap", 2): 20, ("xswap", 3): 20, ("xswap", 4): 20,
                             ("upath", 3): 20, ("upath", 4): 20, ("cpath", 3): 20, ("cpath", 4): 20})

    def test_independent_audit(self):
        self.assertEqual({w["world_id"]: audit_world(w) for w in NEW if audit_world(w)}, {})

    def test_choice_and_outcome_preserved(self):
        for w in NEW:
            s = SRC[w["source_world_id"]]
            self.assertEqual(pair_invariant_features(w), pair_invariant_features(s))
            self.assertEqual(w["oracle"]["rewards"], s["oracle"]["rewards"])
            self.assertEqual(w["split"], s["split"])
            self.assertEqual(len(w["prompt"].splitlines()), len(s["prompt"].splitlines()))
            for a in ("A", "B"):  # terminals and depth-0 unchanged
                self.assertEqual(w["oracle"]["paths"][a][-1], s["oracle"]["paths"][a][-1])

    def test_only_targeted_branch_changes(self):
        for w in NEW:
            s = SRC[w["source_world_id"]]
            kind = w["variant"].split("_")[0]
            opt, non = s["oracle"]["optimal_action"], non_opt(s)
            same = {"upath": opt, "cpath": non}.get(kind)
            diff = {"upath": non, "cpath": opt}.get(kind)
            if same:
                self.assertEqual(w["oracle"]["paths"][same], s["oracle"]["paths"][same])
                self.assertNotEqual(w["oracle"]["paths"][diff], s["oracle"]["paths"][diff])
                self.assertEqual(set(w["oracle"]["paths"][diff]), set(s["oracle"]["paths"][diff]))
            else:  # xswap: membership changes in both branches
                for a in ("A", "B"):
                    self.assertNotEqual(set(w["oracle"]["paths"][a]), set(s["oracle"]["paths"][a]))

    def test_primary_target_changes_in_every_upath_pair(self):
        for w in NEW:
            if w["variant"].startswith("upath"):
                s = SRC[w["source_world_id"]]
                self.assertNotEqual(w["oracle"]["paths"][non_opt(w)][2], s["oracle"]["paths"][non_opt(s)][2])

    def test_depth4_upath_root_line_unchanged(self):
        for w in NEW:
            if w["variant"].startswith("upath") and w["depth"] == 4:
                s = SRC[w["source_world_id"]]
                for a in ("A", "B"):
                    self.assertEqual(w["oracle"]["paths"][a][1], s["oracle"]["paths"][a][1])

    def test_pair_invariant_decoder_has_zero_paired_accuracy(self):
        """Any decoder that sees only answer/rewards/label inventory/length outputs the same
        label for both pair members; the target differs, so it is never right on both."""
        for w in NEW:
            if not w["variant"].startswith("upath"):
                continue
            s = SRC[w["source_world_id"]]
            cands = sorted(set(s["oracle"]["paths"]["A"][1:-1] + s["oracle"]["paths"]["B"][1:-1]))
            t_src, t_new = s["oracle"]["paths"][non_opt(s)][2], w["oracle"]["paths"][non_opt(w)][2]
            for guess in cands:
                self.assertFalse(guess == t_src and guess == t_new)

    def test_committed_file_matches(self):
        p = ROOT / "data" / "generated" / "pilot_v1_pathcf.jsonl"
        if not p.exists():
            self.skipTest("not built")
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows, json.loads(json.dumps(NEW)))


if __name__ == "__main__":
    unittest.main()
