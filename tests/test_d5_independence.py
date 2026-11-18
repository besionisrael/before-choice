"""PROBE-DEV vs LOCKED-TEST independence. The sealed files are only HASHED (sealed.hash_only); content
comparisons use an in-memory regeneration whose bytes are proved identical to the sealed file.
Only booleans / overlap COUNTS are asserted; no test content is printed or analysed."""

import hashlib
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from build_d5 import POOLS, build, labels_of  # noqa: E402
import sealed  # noqa: E402

TEST_DIR = POOLS["LOCKED-TEST"]["out"]
MAN = json.loads((TEST_DIR / sealed.MANIFEST).read_text(encoding="utf-8"))


def jsonl_bytes(rows):
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode("utf-8")


class Independence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev = build("PROBE-DEV")
        cls.test = build("LOCKED-TEST")  # in memory only

    def test_regeneration_equals_sealed_bytes(self):
        self.assertEqual(hashlib.sha256(jsonl_bytes(self.test)).hexdigest(), MAN["files"]["worlds.jsonl"])
        self.assertTrue(all(sealed.hash_only(TEST_DIR).values()))

    def test_no_overlap(self):
        d, t = self.dev, self.test
        overlaps = {
            "family_ids": len({r["family_id"] for r in d} & {r["family_id"] for r in t}),
            "world_ids": len({r["world_id"] for r in d} & {r["world_id"] for r in t}),
            "labels": len(labels_of(d) & labels_of(t)),
            "prompt_sha256": len({r["prompt_sha256"] for r in d} & {r["prompt_sha256"] for r in t}),
        }
        self.assertEqual(overlaps, {"family_ids": 0, "world_ids": 0, "labels": 0, "prompt_sha256": 0})

    def test_test_pool_shape(self):
        t = self.test
        self.assertEqual(len(t), 4800)
        self.assertEqual(len({r["family_id"] for r in t}), 300)
        ids = sorted(int(r["family_id"][1:]) for r in t)
        self.assertEqual((ids[0], ids[-1]), (30000, 30299))
        self.assertTrue(all(r["split"] == "locked_test" for r in t))


if __name__ == "__main__":
    unittest.main()
