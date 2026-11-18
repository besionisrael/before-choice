"""BTC-H1c: token-prefix identity of the binary candidates' arrival keys in late pairs (Gemma tokenizer; no model)."""

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_h1c as G  # noqa: E402
import h1c_token_checks as TC  # noqa: E402

GO = TC.gemma()


class H1cTokens(unittest.TestCase):
    def test_binary_keys_before_first_difference_and_identical_prefix(self):
        rows = G.build(12, 31337, 93000)
        fam = {}
        for w in rows:
            fam.setdefault(w["family_id"], {})[w["variant"]] = w
        for v in fam.values():
            a, b = v["base_L"], v["dswap_L"]
            non = "B" if a["oracle"]["optimal_action"] == "A" else "A"
            ta, ia, oa = GO(a["prompt"])
            tb, ib, ob = GO(b["prompt"])
            first = next(k for k, (x, y) in enumerate(zip(ia, ib)) if x != y)
            p = G.parse(a["prompt"])
            binary = sorted(t for s, t in p["mv"] if s == p["act"][non])
            pa = [TC.arrival_token(a["prompt"], ta, oa, x) for x in binary]
            pb = [TC.arrival_token(b["prompt"], tb, ob, x) for x in binary]
            self.assertEqual(pa, pb)
            self.assertTrue(all(q < first for q in pa))
            self.assertEqual(ia[:max(pa) + 1], ib[:max(pb) + 1])


if __name__ == "__main__":
    unittest.main()
