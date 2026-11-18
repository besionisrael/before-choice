"""Static checks of the (unexecuted) D3 Modal script. Torch-dependent tests live in tests_gemma/."""

import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SRC = (ROOT / "src" / "d3_run_gemma.py").read_text(encoding="utf-8")


@unittest.skipUnless(importlib.util.find_spec("modal"), "modal not installed")
class D3Run(unittest.TestCase):
    def test_imports_without_remote_call(self):
        import modal.cli.import_refs as ir
        mod = ir.import_file_or_module(ir.parse_import_ref(str(ROOT / "src" / "d3_run_gemma.py")))
        self.assertEqual(mod.app.name, "btc-d3-gemma4")
        self.assertEqual((mod.TIMEOUT_S, mod.GUARD_S, mod.SHARD, mod.N_REPLAY), (1500, 1420, 400, 20))
        self.assertEqual(list(mod.POOLS), ["PROBE-DEV"])  # LOCKED-TEST not supported before H1 freeze

    def test_limits_and_volume(self):
        self.assertIn("gpu=GPU, cpu=CPU, memory=MEMORY_MIB, retries=0, max_containers=1, timeout=TIMEOUT_S", SRC)
        self.assertIn("with modal.Volume.ephemeral() as vol:", SRC)      # auto-deleted, also on error
        self.assertNotIn("Volume.from_name", SRC)                       # no persistent named volume
        self.assertIn('capture.with_options(volumes={"/vol": vol})', SRC)
        self.assertIn("verify_shards(out)", SRC)
        self.assertNotIn("locked_test", SRC)

    def test_worst_case_cost_within_objective(self):
        from v2_common import worst_case_rate
        worst = (1500 + 60) * worst_case_rate() + 0.04 + 0.15  # + egress for ~3.3 GB
        self.assertLessEqual(worst, 0.85 * 2.00)


if __name__ == "__main__":
    unittest.main()


class D3RunTransferOrder(unittest.TestCase):
    def test_verify_before_volume_deletion_and_commit_wait(self):
        lines = SRC.splitlines()
        w = next(i for i, l in enumerate(lines) if "with modal.Volume.ephemeral() as vol:" in l)
        ind = len(lines[w]) - len(lines[w].lstrip())
        end = next(i for i in range(w + 1, len(lines)) if lines[i].strip() and (len(lines[i]) - len(lines[i].lstrip())) <= ind)
        block = "\n".join(lines[w:end])
        for needle in ("COMMIT_WAIT_S", "TRANSFER FAILURE", "verify_shards(out)", "INTEGRITY FAILURE", "d3_run_manifest.json"):
            self.assertIn(needle, block, needle)  # all inside the ephemeral-Volume block -> before deletion
        self.assertIn("scaledown_window=2", SRC)
        self.assertIn('"max_mem_gb"', SRC)
