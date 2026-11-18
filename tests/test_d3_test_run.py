"""Static checks of the (unexecuted) BTC-D3-TEST Modal capture script: same limits and transfer order as the approved
D3-DEV-R1 script, prompts-only stage, immediate sealing. Nothing is launched; the sealed pool is not opened."""

import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
P = ROOT / "src" / "sealed_stage" / "d3_run_gemma_test.py"
SRC = P.read_text(encoding="utf-8")
DEV_SRC = (ROOT / "src" / "d3_run_gemma.py").read_text(encoding="utf-8")


@unittest.skipUnless(importlib.util.find_spec("modal"), "modal not installed")
class D3TestRun(unittest.TestCase):
    def test_imports_without_remote_call(self):
        import modal.cli.import_refs as ir
        mod = ir.import_file_or_module(ir.parse_import_ref(str(P)))
        self.assertEqual(mod.app.name, "btc-d3-test-gemma4")
        self.assertEqual((mod.TIMEOUT_S, mod.GUARD_S, mod.SHARD, mod.N_REPLAY, mod.COMMIT_WAIT_S, mod.N_WORLDS),
                         (1500, 1420, 400, 20, 300, 4800))
        self.assertEqual(mod.STAGE, "D3-TEST-CAPTURE")

    def test_imports_as_mounted_in_the_container(self):
        """Regression for attempt 1: Modal mounts the script as /root/<name>.py (no project parents). Importing a copy
        placed outside the project must not raise, and ROOT must then be None."""
        import shutil
        import tempfile
        import modal.cli.import_refs as ir
        with tempfile.TemporaryDirectory() as t:
            q = pathlib.Path(t) / "d3_run_gemma_test_copy.py"
            shutil.copy(P, q)
            mod = ir.import_file_or_module(ir.parse_import_ref(str(q)))
            self.assertIsNone(mod.ROOT)
            self.assertEqual(mod.app.name, "btc-d3-test-gemma4")

    def test_same_function_limits_and_image_as_dev(self):
        for needle in ("gpu=GPU, cpu=CPU, memory=MEMORY_MIB, retries=0, max_containers=1, timeout=TIMEOUT_S",
                       "scaledown_window=2", '"max_mem_gb"', "with modal.Volume.ephemeral() as vol:",
                       'capture.with_options(volumes={"/vol": vol})', "verify_shards(out)",
                       '.add_local_python_source("v2_common", "d3_capture_core")',
                       "run_capture(rows, labels, digest, Path(\"/vol\") / \"shards\", \"cuda\", model, tok, GUARD_S - t_load, SHARD, N_REPLAY)"):
            self.assertIn(needle, SRC, needle)
            self.assertIn(needle, DEV_SRC, needle)
        pin = DEV_SRC[DEV_SRC.index(".pip_install("):DEV_SRC.index(".add_local_python_source")]
        self.assertIn(pin, SRC)  # identical pinned image
        self.assertNotIn("Volume.from_name", SRC)

    def test_verify_inside_volume_block_then_seal(self):
        lines = SRC.splitlines()
        w = next(i for i, l in enumerate(lines) if "with modal.Volume.ephemeral() as vol:" in l)
        ind = len(lines[w]) - len(lines[w].lstrip())
        end = next(i for i in range(w + 1, len(lines)) if lines[i].strip() and (len(lines[i]) - len(lines[i].lstrip())) <= ind)
        block = "\n".join(lines[w:end])
        for needle in ("COMMIT_WAIT_S", "TRANSFER FAILURE", "verify_shards(out)", "INTEGRITY FAILURE", "world_ids_complete"):
            self.assertIn(needle, block, needle)
        self.assertIn("sealed.seal_dir(out, SA.ACT_SEAL)", "\n".join(lines[end:]))

    def test_prompts_only(self):
        self.assertNotIn("worlds.jsonl", SRC)
        self.assertIn('SA.staged_read(STAGE, SA.POOL_DIR, "model_inputs.jsonl"', SRC)
        self.assertIn("preflight()", SRC)


if __name__ == "__main__":
    unittest.main()
