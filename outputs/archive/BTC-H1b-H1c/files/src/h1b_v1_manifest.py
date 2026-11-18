"""BTC-H1b v1.0: write the freeze manifest (hashes of code, frozen decoders, statistics, power analyses, protocol).

    python src/h1b_v1_manifest.py   -> outputs/h1b_freeze/V1_MANIFEST.json (written once; files made read-only)
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ["src/h1b_freeze_dev.py", "src/h1b_x3x4_analysis.py", "src/h1b_explore.py", "src/d3x_positions.py", "src/d3x_capture_core.py",
        "src/d3_capture_core.py", "src/h1_probe.py", "src/h1_features.py", "src/sealed.py", "src/v2_common.py", "src/v2_local.py",
        "src/build_dataset.py", "src/build_path_cf.py", "src/build_d5.py", "oracle.py",
        "src/sealed_stage/sealed_access_h1b.py", "src/sealed_stage/build_h1b_pool.py", "src/sealed_stage/h1b_capture_test.py",
        "src/sealed_stage/h1b_eval_test.py", "src/power_h1b.py", "src/power_h1b_joint.py", "src/power_h1b_cmax.py",
        "tests_cpucheck/test_h1b_v1.py", "tests/test_h1b_capture_run.py"]
ART = ["outputs/h1b_freeze/FREEZE.json", "outputs/h1b_freeze/STANDARDISATION.npz", "outputs/metrics/power_h1b.json",
       "outputs/metrics/power_h1b_joint.json", "outputs/metrics/power_h1b_cmax.json", "outputs/h1b_control_c/RESULT.json",
       "outputs/d3x/probe_dev/MANIFEST.json", "outputs/d3x/plan/positions.jsonl", "reports/BTC-H1b_CONFIRMATORY_PROTOCOL.md",
       "outputs/logs/h1b_v1_tests_cpucheck.log", "outputs/logs/h1b_v1_tests_main.log", "outputs/logs/h1b_v1_tests_gemma.log",
       "outputs/logs/h1b_v1_tests_post_eval.log"]


def main() -> None:
    out = ROOT / "outputs" / "h1b_freeze" / "V1_MANIFEST.json"
    if out.exists():
        raise SystemExit("V1_MANIFEST.json exists: written once")
    sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
    frz = json.loads((ROOT / "outputs" / "h1b_freeze" / "FREEZE.json").read_text(encoding="utf-8"))
    dec_dir = ROOT / "outputs" / "h1b_freeze" / "decoders"
    decs = {p.stem: sha(p.relative_to(ROOT)) for p in sorted(dec_dir.glob("*.pt"))}
    assert decs == frz["decoders_sha256"], "decoder files differ from FREEZE.json"
    used = ["P_u2_L40", "C_bil_u2_L40", "C_lin_u2_L40", "K_u2_L40", "B3plus_u2", "P_c1_L40", "P_c2_L40", "P_u2_L32", "P_u2_L48"]
    man = {"protocol": "BTC-H1b v1.0", "utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
           "status": "FROZEN ARTIFACTS PREPARED; LOCKED-TEST-H1b NOT GENERATED; NO CAPTURE; NO EVALUATION",
           "code_sha256": {p: sha(p) for p in CODE},
           "artifact_sha256": {**{p: sha(p) for p in ART}, **{f"outputs/h1b_freeze/decoders/{n}.pt": decs[n] for n in used}},
           "decoders": used, "decoders_reported_not_used": sorted(set(decs) - set(used)),
           "seeds": {"decoders": [0, 1, 2], "bootstrap": 0, "permutation": 2026111502, "pool": 2026111403, "inner_split": 2026111501},
           "decision_constants": {"min_contributing_families": 875, "c1_sensitivity_min": 0.80, "bootstrap_resamples": 10000,
                                  "alpha_one_sided": 0.025, "k_perm": 50, "primary_layer": 40, "query": "ANS", "keys": "arrival"}}
    out.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    for p in list(dec_dir.glob("*.pt")) + [ROOT / "outputs" / "h1b_freeze" / "FREEZE.json", ROOT / "outputs" / "h1b_freeze" / "STANDARDISATION.npz", out]:
        os.chmod(p, stat.S_IREAD)
    print(json.dumps({"n_code": len(man["code_sha256"]), "n_artifacts": len(man["artifact_sha256"]), "manifest_sha256": sha(out.relative_to(ROOT))}, indent=1))


if __name__ == "__main__":
    main()
