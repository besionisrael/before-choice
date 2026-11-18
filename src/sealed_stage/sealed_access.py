"""Stage-scoped access to sealed pools (BTC-D3-TEST preparation). Built ON TOP of the frozen `sealed.sealed_read`
(which is unchanged); this layer adds the separation the owner required between

  * D3-TEST-CAPTURE : controlled access to the MODEL-VISIBLE prompts only (`model_inputs.jsonl`, oracle-free), to
                      run the forward passes. No world file, no oracle, no label, no analysis.
  * H1-TEST-EVAL    : the analytical unlock (oracle `worlds.jsonl` + the sealed TEST activations), for the single
                      pre-registered evaluation.

Each stage has its OWN unlock directory `outputs/approvals/unlock/<STAGE>/UNLOCK_<seal_id>.json`, written by the
owner, with fields owner_approval_id, approved_utc, scope (== STAGE) and files (list). A read is refused unless
  (1) the unlock file of THAT stage exists for THAT seal, with scope == stage,
  (2) the requested file is allowed for the stage by this module AND listed in the unlock file,
then `sealed.sealed_read` re-checks the sha256 against the seal manifest and appends to the access log.
This directory (src/sealed_stage/) is the only code besides build_d5.py / sealed.py allowed to name the sealed
paths; development modules in src/ remain scanned (tests/test_d5_dev.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import sealed  # noqa: E402

POOL_DIR = ROOT / "data" / "d5" / "locked_test"
POOL_SEAL = "LOCKED-TEST-D5"
ACT_DIR = ROOT / "outputs" / "d3" / "locked_test"
ACT_SEAL = "LOCKED-TEST-D3-ACT"
UNLOCK_ROOT = ROOT / "outputs" / "approvals" / "unlock"

STAGES = {
    "D3-TEST-CAPTURE": {POOL_SEAL: {"model_inputs.jsonl"}},
    "H1-TEST-EVAL": {POOL_SEAL: {"worlds.jsonl", "model_inputs.jsonl"},
                     ACT_SEAL: None},  # None = every file of the activation seal
}


def unlock_dir(stage: str, root: Path = UNLOCK_ROOT) -> Path:
    return root / stage


def check_unlock(stage: str, seal_id: str, rel: str, root: Path = UNLOCK_ROOT) -> dict:
    if stage not in STAGES or seal_id not in STAGES[stage]:
        raise PermissionError(f"stage {stage} has no access to seal {seal_id}")
    allowed = STAGES[stage][seal_id]
    if allowed is not None and rel not in allowed:
        raise PermissionError(f"stage {stage} may not read {rel} of {seal_id}")
    u = sealed.unlock_path(unlock_dir(stage, root), seal_id)
    if not u.exists():
        raise PermissionError(f"no owner unlock for stage {stage}, seal {seal_id} ({u})")
    rec = json.loads(u.read_text(encoding="utf-8"))
    for k in ("owner_approval_id", "approved_utc", "scope", "files"):
        if k not in rec:
            raise PermissionError(f"unlock file missing field {k}")
    if rec["scope"] != stage:
        raise PermissionError(f"unlock scope {rec['scope']!r} != stage {stage!r}")
    if rec["files"] != "*" and rel not in rec["files"]:
        raise PermissionError(f"{rel} not listed in the owner unlock for {stage}")
    return rec


def staged_read(stage: str, d: Path, rel: str, purpose: str, root: Path = UNLOCK_ROOT) -> bytes:
    man = json.loads((d / sealed.MANIFEST).read_text(encoding="utf-8"))
    check_unlock(stage, man["seal_id"], rel, root)
    return sealed.sealed_read(d, rel, unlock_dir(stage, root), f"{stage}: {purpose}")


def reads_outside_stage(d: Path, stage: str) -> list[dict]:
    """Access-log entries of seal directory `d` that the given stage would NOT have been allowed to make."""
    man = json.loads((d / sealed.MANIFEST).read_text(encoding="utf-8"))
    allowed = STAGES.get(stage, {}).get(man["seal_id"], set())
    return [r for r in access_log(d) if allowed is not None and r["file"] not in allowed]


def access_log(d: Path) -> list[dict]:
    p = d / sealed.ACCESS_LOG
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []
