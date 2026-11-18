"""Stage-scoped sealed access for LOCKED-TEST-H1b (same rules as sealed_access.py, new seals and stages).

  H1B-CAPTURE : the model-visible prompts only (`model_inputs.jsonl`); oracle refused
  H1B-EVAL    : oracle `worlds.jsonl`, prompts, and every file of the sealed H1b activations (single evaluation)
Unlock files: outputs/approvals/unlock/<STAGE>/UNLOCK_<seal_id>.json written on the owner's authorisation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import sealed  # noqa: E402

POOL_DIR = ROOT / "data" / "d5" / "locked_test_h1b"
POOL_SEAL = "LOCKED-TEST-H1B"
ACT_DIR = ROOT / "outputs" / "d3" / "locked_test_h1b"
ACT_SEAL = "LOCKED-TEST-H1B-ACT"
UNLOCK_ROOT = ROOT / "outputs" / "approvals" / "unlock"
STAGES = {"H1B-CAPTURE": {POOL_SEAL: {"model_inputs.jsonl"}},
          "H1B-EVAL": {POOL_SEAL: {"worlds.jsonl", "model_inputs.jsonl"}, ACT_SEAL: None}}


def unlock_dir(stage, root=UNLOCK_ROOT):
    return root / stage


def check_unlock(stage, seal_id, rel, root=UNLOCK_ROOT):
    if stage not in STAGES or seal_id not in STAGES[stage]:
        raise PermissionError(f"stage {stage} has no access to seal {seal_id}")
    allowed = STAGES[stage][seal_id]
    if allowed is not None and rel not in allowed:
        raise PermissionError(f"stage {stage} may not read {rel} of {seal_id}")
    u = sealed.unlock_path(unlock_dir(stage, root), seal_id)
    if not u.exists():
        raise PermissionError(f"no owner unlock for stage {stage}, seal {seal_id}")
    rec = json.loads(u.read_text(encoding="utf-8"))
    for k in ("owner_approval_id", "approved_utc", "scope", "files"):
        if k not in rec:
            raise PermissionError(f"unlock file missing field {k}")
    if rec["scope"] != stage:
        raise PermissionError(f"unlock scope {rec['scope']!r} != stage {stage!r}")
    if rec["files"] != "*" and rel not in rec["files"]:
        raise PermissionError(f"{rel} not listed in the owner unlock for {stage}")
    return rec


def staged_read(stage, d, rel, purpose, root=UNLOCK_ROOT) -> bytes:
    man = json.loads((d / sealed.MANIFEST).read_text(encoding="utf-8"))
    check_unlock(stage, man["seal_id"], rel, root)
    return sealed.sealed_read(d, rel, unlock_dir(stage, root), f"{stage}: {purpose}")


def access_log(d: Path) -> list[dict]:
    p = d / sealed.ACCESS_LOG
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


def reads_outside_stage(d: Path, stage: str) -> list[dict]:
    man = json.loads((d / sealed.MANIFEST).read_text(encoding="utf-8"))
    allowed = STAGES.get(stage, {}).get(man["seal_id"], set())
    return [r for r in access_log(d) if allowed is not None and r["file"] not in allowed]
