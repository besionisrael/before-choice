"""Sealing utilities for the LOCKED-TEST pool and its activations (BTC-D5 §4, strengthened).

Rules enforced in code:
  * the only sanctioned way to READ anything under a sealed directory is `sealed_read`, which
    (1) refuses unless an owner unlock file exists for that seal, (2) verifies the sha256 of the file
    against the seal manifest, and (3) appends an access record to the seal's access log;
  * `hash_only` lets development code check integrity (sha256) without exposing content;
  * `seal_dir` writes the manifest (sha256 of every file) and marks every file read-only.
No development module may reference the sealed paths directly (scanned by tests/test_d5_dev.py).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import stat
from pathlib import Path

MANIFEST = "SEAL_MANIFEST.json"
ACCESS_LOG = "SEAL_ACCESS_LOG.jsonl"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def seal_dir(d: Path, seal_id: str) -> dict:
    files = sorted(p for p in d.rglob("*") if p.is_file() and p.name not in (MANIFEST, ACCESS_LOG))
    man = {"seal_id": seal_id, "sealed_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
           "files": {str(p.relative_to(d)).replace("\\", "/"): _sha(p) for p in files}}
    (d / MANIFEST).write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    for p in files + [d / MANIFEST]:
        os.chmod(p, stat.S_IREAD)
    return man


def hash_only(d: Path) -> dict[str, bool]:
    """Integrity check without exposing content: recompute sha256 against the manifest."""
    man = json.loads((d / MANIFEST).read_text(encoding="utf-8"))
    return {rel: _sha(d / rel) == h for rel, h in man["files"].items()}


def unlock_path(unlock_dir: Path, seal_id: str) -> Path:
    return unlock_dir / f"UNLOCK_{seal_id}.json"


def sealed_read(d: Path, rel: str, unlock_dir: Path, purpose: str) -> bytes:
    man = json.loads((d / MANIFEST).read_text(encoding="utf-8"))
    u = unlock_path(unlock_dir, man["seal_id"])
    if not u.exists():
        raise PermissionError(f"sealed pool {man['seal_id']}: no owner unlock file ({u.name})")
    unlock = json.loads(u.read_text(encoding="utf-8"))
    for k in ("owner_approval_id", "approved_utc", "scope"):
        if k not in unlock:
            raise PermissionError(f"unlock file missing field {k}")
    data = (d / rel).read_bytes()
    if hashlib.sha256(data).hexdigest() != man["files"][rel]:
        raise ValueError(f"seal violated: hash mismatch for {rel}")
    with open(d / ACCESS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
                            "file": rel, "purpose": purpose, "approval": unlock["owner_approval_id"]}) + "\n")
    return data
