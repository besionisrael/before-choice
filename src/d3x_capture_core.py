"""BTC-H1b X3+X4 capture core: hidden states at several token positions x a reduced layer grid, in ONE forward pass.

Re-uses the FROZEN D3 fidelity machinery unchanged (d3_capture_core.check_fidelity, _text_parts, _logits_from):
the pre-answer logits are reconstructed from the final hidden state (post-norm) and from the raw block-60 output
(pre-norm) exactly as in D3-DEV-R1 / D3-TEST-R1 (max|d| <= 0.25 over the full vocabulary, A/B decision preserved,
original-tie rule, top-1 within 2d + tau), non-finite values are rejected, and the first N worlds are replayed and
must be bitwise identical.

Per world: acts (P=22, L=9, d) bf16 stored as uint16, in the fixed slot order of d3x_positions (masked slots = 0),
plus meta (n_tokens, ids sha256, logp_A/B, fidelity, mask). Layer convention: 0..60 = HF hidden_states index,
61 = raw block-60 output (pre final norm), for EVERY captured position.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from d3_capture_core import FIDELITY_MAX_ABS, TAU, FidelityError, _logits_from, _text_parts, check_fidelity


def capture_world_multi(model, tok, text: str, tok_pos: list[int], layers: tuple[int, ...], a_id: int, b_id: int,
                        device: str = "cuda", world_id: str = "?", ids_sha256: str | None = None) -> dict:
    import torch
    lm, blocks, norm_mod, head, cap = _text_parts(model)
    n_hf = len(blocks) + 1
    assert all(0 <= l < n_hf or l == n_hf for l in layers), "layer index out of range"
    raw = {}
    h = blocks[-1].register_forward_hook(lambda m, i, o: raw.__setitem__("x", (o[0] if isinstance(o, tuple) else o)[0].detach()))
    try:
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        if ids_sha256 is not None:
            got = hashlib.sha256(",".join(map(str, ids[0].tolist())).encode()).hexdigest()
            assert got == ids_sha256, f"[{world_id}] tokenisation differs from the locally computed positions"
        n = ids.shape[1]
        assert all(t < n for t in tok_pos) and tok_pos[0] == n - 1, f"[{world_id}] positions out of range"
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
            hs = out.hidden_states
            logits = out.logits[0, -1].float()
            l_post = _logits_from(hs[-1][0, -1], norm_mod, head, cap, apply_norm=False)
            l_pre = _logits_from(raw["x"][-1], norm_mod, head, cap, apply_norm=True)
            P, L, d = len(tok_pos), len(layers), hs[0].shape[-1]
            acts = torch.zeros(P, L, d, dtype=hs[0].dtype, device=hs[0].device)
            valid = [k for k, t in enumerate(tok_pos) if t >= 0]
            idx = torch.tensor([tok_pos[k] for k in valid], device=hs[0].device)
            for j, l in enumerate(layers):
                src = raw["x"] if l == n_hf else hs[l][0]
                acts[valid, j] = src[idx]
    finally:
        h.remove()
    lp = torch.log_softmax(logits, -1)
    fid = {name: check_fidelity(logits, l2, a_id, b_id, world_id, name) for name, l2 in (("post_norm", l_post), ("pre_norm", l_pre))}
    if not bool(torch.isfinite(acts).all()):
        raise FidelityError(f"capture fidelity failed [{world_id}]: non-finite activations")
    a16 = acts.to(torch.bfloat16).cpu().view(torch.int16).numpy().view(np.uint16)
    return {"acts_u16": a16, "meta": {"n_tokens": int(n), "logp_A": float(lp[a_id]), "logp_B": float(lp[b_id]),
                                      "fidelity": fid, "mask": [t >= 0 for t in tok_pos]}}


def _npy(a: np.ndarray) -> bytes:
    b = io.BytesIO()
    np.save(b, a, allow_pickle=False)
    return b.getvalue()


class ShardWriterX:
    def __init__(self, root: Path, shard_size: int = 400):
        self.root, self.size, self.buf, self.k, self.manifest = Path(root), shard_size, [], 0, {"shards": []}
        self.root.mkdir(parents=True, exist_ok=True)

    def add(self, world_id: str, rec: dict) -> None:
        self.buf.append((world_id, rec))
        if len(self.buf) >= self.size:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        name = f"xshard_{self.k:03d}"
        files = {f"{name}_acts.npy": _npy(np.stack([r["acts_u16"] for _, r in self.buf])),
                 f"{name}_index.jsonl": "".join(json.dumps({"world_id": w, **r["meta"]}) + "\n" for w, r in self.buf).encode()}
        entry = {"name": name, "n": len(self.buf), "files": {}}
        for fn, data in files.items():
            (self.root / fn).write_bytes(data)
            entry["files"][fn] = hashlib.sha256(data).hexdigest()
        self.manifest["shards"].append(entry)
        self.buf, self.k = [], self.k + 1

    def close(self, extra: dict) -> dict:
        self.flush()
        self.manifest.update(extra)
        (self.root / "MANIFEST.json").write_text(json.dumps(self.manifest, indent=1), encoding="utf-8")
        return self.manifest


def verify_xshards(root: Path, P: int, L: int) -> dict:
    man = json.loads((Path(root) / "MANIFEST.json").read_text(encoding="utf-8"))
    ok, n = True, 0
    for s in man["shards"]:
        for fn, h in s["files"].items():
            ok &= hashlib.sha256((Path(root) / fn).read_bytes()).hexdigest() == h
        a = np.load(Path(root) / f"{s['name']}_acts.npy", mmap_mode="r")
        ok &= a.shape[0] == s["n"] and a.shape[1] == P and a.shape[2] == L and a.dtype == np.uint16
        n += s["n"]
    return {"hashes_and_shapes_ok": bool(ok), "n_worlds": n}


def run_capture_multi(rows, layers, digest, out_dir: Path, device, model, tok, guard_s: float, shard: int, n_replay: int) -> dict:
    """rows: {world_id, prompt, prompt_sha256, expected_n_tokens, tok, ids_sha256}. Oracle-free payload."""
    import time
    from v2_common import MODELS, render
    m = MODELS["gemma4_31b"]
    t0 = time.time()
    assert hashlib.sha256("\n".join(r["prompt_sha256"] for r in rows).encode()).hexdigest() == digest, "digest mismatch"
    for r in rows:
        assert set(r) == {"world_id", "prompt", "prompt_sha256", "expected_n_tokens", "tok", "ids_sha256"}, "unexpected payload fields"
        assert hashlib.sha256(r["prompt"].encode()).hexdigest() == r["prompt_sha256"]
    assert tok.convert_tokens_to_ids("A") == m["A_id"] and tok.convert_tokens_to_ids("B") == m["B_id"]
    model.eval()
    w = ShardWriterX(Path(out_dir), shard)
    first, divergences, max_diff = {}, [], {"post_norm": 0.0, "pre_norm": 0.0}
    plan = rows + rows[:n_replay]
    for i, r in enumerate(plan):
        if time.time() - t0 > guard_s:
            raise RuntimeError(f"guard {guard_s}s reached after {i} worlds (last world {r['world_id']})")
        text = render(tok, "gemma4_31b", r["prompt"])
        rec = capture_world_multi(model, tok, text, r["tok"], tuple(layers), m["A_id"], m["B_id"], device, r["world_id"], r["ids_sha256"])
        assert rec["meta"]["n_tokens"] == r["expected_n_tokens"], f"[{r['world_id']}] token count mismatch"
        for name, f in rec["meta"]["fidelity"].items():
            max_diff[name] = max(max_diff[name], f["max_abs_diff"])
            if f["near_tie_top1"] or f["ab_tie_original"]:
                divergences.append({k: f[k] for k in ("world_id", "reconstruction", "orig_top1", "recon_top1", "max_abs_diff",
                                                      "tolerance", "near_tie_top1", "ab_tie_original")})
        if (i + 1) % 400 == 0:
            print(f"{i + 1}/{len(plan)} observations at {time.time() - t0:.1f}s; divergences logged {len(divergences)}", flush=True)
        if i < len(rows):
            w.add(r["world_id"], rec)
            if i < n_replay:
                first[r["world_id"]] = rec["acts_u16"].copy()
        else:
            assert np.array_equal(first[r["world_id"]], rec["acts_u16"]), f"replay not bitwise identical: {r['world_id']}"
    return w.close({"n_worlds": len(rows), "layers": list(layers), "P": len(rows[0]["tok"]),
                    "replay_bitwise_identical": min(n_replay, len(rows)),
                    "fidelity_summary": {"max_abs_diff": max_diff, "tau": TAU, "limit": FIDELITY_MAX_ABS,
                                         "n_divergences_logged": len(divergences), "divergences": divergences},
                    "layout": "acts[i] = P x L x d uint16(bf16); slots per d3x_positions.SLOT_NAMES; layers 0..60 HF hidden_states, "
                              "61 raw block-60 pre-norm; masked slots are zero", "wall_s": time.time() - t0})
