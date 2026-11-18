"""D3 capture core (torch; runs inside the Modal container and in CPU tests with a tiny model).

Per world, at the last pre-answer position (`<channel|>` for Gemma 4):
  * acts  : 62 x d, bf16 stored losslessly as uint16 words
            [0..60] = HF output_hidden_states (0 = scaled input embeddings; 1..59 = block outputs;
            60 = block-60 output AFTER the final norm), [61] = raw block-60 output BEFORE the final norm (hook)
  * mean_emb : float32 d = mean over ALL prompt positions of hidden_states[0] (input for baseline B1)
  * forced-choice / top-1 fields as in V2 (full vocabulary)
  * fidelity: logits re-derived (a) from the post-norm vector [60] and (b) from the pre-norm vector [61]
    through final norm -> lm_head -> soft-cap: max|diff| <= 0.25, same A/B decision, top-1 equal or numerically indiscernible
Candidate embeddings (frozen, context-free): mean of the scaled input-embedding rows of encode(" " + label).
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

FIDELITY_MAX_ABS = 0.25  # logit units over the FULL vocabulary; hard limit (unchanged)
# TAU: fixed before any GPU run (D3 technical amendment A2, 2026-11-13). Gemma-4 logits are soft-capped to
# |x| < 30 and produced in bf16; the bf16 spacing (1 ULP) is 0.125 for |x| in [16, 32), 0.0625 in [8, 16), ...
# TAU = 0.125 = the LARGEST bf16 ULP attainable under the soft-cap, i.e. one quantisation step of the model's own logits.
TAU = 0.125


class FidelityError(AssertionError):
    pass


def check_fidelity(orig, recon, a_id: int, b_id: int, world_id: str, name: str, tau: float = TAU) -> dict:
    """Pure check of one reconstruction against the model's ORIGINAL logits (torch 1-D float tensors).
    Rules (owner decision 2026-11-13, incident BTC-D3-DEV):
      (1) max |recon - orig| over the full vocabulary <= FIDELITY_MAX_ABS
      (2) identical A/B decision: if the original is strict (A or B), the reconstruction must give the same
          strict sign. [Original exact tie (bf16): the reconstruction must also be numerically indiscernible,
          |rA - rB| <= 2*max_abs_diff + tau -- extension submitted for owner validation; logged as ab_tie_original.]
      (3) top-1: the reconstructed top-1 r1 is accepted iff r1 == original top-1 t1 OR
          orig[t1] - orig[r1] <= 2*max_abs_diff + tau  (any number of indiscernible tokens; not limited to top-2);
          accepted mismatches are flagged near_tie_top1.
    Conformity metrics are NEVER computed from reconstructions (callers use the original logits)."""
    import torch
    if not (bool(torch.isfinite(orig).all()) and bool(torch.isfinite(recon).all())):  # NaN/inf would defeat comparisons
        raise FidelityError(f"capture fidelity failed [{world_id}] ({name}): non-finite logits "
                            f"(orig finite={bool(torch.isfinite(orig).all())}, recon finite={bool(torch.isfinite(recon).all())})")
    d = float((recon - orig).abs().max())
    t1, r1 = int(orig.argmax()), int(recon.argmax())
    margin = float(orig[t1] - orig[r1])
    tol = 2 * d + tau
    oa, ob, ra, rb = float(orig[a_id]), float(orig[b_id]), float(recon[a_id]), float(recon[b_id])
    ab_tie_original = oa == ob
    if ab_tie_original:
        ab_ok = abs(ra - rb) <= tol
    else:
        ab_ok = (ra > rb) == (oa > ob) and ra != rb
    top1_ok = r1 == t1 or margin <= tol
    rec = {"world_id": world_id, "reconstruction": name, "max_abs_diff": d, "orig_top1": t1, "recon_top1": r1,
           "orig_margin_top1_vs_recon_top1": margin, "tolerance": tol, "near_tie_top1": bool(r1 != t1 and top1_ok),
           "ab_decision_same": bool(ab_ok), "ab_tie_original": bool(ab_tie_original), "top1_ok": bool(top1_ok)}
    if d > FIDELITY_MAX_ABS or not ab_ok or not top1_ok:
        raise FidelityError(f"capture fidelity failed [{world_id}] ({name}): {rec}")
    return rec


def _text_parts(model):
    lm = model.model.language_model
    return lm, lm.layers, lm.norm, model.lm_head, getattr(model.config.get_text_config(), "final_logit_softcapping", None)


def _logits_from(vec, norm_mod, head, cap, apply_norm: bool):
    import torch
    x = norm_mod(vec.unsqueeze(0)) if apply_norm else vec.unsqueeze(0)
    y = head(x)[0].float()
    return torch.tanh(y / cap) * cap if cap else y


def capture_world(model, tok, text: str, a_id: int, b_id: int, device: str = "cuda", world_id: str = "?") -> dict:
    import torch
    lm, layers, norm_mod, head, cap = _text_parts(model)
    raw = {}
    h = layers[-1].register_forward_hook(lambda m, i, o: raw.__setitem__("x", (o[0] if isinstance(o, tuple) else o)[0, -1].detach()))
    try:
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
            hs = out.hidden_states
            logits = out.logits[0, -1].float()
            post = hs[-1][0, -1]
            pre = raw["x"]
            l_post = _logits_from(post, norm_mod, head, cap, apply_norm=False)
            l_pre = _logits_from(pre, norm_mod, head, cap, apply_norm=True)
            acts = torch.stack([x[0, -1] for x in hs] + [pre])  # (62, d) in model dtype
            mean_emb = hs[0][0].float().mean(dim=0)
    finally:
        h.remove()
    lp = torch.log_softmax(logits, -1)
    top = lp.topk(5)
    fid = {name: check_fidelity(logits, l2, a_id, b_id, world_id, name)
           for name, l2 in (("post_norm", l_post), ("pre_norm", l_pre))}
    # conformity / forced-choice fields below come from the ORIGINAL logits only
    if not (bool(torch.isfinite(acts).all()) and bool(torch.isfinite(mean_emb).all())):
        raise FidelityError(f"capture fidelity failed [{world_id}]: non-finite activations or mean_emb")
    a16 = acts.to(torch.bfloat16).cpu().view(torch.int16).numpy().view(np.uint16)
    return {"acts_u16": a16, "mean_emb": mean_emb.cpu().numpy().astype(np.float32),
            "meta": {"n_tokens": int(ids.shape[1]), "logp_A": float(lp[a_id]), "logp_B": float(lp[b_id]),
                     "top1_id": int(top.indices[0]), "top1_in_AB": int(top.indices[0]) in (a_id, b_id),
                     "top5": [[int(j), tok.convert_ids_to_tokens(int(j)), float(v)] for v, j in zip(top.values, top.indices)],
                     "fidelity": fid}}


def candidate_embeddings(model, tok, labels: list[str]) -> tuple[np.ndarray, list[list[int]]]:
    import torch
    emb = model.model.language_model.embed_tokens
    out, toks = [], []
    with torch.no_grad():
        for lab in labels:
            ids = tok.encode(" " + lab, add_special_tokens=False)
            toks.append(ids)
            out.append(emb(torch.tensor([ids], device=emb.weight.device))[0].float().mean(0).cpu().numpy())
    return np.stack(out).astype(np.float32), toks


def _npy_bytes(a: np.ndarray) -> bytes:
    b = io.BytesIO()
    np.save(b, a, allow_pickle=False)
    return b.getvalue()


class ShardWriter:
    """Writes fixed-size shards to `root` (a mounted Volume in the container, a temp dir in tests)."""

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
        name = f"shard_{self.k:03d}"
        files = {f"{name}_acts.npy": _npy_bytes(np.stack([r["acts_u16"] for _, r in self.buf])),
                 f"{name}_mean_emb.npy": _npy_bytes(np.stack([r["mean_emb"] for _, r in self.buf])),
                 f"{name}_index.jsonl": "".join(json.dumps({"world_id": w, **r["meta"]}) + "\n" for w, r in self.buf).encode()}
        entry = {"name": name, "n": len(self.buf), "files": {}}
        for fn, data in files.items():
            (self.root / fn).write_bytes(data)
            entry["files"][fn] = hashlib.sha256(data).hexdigest()
        self.manifest["shards"].append(entry)
        self.buf, self.k = [], self.k + 1

    def close(self, extra: dict | None = None) -> dict:
        self.flush()
        if extra:
            self.manifest.update(extra)
        (self.root / "MANIFEST.json").write_text(json.dumps(self.manifest, indent=1), encoding="utf-8")
        return self.manifest


def verify_shards(root: Path) -> dict:
    """Local verification after download: every file's sha256 matches the manifest; shapes consistent."""
    man = json.loads((Path(root) / "MANIFEST.json").read_text(encoding="utf-8"))
    ok, n = True, 0
    for key, fn in (("candidate_embeddings_sha256", "candidate_embeddings.npy"), ("candidate_labels_sha256", "candidate_labels.json")):
        if key in man:
            ok &= hashlib.sha256((Path(root) / fn).read_bytes()).hexdigest() == man[key]
    for s in man["shards"]:
        for fn, h in s["files"].items():
            ok &= hashlib.sha256((Path(root) / fn).read_bytes()).hexdigest() == h
        a = np.load(Path(root) / f"{s['name']}_acts.npy")
        e = np.load(Path(root) / f"{s['name']}_mean_emb.npy")
        ok &= a.shape[0] == e.shape[0] == s["n"] and a.dtype == np.uint16 and e.dtype == np.float32
        n += s["n"]
    return {"hashes_and_shapes_ok": bool(ok), "n_worlds": n}


def u16_to_bf16_f32(a: np.ndarray) -> np.ndarray:
    """Exact bf16 -> float32 expansion of stored uint16 words (numpy only; no torch needed locally)."""
    return (a.astype(np.uint32) << 16).view(np.float32)


def run_capture(rows, labels, digest, out_dir: Path, device, model, tok, guard_s: int, shard: int, n_replay: int) -> dict:
    """Full capture loop (container and CPU tests). Oracle-free payload; asserts integrity; bitwise replay."""
    import time
    from v2_common import MODELS, render
    m = MODELS["gemma4_31b"]
    t0 = time.time()
    assert hashlib.sha256("\n".join(r["prompt_sha256"] for r in rows).encode()).hexdigest() == digest, "digest mismatch"
    for r in rows:
        assert set(r) == {"world_id", "prompt", "prompt_sha256", "expected_n_tokens"}, "unexpected payload fields"
        assert hashlib.sha256(r["prompt"].encode()).hexdigest() == r["prompt_sha256"]
    assert tok.convert_tokens_to_ids("A") == m["A_id"] and tok.convert_tokens_to_ids("B") == m["B_id"]
    model.eval()
    w = ShardWriter(Path(out_dir), shard)
    first, divergences, max_diff = {}, [], {"post_norm": 0.0, "pre_norm": 0.0}
    plan = rows + rows[:n_replay]
    for i, r in enumerate(plan):
        if time.time() - t0 > guard_s:
            raise RuntimeError(f"guard {guard_s}s reached after {i} worlds (last world {r['world_id']})")
        text = render(tok, "gemma4_31b", r["prompt"])
        rec = capture_world(model, tok, text, m["A_id"], m["B_id"], device, world_id=r["world_id"])
        assert rec["meta"]["n_tokens"] == r["expected_n_tokens"], f"[{r['world_id']}] token count {rec['meta']['n_tokens']} != {r['expected_n_tokens']}"
        for name, f in rec["meta"]["fidelity"].items():
            max_diff[name] = max(max_diff[name], f["max_abs_diff"])
            if f["near_tie_top1"] or f["ab_tie_original"]:
                divergences.append({k: f[k] for k in ("world_id", "reconstruction", "orig_top1", "recon_top1",
                                                      "orig_margin_top1_vs_recon_top1", "max_abs_diff", "tolerance",
                                                      "near_tie_top1", "ab_tie_original")})
        if (i + 1) % 400 == 0:
            print(f"{i + 1}/{len(plan)} observations at {time.time() - t0:.1f}s; divergences logged {len(divergences)}", flush=True)
        if i < len(rows):
            w.add(r["world_id"], rec)
            if i < n_replay:
                first[r["world_id"]] = rec["acts_u16"].copy()
        else:
            assert np.array_equal(first[r["world_id"]], rec["acts_u16"]), f"replay not bitwise identical: {r['world_id']}"
    cand, cand_toks = candidate_embeddings(model, tok, labels)
    cand_bytes = _npy_bytes(cand)
    (Path(out_dir) / "candidate_embeddings.npy").write_bytes(cand_bytes)
    (Path(out_dir) / "candidate_labels.json").write_text(json.dumps({"labels": labels, "token_ids": cand_toks}), encoding="utf-8")
    return w.close({"n_worlds": len(rows), "replay_bitwise_identical": min(n_replay, len(rows)),
                    "fidelity_summary": {"max_abs_diff": max_diff, "tau": TAU, "limit": FIDELITY_MAX_ABS,
                                         "n_divergences_logged": len(divergences), "divergences": divergences},
                    "candidate_embeddings_sha256": hashlib.sha256(cand_bytes).hexdigest(),
                    "candidate_labels_sha256": hashlib.sha256((Path(out_dir) / "candidate_labels.json").read_bytes()).hexdigest(),
                    "layout": "acts[i] = 62 x d uint16(bf16): 0..60 HF hidden_states (60 post-final-norm), 61 raw block-60 pre-norm; "
                              "mean_emb[i] = mean over prompt positions of hidden_states[0] (float32)",
                    "wall_s": time.time() - t0})
