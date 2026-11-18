"""D3 design check on CPU (no weights): fidelity of hidden-state capture for the Gemma 4 architecture.

Uses a tiny random Gemma4ForConditionalGeneration built from the REAL config structure
(v2_cpu_checks.tiny_config) and the real tokenizer + template. Verifies:
  1. output_hidden_states returns num_layers + 1 tensors; [0] = scaled input embeddings.
  2. forward hooks on each decoder layer reproduce hidden_states[1..L] exactly; documents whether
     hidden_states[L] is pre- or post-final-norm.
  3. logits at the pre-answer position are reproduced from the captured last state
     (final norm -> lm_head -> soft-capping), i.e. the captured vector is the one the answer is read from.
  4. bitwise determinism of a repeated forward; bf16 storage as raw uint16 round-trips exactly.
Run in .venv-gemma4. Writes outputs/v2_audit/d3_capture_check.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoConfig, AutoTokenizer, Gemma4ForConditionalGeneration  # noqa: E402

from v2_common import MODELS, load_model_inputs, render  # noqa: E402
from v2_cpu_checks import tiny_config  # noqa: E402


def main() -> None:
    m = MODELS["gemma4_31b"]
    cfg = AutoConfig.from_pretrained(m["repo"], revision=m["revision"])
    tok = AutoTokenizer.from_pretrained(m["repo"], revision=m["revision"])
    torch.manual_seed(0)
    model = Gemma4ForConditionalGeneration._from_config(tiny_config(cfg), dtype=torch.bfloat16).eval()
    lm = model.model.language_model
    layers = lm.layers
    rep = {"tiny_layers": len(layers), "module_path_layers": "model.model.language_model.layers",
           "module_path_norm": "model.model.language_model.norm"}
    rows = load_model_inputs(ROOT, "SCREEN")[:3]
    caught = {}
    hooks = [layer.register_forward_hook(lambda mod, inp, out, i=i: caught.__setitem__(i, (out[0] if isinstance(out, tuple) else out).detach()))
             for i, layer in enumerate(layers)]
    checks = []
    for r in rows:
        ids = tok(render(tok, "gemma4_31b", r["prompt"]), return_tensors="pt", add_special_tokens=False).input_ids
        caught.clear()
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
            out2 = model(input_ids=ids, output_hidden_states=True)
        hs = out.hidden_states
        n_hs = len(hs)
        hook_eq = [torch.equal(caught[i][0, -1], hs[i + 1][0, -1]) for i in range(len(layers) - 1)]
        last_hook_eq = torch.equal(caught[len(layers) - 1][0, -1], hs[-1][0, -1])
        normed_last = lm.norm(caught[len(layers) - 1])[0, -1]
        hs_last_is_post_norm = torch.allclose(hs[-1][0, -1].float(), normed_last.float(), atol=1e-2)
        # logits reconstruction from the captured final state
        final = hs[-1][0, -1] if hs_last_is_post_norm else normed_last
        logits = model.lm_head(final.unsqueeze(0))[0].float()
        cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
        if cap:
            logits = torch.tanh(logits / cap) * cap
        logits_eq = torch.allclose(logits, out.logits[0, -1].float(), atol=1e-2)
        emb = lm.embed_tokens(ids)[0, -1]
        emb_eq = torch.allclose(emb.float(), hs[0][0, -1].float(), atol=1e-2)
        det = all(torch.equal(a, b) for a, b in zip(out.hidden_states, out2.hidden_states))
        stacked = torch.stack([h[0, -1] for h in hs])  # (L+1, d) bf16
        rt = stacked.view(torch.int16).numpy().tobytes()
        back = torch.frombuffer(bytearray(rt), dtype=torch.int16).view(torch.bfloat16).reshape(stacked.shape)
        checks.append({"world_id": r["world_id"], "n_hidden_states": n_hs, "hooks_equal_layers_1..L-1": all(hook_eq),
                       "hook_last_equals_hs_last": bool(last_hook_eq), "hs_last_is_post_final_norm": bool(hs_last_is_post_norm),
                       "logits_reconstructed_from_capture": bool(logits_eq), "hs0_equals_embed_tokens_output": bool(emb_eq),
                       "bitwise_deterministic": bool(det), "bf16_raw_roundtrip_exact": bool(torch.equal(back, stacked))})
    for h in hooks:
        h.remove()
    rep["checks"] = checks
    rep["all_pass"] = all(c["n_hidden_states"] == len(layers) + 1 and c["hooks_equal_layers_1..L-1"]
                          and c["logits_reconstructed_from_capture"] and c["bitwise_deterministic"]
                          and c["bf16_raw_roundtrip_exact"] for c in checks)
    (ROOT / "outputs" / "v2_audit" / "d3_capture_check.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
