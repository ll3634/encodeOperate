#!/usr/bin/env python3
"""Gradient erasure dose-response for {A, E, D3, D1} at L20.

Runs only NEW α ∈ {0.25, 0.50, 0.75} for the four directions on
the same N=100 §3 prompts (1,200 forward passes).
Reuses cached α=0/1/2 from:
  results/evidence_erasure_test/per_prompt_margins.npz   (A, E)
  results/evidence_erasure_test/random_control/new_margins.npz (D3, D1)

Also captures L20 hidden states at the decision token for the
projection-vs-effect appendix figure (one extra hook'd baseline
pass with state capture; reuses the same baseline margins to
verify match).
"""
import json, time
from pathlib import Path
import numpy as np
import sys, torch
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from agent.prompts import ACTION_TOKENS
from steering.hook_utils import get_model_layers
from evidence_erasure_test import (ProjectionFlipHook, build_p0_prompt,
                                   forward_margin, LAYER)

ALPHAS_NEW = [0.25, 0.50, 0.75]
ALPHAS_ALL = [0.00, 0.25, 0.50, 0.75, 1.00, 2.00]
DIR_NAMES  = ["A", "E", "D3", "D1"]

OUT_TEST = Path("results/evidence_erasure_test")
OUT      = OUT_TEST / "dose_response"; OUT.mkdir(parents=True, exist_ok=True)


def load_directions():
    E = np.load("results/phase1_probe/probe_direction_l20.npz",
                allow_pickle=True)["decision_direction"].astype(np.float32)
    A = np.load("steering/directions/direction_decomp_full_layer20.npz",
                allow_pickle=True)["decision_direction"].astype(np.float32)
    D3 = np.load("results/ocft/per_candidate/D3_candidate_present/direction.npy"
                 ).astype(np.float32)
    D1 = np.load("results/ocft/per_candidate/D1_source/direction.npy"
                 ).astype(np.float32)
    out = {}
    for nm, v in [("A", A), ("E", E), ("D3", D3), ("D1", D1)]:
        out[nm] = (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)
    return out


class HiddenStateCaptureHook:
    """Captures hidden state at LAYER, last token, per call."""
    def __init__(self, model, layer=LAYER):
        self.model = model; self.layer = layer
        self.handle = None; self.last = None
    def __enter__(self):
        layers = get_model_layers(self.model)
        def hook_fn(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.last = hidden[:, -1, :].detach().float().cpu().numpy()[0].copy()
            return output
        self.handle = layers[self.layer].register_forward_hook(hook_fn)
        return self
    def __exit__(self, *a):
        if self.handle is not None:
            self.handle.remove(); self.handle = None
        return False


def rebuild_prompts(tok, sample_ids):
    label_data = [json.loads(l) for l in open("results/phase1_probe/labels.jsonl")]
    by_id = {ld["sample_id"]: ld for ld in label_data}
    bl = {}
    with open("results/l20_rho020_n500/baseline_results.jsonl") as f:
        for line in f:
            ep = json.loads(line); bl[ep["sample_id"]] = ep
    return [build_p0_prompt(tok, by_id[sid]["question"],
                            bl[sid]["steps"][0]["action_input"],
                            bl[sid]["steps"][0]["observation"])
            for sid in sample_ids]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache = np.load(OUT_TEST / "per_prompt_margins.npz")
    sample_ids = [str(s) for s in cache["sample_ids"]]
    base_cached = cache["baseline"].astype(np.float32)
    cached = {
        ("A", 1.0): cache["erase_A"].astype(np.float32),
        ("A", 2.0): cache["flip_A"].astype(np.float32),
        ("E", 1.0): cache["erase_E"].astype(np.float32),
        ("E", 2.0): cache["flip_E"].astype(np.float32),
    }
    rc = np.load(OUT_TEST / "random_control" / "new_margins.npz")
    for nm in ("D3", "D1"):
        cached[(nm, 1.0)] = rc[f"{nm}_erase"].astype(np.float32)
        cached[(nm, 2.0)] = rc[f"{nm}_flip"].astype(np.float32)
    n = len(sample_ids)
    print(f"[cache] N={n}; reused α∈{{1.0, 2.0}} for A,E,D3,D1")

    dirs = load_directions()

    print("\n[load] Qwen/Qwen2.5-7B-Instruct")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    prompts = rebuild_prompts(tok, sample_ids)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True).eval()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]

    # 1) New α sweep + capture baseline hidden states at L20 last token
    print(f"\n[plan] new: {len(ALPHAS_NEW)} α × {len(DIR_NAMES)} dir × {n} = "
          f"{len(ALPHAS_NEW)*len(DIR_NAMES)*n} forwards (+{n} baseline-capture)")
    new_margins = {(nm, a): np.zeros(n, dtype=np.float32)
                   for nm in DIR_NAMES for a in ALPHAS_NEW}
    base_recap  = np.zeros(n, dtype=np.float32)
    H = np.zeros((n, 3584), dtype=np.float32)
    t0 = time.time()
    for i, p in enumerate(prompts):
        # baseline pass with hidden capture (also re-measures margin)
        cap = HiddenStateCaptureHook(model)
        with cap:
            ids = tok.encode(p, return_tensors="pt").to(next(model.parameters()).device)
            with torch.no_grad():
                logits = model(ids).logits[0, -1, :]
            from evidence_erasure_test import margin_from_logits
            base_recap[i] = margin_from_logits(logits, tool_ids, fin_ids)
        H[i] = cap.last
        for nm in DIR_NAMES:
            vec = dirs[nm]
            for a in ALPHAS_NEW:
                new_margins[(nm, a)][i] = forward_margin(
                    model, tok, p,
                    lambda v=vec, aa=a: ProjectionFlipHook(model, v, factor=aa),
                    tool_ids, fin_ids)
        if (i + 1) % 10 == 0 or i == 0:
            eta = (time.time() - t0) / (i + 1) * (n - i - 1)
            print(f"  [{i+1:>3d}/{n}] ETA={eta:.0f}s")
    print(f"[done] forwards: {time.time() - t0:.0f}s")

    # Sanity: baseline recapture must equal cached baseline
    diff = float(np.abs(base_recap - base_cached).max())
    print(f"[sanity] |base_recap - base_cached|_max = {diff:.6f} "
          f"(must be ~0; otherwise pipeline drift)")

    np.savez(OUT / "per_prompt_margins_alpha_new.npz",
             sample_ids=np.array(sample_ids),
             baseline_recapture=base_recap,
             hidden_L20=H,
             **{f"{nm}_alpha{a:.2f}".replace(".","p"): new_margins[(nm, a)]
                for nm in DIR_NAMES for a in ALPHAS_NEW})

    from dose_response_io import analyse_and_write
    analyse_and_write(cached, new_margins, base_cached, dirs, H,
                      sample_ids, OUT_TEST, OUT)


if __name__ == "__main__":
    main()
