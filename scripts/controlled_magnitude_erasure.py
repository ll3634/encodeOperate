#!/usr/bin/env python3
"""Controlled-magnitude erasure: h' = h - c * sign(h.D_hat) * D_hat.

Removes a FIXED amount of energy along D_hat (in the natural sign
direction), regardless of |h.D_hat|.  This isolates readout gain
from representation strength.

Magnitudes:
    c_E = mean_i |h_i . E_hat|       (E-scale)
    c_A = mean_i |h_i . A_hat|       (A-scale, much larger)

Directions: A, E, D3, D1, D2, D4 + 10 cached random unit dirs.
Plus: E(theta) on the E->D3 path at all 7 angles.
Total: 16 + 7 = 23 dirs x 2 magnitudes x N=100 = 4,600 forwards.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from agent.prompts import ACTION_TOKENS
from steering.hook_utils import get_model_layers
from evidence_erasure_test import (build_p0_prompt, margin_from_logits, LAYER)
from dose_response_erasure import rebuild_prompts
from nullspace_rotation_io import (load_directions as load_core_directions,
                                   construct_family, ANGLES_DEG)

OUT = Path("results/evidence_erasure_test/controlled_magnitude")
OUT.mkdir(parents=True, exist_ok=True)
N_BOOT = 2000
SEED_BOOT = 12345
RANDOM_K = 10  # use first 10 of cached 20 random dirs


# --------------------------------------------------------------------- hook
class ControlledMagnitudeHook:
    """h' = h - c * sign(h . D_hat) * D_hat at LAYER, last token, single-shot."""

    def __init__(self, model, direction, c, layer=LAYER, max_interventions=1):
        self.model = model
        d = np.asarray(direction, np.float32).reshape(-1)
        n = float(np.linalg.norm(d))
        assert n > 1e-8, "direction has zero norm"
        self.unit_np = (d / n).astype(np.float32)
        self.c = float(c)
        self.layer = layer
        self.max_int = max_interventions
        self.handle = None
        self.unit = None
        self._count = 0

    def __enter__(self):
        layers = get_model_layers(self.model)

        def hook_fn(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self.max_int is not None and self._count >= self.max_int:
                return output
            self._count += 1
            if self.unit is None:
                self.unit = torch.tensor(self.unit_np, dtype=hidden.dtype,
                                         device=hidden.device)
            seq_len = hidden.shape[1]
            pos = seq_len - 1
            h = hidden[:, pos, :]
            proj_signed = (h * self.unit).sum(dim=-1, keepdim=True)  # (1,1)
            sign = torch.sign(proj_signed)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            hidden[:, pos, :] = h - self.c * sign * self.unit
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        self.handle = layers[self.layer].register_forward_hook(hook_fn)
        return self

    def __exit__(self, *a):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.unit = None
        self._count = 0


def forward_margin(model, tok, prompt, hook_factory, tool_ids, fin_ids):
    ids = tok.encode(prompt, return_tensors="pt").to(next(model.parameters()).device)
    if hook_factory is None:
        with torch.no_grad():
            logits = model(ids).logits[0, -1, :]
    else:
        with hook_factory():
            with torch.no_grad():
                logits = model(ids).logits[0, -1, :]
    return margin_from_logits(logits, tool_ids, fin_ids)


def boot_ci(x: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED_BOOT):
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = x[idx].mean()
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_all_directions():
    core = load_core_directions()  # A, E, D3, D1
    extras = {
        "D2": np.load("results/ocft/per_candidate/D2_action_prior/direction.npy").astype(np.float64),
        "D4": np.load("results/ocft/per_candidate/D4_obs_length/direction.npy").astype(np.float64),
    }
    rand_mat = np.load("results/evidence_erasure_test/random_control/random_directions.npy"
                       ).astype(np.float64)
    out = {"A": core["A"], "E": core["E"], "D3": core["D3"], "D1": core["D1"],
           **{k: v / (np.linalg.norm(v) + 1e-12) for k, v in extras.items()}}
    for i in range(RANDOM_K):
        v = rand_mat[i]
        out[f"r_{i+1:02d}"] = v / (np.linalg.norm(v) + 1e-12)
    return out


def build_rotation_directions():
    core = load_core_directions()
    family, _meta = construct_family(core)
    return {f"E_to_D3__theta{t:02d}": family["E_to_D3"][t] for t in ANGLES_DEG}


if __name__ == "__main__":
    print("[smoke] this module is imported by controlled_run.py")
