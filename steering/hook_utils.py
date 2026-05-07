#!/usr/bin/env python3
"""
PyTorch hooks for hidden-state steering.
Extracted and adapted from causal_intervention.py for E2E agent use.
"""

import os
import torch
import numpy as np
from typing import Optional, Callable


def _path_b_log_enabled() -> bool:
    return os.environ.get("PATH_B_LOG_HOOK") == "1"


def get_model_layers(model):
    """Get the list of transformer layers from the model."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers  # Qwen, LLaMA, etc.
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h  # GPT-2 style
    else:
        raise ValueError("Cannot find model layers. Unsupported architecture.")


def compute_rms(arr: np.ndarray) -> float:
    """Compute root mean square of array."""
    return float(np.sqrt(np.mean(arr ** 2)))


class SteeringHook:
    """
    Context manager for applying hidden-state steering via PyTorch hooks.
    
    Usage:
        with SteeringHook(model, direction, alpha, layer=12, position=-1) as hook:
            outputs = model(input_ids)
            # hidden states at layer 12, position -1 are modified by +alpha*direction
    """
    
    def __init__(
        self,
        model,
        direction: np.ndarray,
        alpha: float,
        layer: int = 12,
        position: int = -1,
        mode: str = "addition",
        max_interventions: Optional[int] = None,
    ):
        """
        Args:
            model: HuggingFace model
            direction: Steering direction vector (numpy array)
            alpha: Steering strength (scaled)
            layer: Target layer index (negative = from end)
            position: Token position to intervene (-1 = last prompt token)
            mode: "addition" (h' = h + α·d), "projection", or "clamping"
            max_interventions: If set, only apply steering for the first N forward
                passes (decision-only steering). None = unlimited.
        """
        self.model = model
        self.direction = direction
        self.alpha = alpha
        self.layer = layer
        self.position = position
        self.mode = mode
        self.max_interventions = max_interventions
        self.handle = None
        self._direction_tensor = None
        self._direction_unit = None
        self._intervention_count = 0
        # Path B instrumentation (env-gated; no-op when PATH_B_LOG_HOOK unset).
        # Runner sets _current_item_id before each forward; logs append here.
        self._path_b_log = []
        self._current_item_id = None
        
    def __enter__(self):
        layers = get_model_layers(self.model)
        num_layers = len(layers)
        actual_layer = self.layer if self.layer >= 0 else num_layers + self.layer
        
        if actual_layer < 0 or actual_layer >= num_layers:
            raise ValueError(f"Layer {self.layer} out of range [0, {num_layers})")
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # Decision-only steering: skip intervention after max_interventions
            if self.max_interventions is not None and self._intervention_count >= self.max_interventions:
                return output

            self._intervention_count += 1

            # Lazy init tensors on correct device/dtype
            if self._direction_tensor is None:
                self._direction_tensor = torch.tensor(
                    self.direction * self.alpha,
                    dtype=hidden.dtype,
                    device=hidden.device
                )
                d_norm = np.linalg.norm(self.direction)
                if d_norm > 0:
                    self._direction_unit = torch.tensor(
                        self.direction / d_norm,
                        dtype=hidden.dtype,
                        device=hidden.device
                    )
                else:
                    self._direction_unit = self._direction_tensor
            
            # Compute intervention position
            seq_len = hidden.shape[1]
            if self.position < 0:
                pos = seq_len + self.position
            else:
                pos = self.position
            pos = max(0, min(pos, seq_len - 1))
            
            # Apply intervention
            if pos < seq_len:
                h = hidden[:, pos, :]
                # Path B: capture h_before for actual-delta diagnostic
                log_this = _path_b_log_enabled()
                if log_this:
                    h_before = h.detach().clone()
                if self.mode == "addition":
                    hidden[:, pos, :] = h + self._direction_tensor
                elif self.mode == "projection":
                    proj = torch.sum(h * self._direction_unit, dim=-1, keepdim=True) * self._direction_unit
                    hidden[:, pos, :] = h - proj
                elif self.mode == "clamping":
                    proj = torch.sum(h * self._direction_unit, dim=-1, keepdim=True) * self._direction_unit
                    hidden[:, pos, :] = h - proj + self._direction_tensor
                else:
                    hidden[:, pos, :] = h + self._direction_tensor

                if log_this:
                    h_after = hidden[:, pos, :].detach()
                    actual_delta = (h_after - h_before).flatten().float()
                    intended_delta = self._direction_tensor.detach().flatten().float()
                    a_norm = float(actual_delta.norm().item())
                    i_norm = float(intended_delta.norm().item())
                    if a_norm > 0 and i_norm > 0:
                        cos_ai = float(torch.dot(actual_delta, intended_delta).item() / (a_norm * i_norm))
                    else:
                        cos_ai = float("nan")
                    h_before_norm = float(h_before.flatten().float().norm().item())
                    self._path_b_log.append({
                        "item_id": self._current_item_id,
                        "layer": int(actual_layer),
                        "alpha": float(self.alpha),
                        "pos": int(pos),
                        "seq_len": int(seq_len),
                        "intervention_index": int(self._intervention_count),
                        "h_before_norm": h_before_norm,
                        "h_after_norm": float(h_after.flatten().float().norm().item()),
                        "intended_delta_norm": i_norm,
                        "actual_delta_norm": a_norm,
                        "cos_actual_intended": cos_ai,
                        "norm_ratio_actual_over_h_before": (a_norm / h_before_norm) if h_before_norm > 0 else float("nan"),
                        "mode": self.mode,
                    })

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        
        self.handle = layers[actual_layer].register_forward_hook(hook_fn)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self._direction_tensor = None
        self._direction_unit = None
        self._intervention_count = 0
        return False


class CrossAxisHook:
    """
    Rank-1 dynamic cross-axis intervention.

    At ``layer``/``position``, modify the residual stream as:
        h' = h + alpha * (h . u_in) * u_out
    where ``u_in`` and ``u_out`` are unit-norm vectors.  The magnitude of the
    push along ``u_out`` is gated by the projection of the live hidden state
    onto ``u_in`` -- i.e. "however much of the input axis is currently in h,
    push that much along the output axis".  Used to test whether mapping the
    evidence axis onto the action axis makes evidence operative for the
    stop/continue decision (Arm B of the reconnection sweep).
    """

    def __init__(
        self,
        model,
        u_in: np.ndarray,
        u_out: np.ndarray,
        alpha: float,
        layer: int = 20,
        position: int = -1,
        max_interventions: Optional[int] = None,
    ):
        self.model = model
        self.u_in = u_in / (np.linalg.norm(u_in) + 1e-12)
        self.u_out = u_out / (np.linalg.norm(u_out) + 1e-12)
        self.alpha = alpha
        self.layer = layer
        self.position = position
        self.max_interventions = max_interventions
        self.handle = None
        self._u_in_t = None
        self._u_out_t = None
        self._intervention_count = 0

    def __enter__(self):
        layers = get_model_layers(self.model)
        num_layers = len(layers)
        actual_layer = self.layer if self.layer >= 0 else num_layers + self.layer
        if actual_layer < 0 or actual_layer >= num_layers:
            raise ValueError(f"Layer {self.layer} out of range [0, {num_layers})")

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            if self.max_interventions is not None and self._intervention_count >= self.max_interventions:
                return output
            self._intervention_count += 1

            if self._u_in_t is None:
                self._u_in_t = torch.tensor(self.u_in, dtype=hidden.dtype, device=hidden.device)
                self._u_out_t = torch.tensor(self.u_out, dtype=hidden.dtype, device=hidden.device)

            seq_len = hidden.shape[1]
            pos = seq_len + self.position if self.position < 0 else self.position
            pos = max(0, min(pos, seq_len - 1))

            h = hidden[:, pos, :]
            proj = (h * self._u_in_t).sum(dim=-1, keepdim=True)
            hidden[:, pos, :] = h + self.alpha * proj * self._u_out_t

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        self.handle = layers[actual_layer].register_forward_hook(hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self._u_in_t = None
        self._u_out_t = None
        self._intervention_count = 0
        return False


class KVGroupScalingHook:
    """
    Context manager for scaling specific KV-group head outputs at attn's o_proj.

    Hooks into layers[layer].self_attn.o_proj to scale the pre-projection
    output of heads belonging to a specific KV group at the last token position.

    Qwen2.5-7B GQA: 28 Q-heads, 4 KV groups of 7 heads each, head_dim=128.
    KV group k contains Q-heads [k*7, ..., k*7+6].

    Usage:
        with KVGroupScalingHook(model, layer=18, kv_group=2, alpha=2.0) as hook:
            outputs = model.generate(input_ids, ...)
    """

    def __init__(
        self,
        model,
        layer: int = 18,
        kv_group: int = 2,
        alpha: float = 2.0,
        n_heads: int = 28,
        n_kv_heads: int = 4,
        head_dim: int = 128,
        max_interventions: Optional[int] = None,
    ):
        self.model = model
        self.layer = layer
        self.kv_group = kv_group
        self.alpha = alpha
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_interventions = max_interventions
        self.handle = None
        self._intervention_count = 0

        # Compute head range for this KV group
        heads_per_kv = n_heads // n_kv_heads  # 7
        self.h_start = kv_group * heads_per_kv
        self.h_end = self.h_start + heads_per_kv
        self.slice_start = self.h_start * head_dim
        self.slice_end = self.h_end * head_dim

    def __enter__(self):
        layers = get_model_layers(self.model)
        target_layer = layers[self.layer]
        o_proj = target_layer.self_attn.o_proj

        alpha = self.alpha
        s, e = self.slice_start, self.slice_end

        def hook_fn(module, inp, out):
            # Decision-only: skip after max_interventions
            if self.max_interventions is not None and self._intervention_count >= self.max_interventions:
                return out

            self._intervention_count += 1

            x = inp[0]  # (batch, seq, n_heads * head_dim)
            last_pos = x.shape[1] - 1

            # Scale the KV group's heads at last token position
            x_modified = x.clone()
            x_modified[0, last_pos, s:e] = x[0, last_pos, s:e] * alpha

            # Recompute o_proj output for last token with scaled input
            patched_out = module.weight @ x_modified[0, last_pos, :]
            if module.bias is not None:
                patched_out = patched_out + module.bias

            out_modified = out.clone()
            out_modified[0, last_pos, :] = patched_out
            return out_modified

        self.handle = o_proj.register_forward_hook(hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self._intervention_count = 0
        return False


class KVGroupDirectionalScalingHook:
    """
    Context manager that scales a KV group's contribution to the attention
    output decomposed along a reference direction (e.g. evidence_dir).

    At each intervention step, isolates the KV group's contribution to the
    o_proj output at the last token position:

        kv_contrib = W_O[:, s:e] @ x[0, last, s:e]

    Then projects it onto the reference direction e_hat and splits into:

        kv_parallel  = (kv_contrib · e_hat) * e_hat
        kv_orthogonal = kv_contrib - kv_parallel

    Replaces the group's contribution with:

        new_kv_contrib = alpha_parallel * kv_parallel + alpha_orth * kv_orthogonal

    alpha_parallel=alpha_orth=1.0 is the identity (no change).

    Usage:
        with KVGroupDirectionalScalingHook(
            model, layer=18, kv_group=2, direction=evidence_dir,
            alpha_parallel=1.0, alpha_orth=2.0,
        ) as hook:
            outputs = model.generate(...)
    """

    def __init__(
        self,
        model,
        layer: int = 18,
        kv_group: int = 2,
        direction: np.ndarray = None,
        alpha_parallel: float = 1.0,
        alpha_orth: float = 1.0,
        n_heads: int = 28,
        n_kv_heads: int = 4,
        head_dim: int = 128,
        max_interventions: Optional[int] = None,
    ):
        if direction is None:
            raise ValueError("direction (reference vector) is required")
        self.model = model
        self.layer = layer
        self.kv_group = kv_group
        self.alpha_parallel = alpha_parallel
        self.alpha_orth = alpha_orth
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_interventions = max_interventions
        self.handle = None
        self._intervention_count = 0

        heads_per_kv = n_heads // n_kv_heads
        self.h_start = kv_group * heads_per_kv
        self.h_end = self.h_start + heads_per_kv
        self.slice_start = self.h_start * head_dim
        self.slice_end = self.h_end * head_dim

        dir_arr = np.asarray(direction, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(dir_arr))
        if norm < 1e-8:
            raise ValueError("direction has near-zero norm")
        self._e_hat_np = (dir_arr / norm).astype(np.float32)
        self._e_hat = None

    def __enter__(self):
        layers = get_model_layers(self.model)
        target_layer = layers[self.layer]
        o_proj = target_layer.self_attn.o_proj

        s, e = self.slice_start, self.slice_end
        a_par = self.alpha_parallel
        a_orth = self.alpha_orth

        weight = o_proj.weight
        device = weight.device
        dtype = weight.dtype
        self._e_hat = torch.from_numpy(self._e_hat_np).to(device=device, dtype=dtype)

        def hook_fn(module, inp, out):
            if self.max_interventions is not None and self._intervention_count >= self.max_interventions:
                return out

            self._intervention_count += 1

            x = inp[0]
            last_pos = x.shape[1] - 1

            W_slice = module.weight[:, s:e]
            x_slice = x[0, last_pos, s:e]
            kv_contrib = W_slice @ x_slice

            e_hat = self._e_hat
            proj = torch.dot(kv_contrib, e_hat)
            kv_par = proj * e_hat
            kv_orth = kv_contrib - kv_par

            new_kv = a_par * kv_par + a_orth * kv_orth
            delta = new_kv - kv_contrib

            out_modified = out.clone()
            out_modified[0, last_pos, :] = out[0, last_pos, :] + delta
            return out_modified

        self.handle = o_proj.register_forward_hook(hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self._intervention_count = 0
        self._e_hat = None
        return False


class MultiLayerSteeringHook:
    """
    Apply steering across multiple layers with distributed rho.
    per_layer_rho = total_rho / sqrt(n_layers)
    """
    
    def __init__(
        self,
        model,
        direction: np.ndarray,
        total_alpha: float,
        layers: list,
        position: int = -1,
        mode: str = "addition"
    ):
        self.model = model
        self.direction = direction
        self.layers = layers
        self.position = position
        self.mode = mode
        
        # Distribute alpha across layers
        n_layers = len(layers)
        self.per_layer_alpha = total_alpha / np.sqrt(n_layers)
        self.hooks = []
        
    def __enter__(self):
        for layer in self.layers:
            hook = SteeringHook(
                self.model, self.direction, self.per_layer_alpha,
                layer=layer, position=self.position, mode=self.mode
            )
            hook.__enter__()
            self.hooks.append(hook)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for hook in self.hooks:
            hook.__exit__(exc_type, exc_val, exc_tb)
        self.hooks = []
        return False

