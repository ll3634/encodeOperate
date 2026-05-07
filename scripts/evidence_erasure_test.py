#!/usr/bin/env python3
"""Evidence Erasure Test — causal non-operativity of E vs A at L20.

Five conditions on each of N=100 §3 prompts at L20, last token:
  baseline    — no intervention (cross-checked against §3 cached baseline)
  erase_E     — h' = h - (h·Ê)Ê         (Ê = E/||E||)
  flip_E      — h' = h - 2(h·Ê)Ê
  erase_A     — h' = h - (h·Â)Â         (Â = A/||A||)
  flip_A      — h' = h - 2(h·Â)Â

NO RMS normalisation. Natural projection scale only. Margin =
logsumexp_logits(tool_ids) - logsumexp_logits(fin_ids) at last token,
matching §3 / OCFT exactly.
"""
import json, time
from pathlib import Path
import numpy as np
import sys, torch
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import get_model_layers

LAYER = 20
SEED = 20260502
N = 100
OUT = Path("results/evidence_erasure_test"); OUT.mkdir(parents=True, exist_ok=True)


class ProjectionFlipHook:
    """h' = h - factor * (h·ê) ê at LAYER, last-token, single-shot.
    factor=1.0 -> erase; factor=2.0 -> flip-sign-of-projection."""

    def __init__(self, model, direction, factor=1.0, layer=LAYER, max_interventions=1):
        self.model = model
        d = np.asarray(direction, np.float32).reshape(-1)
        n = float(np.linalg.norm(d))
        assert n > 1e-8, "direction has zero norm"
        self.unit_np = (d / n).astype(np.float32)
        self.factor = float(factor)
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
                self.unit = torch.tensor(self.unit_np,
                                         dtype=hidden.dtype,
                                         device=hidden.device)
            seq_len = hidden.shape[1]
            pos = seq_len - 1
            h = hidden[:, pos, :]
            proj = (h * self.unit).sum(dim=-1, keepdim=True) * self.unit
            hidden[:, pos, :] = h - self.factor * proj
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
        return False


def margin_from_logits(logits, tool_ids, fin_ids):
    lp = torch.log_softmax(logits.float(), dim=-1)
    return (torch.logsumexp(lp[tool_ids], 0) -
            torch.logsumexp(lp[fin_ids], 0)).item()


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


def build_p0_prompt(tok, q, query, obs):
    pb = PromptBuilder(tools=["search", "calculator"])
    msgs = pb.build_full_prompt(q, [{"action": "search", "action_input": query,
                                     "observation": obs[:1500]}])
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)


def boot_mean_ci(x, B=2000, level=95.0, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(x.mean()), float(lo), float(hi)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[init] L{LAYER} N={N}")
    E = np.load("results/phase1_probe/probe_direction_l20.npz",
                allow_pickle=True)["decision_direction"].astype(np.float32)
    A = np.load("steering/directions/direction_decomp_full_layer20.npz",
                allow_pickle=True)["decision_direction"].astype(np.float32)
    print(f"  ||E||={np.linalg.norm(E):.4f}  ||A||={np.linalg.norm(A):.4f}  "
          f"cos(E,A)={float(np.dot(E,A)/(np.linalg.norm(E)*np.linalg.norm(A))):+.4f}")

    label_data = [json.loads(l) for l in open("results/phase1_probe/labels.jsonl")]
    bl_map = {}
    with open("results/l20_rho020_n500/baseline_results.jsonl") as f:
        for line in f:
            ep = json.loads(line); bl_map[ep["sample_id"]] = ep

    print(f"\n[load] Qwen/Qwen2.5-7B-Instruct")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

    prompts, sample_ids, baseline_behavior = [], [], []
    for ld in label_data:
        ep = bl_map.get(ld["sample_id"])
        if not ep or not ep.get("steps"): continue
        s0 = ep["steps"][0]
        if not s0.get("observation"): continue
        prompts.append(build_p0_prompt(tok, ld["question"], s0["action_input"], s0["observation"]))
        sample_ids.append(ld["sample_id"])
        # Behavioral baseline from §3 cache: did the agent emit a SECOND search?
        # (len(steps)>=2 is misleading: step[1] is usually the Final Answer step.)
        s1_action = ep["steps"][1].get("action") if len(ep["steps"]) >= 2 else None
        baseline_behavior.append(int(s1_action == "search"))
        if len(prompts) >= N: break
    print(f"[prompts] N={len(prompts)}")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True).eval()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]
    print(f"[tokens] tool_ids={tool_ids} fin_ids={fin_ids}")

    cond_specs = [
        ("baseline", None),
        ("erase_E",  lambda: ProjectionFlipHook(model, E, factor=1.0)),
        ("flip_E",   lambda: ProjectionFlipHook(model, E, factor=2.0)),
        ("erase_A",  lambda: ProjectionFlipHook(model, A, factor=1.0)),
        ("flip_A",   lambda: ProjectionFlipHook(model, A, factor=2.0)),
    ]
    n = len(prompts)
    margins = {c: np.zeros(n, dtype=np.float32) for c, _ in cond_specs}
    t0 = time.time()
    for i, p in enumerate(prompts):
        for c, hf in cond_specs:
            margins[c][i] = forward_margin(model, tok, p, hf, tool_ids, fin_ids)
        if (i + 1) % 10 == 0 or i == 0:
            eta = (time.time() - t0) / (i + 1) * (n - i - 1)
            print(f"  [{i+1:>3d}/{n}]  ETA={eta:.0f}s  "
                  f"base[{i}]={margins['baseline'][i]:+.3f}")
    print(f"[done] forwards: {time.time() - t0:.0f}s")

    np.savez(OUT / "per_prompt_margins.npz",
             sample_ids=np.array(sample_ids),
             baseline_behavior=np.array(baseline_behavior, dtype=np.int8),
             **{c: margins[c] for c, _ in cond_specs})
    from evidence_erasure_io import analyse_and_write
    analyse_and_write(margins, sample_ids, baseline_behavior, OUT)


if __name__ == "__main__":
    main()
