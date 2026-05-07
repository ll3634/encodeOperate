#!/usr/bin/env python3
"""Mechanistic bridge: does task_missingness shift the L20 action channel?

For each (sample_id, target, cue) record in pairs.jsonl, capture at the
decision token:
  - residual_L20      : output of transformer block 20
  - mlp_L20_input     : output of post_attention_layernorm at L20 (MLP input)
  - mlp_L20_output    : raw MLP output at L20 (pre residual add)
and project each onto
  - action_dir        : decision_direction from direction_search_v3_layer20.npz
  - evidence_dir      : decision_direction from probe_direction_l20.npz
Also record the first-token search-stop margin as sanity check.

Per-sample, compute Δ = cue - neutral within target. Report target-conditional
mean ± 95% CI and permutation p-value on each projection. The claim is:
task_missingness at SF shifts the action-channel projection with a small or
null shift on the evidence-probe projection; distractor shows no action shift.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import get_model_layers


def load_dir(path, key="decision_direction"):
    d = np.load(path)
    v = d[key].astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def make_margin_ids(tokenizer):
    tool_ids = [tokenizer.encode(t, add_special_tokens=False)[0]
                for t in ACTION_TOKENS["tool_call"] if tokenizer.encode(t, add_special_tokens=False)]
    fin_ids  = [tokenizer.encode(t, add_special_tokens=False)[0]
                for t in ACTION_TOKENS["finish"]    if tokenizer.encode(t, add_special_tokens=False)]
    return tool_ids, fin_ids


def capture(model, tokenizer, prompt, device, tool_ids, fin_ids):
    layers = get_model_layers(model)
    cap = {}
    handles = []

    def h_layer(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        cap["residual_L20"] = h[0, -1, :].detach().float().cpu().numpy()

    def h_mlp_in(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        cap["mlp_L20_input"] = h[0, -1, :].detach().float().cpu().numpy()

    def h_mlp_out(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        cap["mlp_L20_output"] = h[0, -1, :].detach().float().cpu().numpy()

    handles.append(layers[20].register_forward_hook(h_layer))
    handles.append(layers[20].post_attention_layernorm.register_forward_hook(h_mlp_in))
    handles.append(layers[20].mlp.register_forward_hook(h_mlp_out))

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(input_ids)
    for h in handles:
        h.remove()

    lp = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
    tl = torch.logsumexp(lp[tool_ids], 0).item() if tool_ids else -100.0
    fl = torch.logsumexp(lp[fin_ids],  0).item() if fin_ids  else -100.0
    cap["margin"] = float(tl - fl)
    return cap


def perm_two(x, n=20000, seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    null = (rng.choice([-1.0, 1.0], size=(n, len(x))) * x).mean(axis=1)
    return float((np.abs(null) >= abs(x.mean())).mean())


def boot_ci(x, n=20000, seed=1):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def summarize(name, x):
    x = np.asarray(x)
    m, lo, hi = boot_ci(x)
    return {"name": name, "n": int(len(x)),
            "mean": m, "median": float(np.median(x)),
            "ci95": [lo, hi], "perm_p_two_sided": perm_two(x),
            "n_pos": int((x > 0).sum()), "n_neg": int((x < 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/anti_cue_tm_n100/pairs.jsonl")
    ap.add_argument("--out",   default="results/anti_cue_tm_n100/l20_bridge.json")
    ap.add_argument("--projections", default="results/anti_cue_tm_n100/l20_bridge_projections.jsonl")
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--action-dir", default="steering/directions/direction_search_v3_layer20.npz")
    ap.add_argument("--evidence-dir", default="results/phase1_probe/probe_direction_l20.npz")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    action_dir_raw = load_dir(args.action_dir,   key="decision_direction")
    evidence_dir   = load_dir(args.evidence_dir, key="decision_direction")
    # sign convention: orient action_dir so that positive projection => search-leaning
    # by checking which sign gives positive (h_high_margin - h_low_margin) · action_dir.
    ad = np.load(args.action_dir)
    hhi = ad["h_high_margin_mean"].astype(np.float32)
    hlo = ad["h_low_margin_mean"].astype(np.float32)
    sign = 1.0 if float(np.dot(hhi - hlo, action_dir_raw)) > 0 else -1.0
    action_dir = sign * action_dir_raw
    proj_hi_lo = float(np.dot(hhi - hlo, action_dir))
    cos_ae = float(np.dot(action_dir, evidence_dir))
    print(f"[dir] sign_flip = {sign:+.0f}  (raw decision_direction points toward "
          f"{'search' if sign > 0 else 'stop'}; using +search convention)")
    print(f"[dir] (h_high - h_low) . action_dir = {proj_hi_lo:+.4f}  (positive => search-leaning)")
    print(f"[dir] cos(action, evidence) = {cos_ae:+.4f}")

    records = [json.loads(l) for l in open(args.pairs)]
    if args.limit:
        records = records[:args.limit]
    print(f"[info] {len(records)} records")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype,
                                                 device_map="auto", trust_remote_code=True)
    model.eval(); device = next(model.parameters()).device
    tool_ids, fin_ids = make_margin_ids(tok)
    builder = PromptBuilder()

    rows = []
    t0 = time.time()
    for i, r in enumerate(records):
        steps = [{"action": "search", "action_input": f"about: {r['question'][:80]}",
                  "observation": r["obs"]}]
        messages = builder.build_full_prompt(r["question"], steps)
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        cap = capture(model, tok, prompt, device, tool_ids, fin_ids)
        row = {
            "sample_id": r["sample_id"], "target": r["target"], "cue": r["cue"],
            "condition_id": r["condition_id"], "margin": cap["margin"],
            "proj_residL20_action":    float(np.dot(cap["residual_L20"],    action_dir)),
            "proj_residL20_evidence":  float(np.dot(cap["residual_L20"],    evidence_dir)),
            "proj_mlpL20in_action":    float(np.dot(cap["mlp_L20_input"],  action_dir)),
            "proj_mlpL20in_evidence":  float(np.dot(cap["mlp_L20_input"],  evidence_dir)),
            "proj_mlpL20out_action":   float(np.dot(cap["mlp_L20_output"], action_dir)),
            "proj_mlpL20out_evidence": float(np.dot(cap["mlp_L20_output"], evidence_dir)),
        }
        rows.append(row)
        if (i + 1) % 50 == 0 or i + 1 == len(records):
            print(f"  [{i+1}/{len(records)}] {time.time()-t0:.1f}s")

    Path(args.projections).parent.mkdir(parents=True, exist_ok=True)
    with open(args.projections, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

    sids = sorted(set(r["sample_id"] for r in rows))
    pivot = {s: {t: {} for t in ("sf", "distractor")} for s in sids}
    for r in rows:
        pivot[r["sample_id"]][r["target"]][r["cue"]] = r

    METRICS = ["margin",
               "proj_residL20_action", "proj_residL20_evidence",
               "proj_mlpL20in_action", "proj_mlpL20in_evidence",
               "proj_mlpL20out_action", "proj_mlpL20out_evidence"]

    def deltas(target, metric):
        xs = []
        for s in sids:
            if "neutral" in pivot[s][target] and "task_missingness" in pivot[s][target]:
                xs.append(pivot[s][target]["task_missingness"][metric]
                          - pivot[s][target]["neutral"][metric])
        return np.array(xs)

    summary = {"cos_action_evidence": cos_ae,
               "sign_check_high_minus_low_dot_action": proj_hi_lo,
               "per_metric": {}, "locality_interaction": {}}
    print(f'\n=== Δ (task_missingness − neutral) per metric, by target ===')
    print(f'{"metric":30s} {"target":10s}  N   mean     CI95                 p2')
    for met in METRICS:
        for tgt in ("sf", "distractor"):
            d = deltas(tgt, met)
            s = summarize(f"Δ{met} | {tgt}", d)
            summary["per_metric"][f"{met}|{tgt}"] = s
            print(f'  {met:28s} {tgt:10s} {s["n"]:3d} {s["mean"]:+7.3f}  '
                  f'[{s["ci95"][0]:+6.3f},{s["ci95"][1]:+6.3f}]  p2={s["perm_p_two_sided"]:.4g}')
        d_sf = deltas("sf", met); d_ds = deltas("distractor", met)
        inter = d_sf - d_ds
        s = summarize(f"locality[{met}] = SF-effect - dist-effect", inter)
        summary["locality_interaction"][met] = s
        print(f'  {met:28s} {"INTER":10s} {s["n"]:3d} {s["mean"]:+7.3f}  '
              f'[{s["ci95"][0]:+6.3f},{s["ci95"][1]:+6.3f}]  p2={s["perm_p_two_sided"]:.4g}')

    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\n[wrote] {args.out}\n[wrote] {args.projections}")


if __name__ == "__main__":
    main()
