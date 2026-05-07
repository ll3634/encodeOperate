#!/usr/bin/env python3
"""
Action-Direction Construct-Validity Audit v2 (Qwen2.5-7B, L20)
===============================================================
Reviewer attack: action_v1 (steering/directions/direction_search_v3_layer20.npz)
was extracted via h(low_margin) − h(high_margin) on PopQA, where margin
co-varies with parametric extractability. Decomposing it against the evidence
direction may be circular if the extraction conflates extractability with
action.

Three alternative extractions, each on a different methodology:
  A. action_v2_within   — within-T0, first_action=search vs first_action=stop
                          difference-of-means (with documented fallback if T0
                          has zero search-class examples)
  B. action_v3_logit    — gradient of margin (logsumexp Action − logsumexp Final)
                          w.r.t. L20 hidden at the decision token, averaged
                          across prompts; sign-flipped to match v1 ("stop" pole)
  C. action_v4_crossval — K=20 random unit-RMS dirs ranked on split A by mean
                          Δmargin under ρ=−0.20 injection; top-1 is v4;
                          decomposition evaluated on held-out split B

For each variant: cos(v, evidence_dir_L20), §8.3 functional decomposition
(full / parallel / perpendicular / random K=20) on split B (HotpotQA p0),
and McNemar test on margin-sign flip vs baseline.

Pass: parallel within random null band AND perp within 15% of full shift.
≥ 2 of 3 must reproduce dissociation.

Usage: python scripts/action_dir_construct_validity_v2.py
"""

import os, sys, json, argparse, time
import numpy as np
from pathlib import Path

import torch
from scipy.stats import binomtest, percentileofscore

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import SteeringHook, get_model_layers
from steering.directions import load_direction

LAYER = 20
RHO = -0.20
SEED = 20260429


# ─── Generic helpers ─────────────────────────────────────────────────────────

def margin_from_logits(logits, tool_ids, fin_ids):
    lp = torch.log_softmax(logits, dim=-1)
    return (torch.logsumexp(lp[tool_ids], 0) -
            torch.logsumexp(lp[fin_ids], 0)).item()


def normalize_rms(d, target=1.0):
    rms = float(np.sqrt(np.mean(d ** 2)))
    return d * (target / rms) if rms > 1e-12 else d


def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def gen_random_dirs(dim, n, seed):
    rng = np.random.RandomState(seed)
    return [normalize_rms(rng.randn(dim).astype(np.float32), 1.0) for _ in range(n)]


def decompose_against(v, e_unit):
    par = float(np.dot(v, e_unit)) * e_unit
    perp = v - par
    return par.astype(np.float32), perp.astype(np.float32)


def mcnemar(baseline_pos, steered_pos):
    """Returns (b=rescue, c=regression, p_two_sided). Exact binomial."""
    b = int(np.sum(steered_pos & ~baseline_pos))
    c = int(np.sum(~steered_pos & baseline_pos))
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = float(binomtest(min(b, c), n, p=0.5, alternative="two-sided").pvalue)
    return b, c, p


def build_p0_prompt(tokenizer, question, query, observation):
    pb = PromptBuilder(tools=["search", "calculator"])
    steps = [{"action": "search", "action_input": query,
              "observation": observation[:1500]}]
    msgs = pb.build_full_prompt(question, steps)
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)


def build_t0_prompt(tokenizer, rec):
    return build_p0_prompt(tokenizer, rec["question"],
                           rec["question"], rec["observation"])


# ─── Forward / capture utilities ─────────────────────────────────────────────

def capture_hidden_margin_top1(model, tokenizer, prompt, layer,
                                tool_ids, fin_ids):
    """One forward; returns (h_L20[-1], margin, top1_token_id)."""
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    layers = get_model_layers(model)
    target = layers[layer]
    captured = {}

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h[0, -1, :].detach().float().cpu().numpy().copy()

    handle = target.register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1, :]
    finally:
        handle.remove()
    m = margin_from_logits(logits, tool_ids, fin_ids)
    top1 = int(torch.argmax(logits).item())
    return captured["h"], m, top1


def measure_margin(model, tokenizer, prompt, direction, rho, layer,
                   tool_ids, fin_ids):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if direction is not None and abs(rho) > 1e-8:
        d_rms = float(np.sqrt(np.mean(direction ** 2)))
        hidden_rms = 0.65
        alpha = rho * (hidden_rms / d_rms)
        with SteeringHook(model, direction, alpha, layer=layer,
                          position=-1, max_interventions=1):
            with torch.no_grad():
                logits = model(input_ids).logits[0, -1, :]
    else:
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1, :]
    return margin_from_logits(logits, tool_ids, fin_ids)


def grad_margin_wrt_l20(model, tokenizer, prompt, layer, tool_ids, fin_ids):
    """Backward pass: returns d(margin)/d(h_L{layer}[-1]) as a unit-RMS direction
    contribution for this prompt (no normalization here; caller averages)."""
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    layers = get_model_layers(model)
    target = layers[layer]
    captured = {}

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h.retain_grad()
        captured["h"] = h

    handle = target.register_forward_hook(hook)
    try:
        # No torch.no_grad here — we need grads
        logits = model(input_ids).logits[0, -1, :]
        lp = torch.log_softmax(logits, dim=-1)
        margin = (torch.logsumexp(lp[tool_ids], 0) -
                  torch.logsumexp(lp[fin_ids], 0))
        model.zero_grad(set_to_none=True)
        margin.backward()
        g = captured["h"].grad[0, -1, :].detach().float().cpu().numpy().copy()
    finally:
        handle.remove()
    return g  # gradient toward INCREASING margin (= toward search)


# ─── Variant extractors ──────────────────────────────────────────────────────

def extract_v2_within_t0(hiddens, margins, top1_ids, action_id, final_id,
                          min_per_class=3):
    """Spec: first_action=search vs first_action=stop within T0, diff-of-means.
    Fallbacks (in order, documented in report):
      1. top1 token in {Action, Final} with ≥min_per_class each
      2. sign(margin) with ≥min_per_class each
      3. top-third vs bottom-third by margin
    Returns (v2, method_used, n_search, n_stop)."""
    is_action = (top1_ids == action_id)
    is_final = (top1_ids == final_id)
    if is_action.sum() >= min_per_class and is_final.sum() >= min_per_class:
        h_search = hiddens[is_action].mean(axis=0)
        h_stop = hiddens[is_final].mean(axis=0)
        v = h_stop - h_search  # +v points toward stop (matches v1)
        return v.astype(np.float32), "top1_token", int(is_action.sum()), int(is_final.sum())
    pos = (margins > 0)
    neg = (margins < 0)
    if pos.sum() >= min_per_class and neg.sum() >= min_per_class:
        h_search = hiddens[pos].mean(axis=0)
        h_stop = hiddens[neg].mean(axis=0)
        v = h_stop - h_search
        return v.astype(np.float32), "sign_margin", int(pos.sum()), int(neg.sum())
    order = np.argsort(margins)
    k = max(min_per_class, len(order) // 3)
    bot = order[:k]; top = order[-k:]
    h_search = hiddens[top].mean(axis=0)  # top margin = closer to search
    h_stop = hiddens[bot].mean(axis=0)
    v = h_stop - h_search
    return v.astype(np.float32), "margin_thirds_fallback", int(len(top)), int(len(bot))


def extract_v3_logit_jacobian(model, tokenizer, prompts, layer,
                               tool_ids, fin_ids):
    """Average d(margin)/d(h_L{layer}[-1]) across prompts. Sign-flip so +v
    points toward stop (matching v1's convention)."""
    grads = []
    for p in prompts:
        g = grad_margin_wrt_l20(model, tokenizer, p, layer, tool_ids, fin_ids)
        grads.append(g)
    mean_g = np.mean(np.stack(grads), axis=0).astype(np.float32)
    # gradient of margin wrt h points toward INCREASING margin (= search).
    # v1 convention: +v toward stop. So flip sign.
    return (-mean_g).astype(np.float32)


def extract_v4_crossval(model, tokenizer, split_a_prompts, baseline_a,
                         random_dirs, layer, rho, tool_ids, fin_ids):
    """Cross-validated steering direction.
    On split A under ρ=−0.20: rank K random dirs by mean Δmargin.
    Per spec ('ρ=0.20 maximizes 2nd-search rate'): with ρ=−0.20 in this
    codebase, max +Δmargin == direction whose negative-scaled injection
    pushes toward search == direction itself points toward stop. v1 sign.
    Returns (v4, ranked_idx, ranked_shifts, top_idx)."""
    n_dirs = len(random_dirs)
    shifts = np.zeros((n_dirs, len(split_a_prompts)), dtype=np.float32)
    for di, d in enumerate(random_dirs):
        for pi, p in enumerate(split_a_prompts):
            m = measure_margin(model, tokenizer, p, d, rho, layer,
                               tool_ids, fin_ids)
            shifts[di, pi] = m - baseline_a[pi]
    mean_shifts = shifts.mean(axis=1)
    order = np.argsort(-mean_shifts)  # descending
    top_idx = int(order[0])
    v4 = np.array(random_dirs[top_idx], dtype=np.float32)
    return v4, order.tolist(), mean_shifts.tolist(), top_idx


# ─── Main ────────────────────────────────────────────────────────────────────

def run_decomp(model, tokenizer, vname, vfull, e_unit, eval_prompts,
               eval_baseline, random_dirs, layer, rho, tool_ids, fin_ids,
               action_id, final_id):
    """Inject full / parallel / perp / each random dir; collect Δmargin and
    margin-sign flips. Returns dict with all numbers."""
    par, perp = decompose_against(vfull, e_unit)
    full_n = normalize_rms(vfull, 1.0)
    par_norm_pre = float(np.sqrt(np.mean(par ** 2)))
    par_n = normalize_rms(par, 1.0) if par_norm_pre > 1e-8 else par.astype(np.float32)
    perp_n = normalize_rms(perp, 1.0)

    def shifts_and_signs(direction):
        sh, post_pos = [], []
        for pi, p in enumerate(eval_prompts):
            m = measure_margin(model, tokenizer, p, direction, rho, layer,
                               tool_ids, fin_ids)
            sh.append(m - eval_baseline[pi])
            post_pos.append(m > 0)
        return np.array(sh, dtype=np.float32), np.array(post_pos)

    base_pos = np.array([m > 0 for m in eval_baseline])
    full_sh, full_pos = shifts_and_signs(full_n)
    par_sh, par_pos = shifts_and_signs(par_n)
    perp_sh, perp_pos = shifts_and_signs(perp_n)

    full_b, full_c, full_p = mcnemar(base_pos, full_pos)
    par_b, par_c, par_p = mcnemar(base_pos, par_pos)
    perp_b, perp_c, perp_p = mcnemar(base_pos, perp_pos)

    rand_means, rand_abs_means = [], []
    for d in random_dirs:
        rs, _ = shifts_and_signs(d)
        rand_means.append(float(rs.mean()))
        rand_abs_means.append(float(np.mean(np.abs(rs))))
    rand_means = np.array(rand_means)
    rand_abs_means = np.array(rand_abs_means)

    # Pass criterion: parallel within random null, perp within 15% of full
    par_in_null = (np.abs(par_sh).mean() <= np.percentile(rand_abs_means, 97.5))
    full_abs = abs(full_sh.mean())
    perp_abs = abs(perp_sh.mean())
    perp_recovers = (full_abs > 1e-6
                     and np.sign(perp_sh.mean()) == np.sign(full_sh.mean())
                     and abs(perp_abs - full_abs) / full_abs <= 0.15)

    return {
        "cos_with_evidence": cos_sim(vfull, e_unit),
        "parallel_norm_pre_renorm": par_norm_pre,
        "full": {"mean_shift": float(full_sh.mean()),
                 "abs_mean_shift": float(np.mean(np.abs(full_sh))),
                 "mcnemar_b_rescue": full_b, "mcnemar_c_regression": full_c,
                 "mcnemar_p": full_p},
        "parallel": {"mean_shift": float(par_sh.mean()),
                     "abs_mean_shift": float(np.mean(np.abs(par_sh))),
                     "mcnemar_b_rescue": par_b, "mcnemar_c_regression": par_c,
                     "mcnemar_p": par_p},
        "perpendicular": {"mean_shift": float(perp_sh.mean()),
                          "abs_mean_shift": float(np.mean(np.abs(perp_sh))),
                          "mcnemar_b_rescue": perp_b,
                          "mcnemar_c_regression": perp_c,
                          "mcnemar_p": perp_p},
        "random_null": {
            "n": len(random_dirs),
            "signed_mean": float(rand_means.mean()),
            "signed_std": float(rand_means.std()),
            "abs_mean": float(rand_abs_means.mean()),
            "abs_std": float(rand_abs_means.std()),
            "abs_p97_5": float(np.percentile(rand_abs_means, 97.5)),
        },
        "parallel_pctile_signed_in_random": float(
            percentileofscore(rand_means, par_sh.mean())),
        "parallel_pctile_abs_in_random": float(
            percentileofscore(rand_abs_means, np.abs(par_sh).mean())),
        "parallel_in_random_null": bool(par_in_null),
        "perp_within_15pct_of_full": bool(perp_recovers),
        "dissociation_holds": bool(par_in_null and perp_recovers),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=100,
                    help="HotpotQA p0 prompts (split 50/50 train/eval for v4)")
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--pairs-path",
                    default="results/extractability_support_toggle/pairs.jsonl")
    ap.add_argument("--baseline-trace",
                    default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--labels-path",
                    default="results/phase1_probe/labels.jsonl")
    ap.add_argument("--output-dir",
                    default="results/action_direction_construct_validity")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[init] L{LAYER} rho={RHO} seed={SEED} K_random={args.n_random}"
          f" n_eval={args.n_eval}")

    v1, _ = load_direction(
        "steering/directions/direction_search_v3_layer20.npz",
        key="decision_direction_normalized")
    v1 = normalize_rms(v1.astype(np.float32), 1.0)
    e_raw = np.load("results/phase1_probe/probe_direction_l20.npz")["decision_direction"]
    e_unit = (e_raw / np.linalg.norm(e_raw)).astype(np.float32)
    dim = v1.shape[0]
    print(f"[init] cos(v1,evidence)={cos_sim(v1,e_unit):+.4f} dim={dim}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"[load] {name}")
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    tool_ids = [tok.encode(t, add_special_tokens=False)[0]
                for t in ACTION_TOKENS["tool_call"]]
    fin_ids = [tok.encode(t, add_special_tokens=False)[0]
               for t in ACTION_TOKENS["finish"]]
    action_id, final_id = tool_ids[0], fin_ids[0]
    print(f"[init] action_token_id={action_id}  final_token_id={final_id}")


    # ── PHASE 1: T0 hidden capture for v2 + first_action labels ────────────
    print(f"\n[phase1] T0 capture (extractability fixed)")
    pairs = [json.loads(l) for l in open(args.pairs_path)]
    t0 = [r for r in pairs if r["condition"] == "T0"]
    t0_h, t0_m, t0_top1 = [], [], []
    t0_t = time.time()
    for i, rec in enumerate(t0):
        prompt = build_t0_prompt(tok, rec)
        h, m, t1 = capture_hidden_margin_top1(model, tok, prompt, LAYER,
                                               tool_ids, fin_ids)
        t0_h.append(h); t0_m.append(m); t0_top1.append(t1)
        if (i + 1) % 10 == 0:
            print(f"  T0 [{i+1}/{len(t0)}] {time.time()-t0_t:.0f}s")
    t0_h = np.stack(t0_h).astype(np.float32)
    t0_m = np.array(t0_m, dtype=np.float32)
    t0_top1 = np.array(t0_top1, dtype=np.int64)
    n_search = int((t0_top1 == action_id).sum())
    n_stop = int((t0_top1 == final_id).sum())
    n_other = int(len(t0_top1) - n_search - n_stop)
    print(f"[phase1] T0 first_action: search={n_search} stop={n_stop} other={n_other}"
          f"  margin range=[{t0_m.min():.2f},{t0_m.max():.2f}]")

    v2, v2_method, v2_n_search, v2_n_stop = extract_v2_within_t0(
        t0_h, t0_m, t0_top1, action_id, final_id)
    v2 = normalize_rms(v2, 1.0)
    print(f"[v2] method={v2_method} n_search={v2_n_search} n_stop={v2_n_stop}")

    # ── PHASE 2: Build HotpotQA p0 eval prompts; split A/B ─────────────────
    print(f"\n[phase2] HotpotQA p0 prompts")
    label_data = [json.loads(l) for l in open(args.labels_path)]
    bl_map = {}
    with open(args.baseline_trace) as f:
        for line in f:
            ep = json.loads(line)
            bl_map[ep["sample_id"]] = ep
    eval_prompts = []
    for ld in label_data:
        ep = bl_map.get(ld["sample_id"])
        if not ep or not ep.get("steps"):
            continue
        s0 = ep["steps"][0]
        if s0.get("action") != "search" or not s0.get("observation"):
            continue
        eval_prompts.append(build_p0_prompt(
            tok, ld["question"], s0["action_input"], s0["observation"]))
        if len(eval_prompts) >= args.n_eval:
            break
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(eval_prompts))
    half = len(eval_prompts) // 2
    idx_a = perm[:half]; idx_b = perm[half:2 * half]
    prompts_a = [eval_prompts[i] for i in idx_a]
    prompts_b = [eval_prompts[i] for i in idx_b]
    print(f"[phase2] |A|={len(prompts_a)}  |B|={len(prompts_b)}")

    # Baselines for both splits
    print(f"[phase2] baselines on split A and B")
    base_a = [measure_margin(model, tok, p, None, 0.0, LAYER, tool_ids, fin_ids)
              for p in prompts_a]
    base_b = [measure_margin(model, tok, p, None, 0.0, LAYER, tool_ids, fin_ids)
              for p in prompts_b]

    # ── PHASE 3: Variant B (logit Jacobian) on T0 prompts ──────────────────
    print(f"\n[phase3] v3 logit-Jacobian on T0 ({len(t0)} backward passes)")
    t3 = time.time()
    t0_prompts = [build_t0_prompt(tok, r) for r in t0]
    v3 = extract_v3_logit_jacobian(model, tok, t0_prompts, LAYER,
                                    tool_ids, fin_ids)
    v3 = normalize_rms(v3, 1.0)
    print(f"[v3] mean-grad-direction extracted in {time.time()-t3:.0f}s")

    # ── PHASE 4: Variant C (cross-validated steering) ──────────────────────
    print(f"\n[phase4] v4 cross-val: K={args.n_random} random dirs on split A")
    rand_dirs_train = gen_random_dirs(dim, args.n_random, SEED)
    t4 = time.time()
    v4, v4_order, v4_shifts, v4_top_idx = extract_v4_crossval(
        model, tok, prompts_a, base_a, rand_dirs_train, LAYER, RHO,
        tool_ids, fin_ids)
    v4 = normalize_rms(v4, 1.0)
    print(f"[v4] split-A best idx={v4_top_idx} mean Δm={v4_shifts[v4_top_idx]:+.3f}"
          f"  ({time.time()-t4:.0f}s)")

    # ── PHASE 5: §8.3 decomposition for each variant on split B ────────────
    print(f"\n[phase5] §8.3 decomposition on split B (held-out for v4)")
    rand_dirs_eval = gen_random_dirs(dim, args.n_random, SEED + 1)
    decomp = {}
    for vname, vfull in [("v1_reference", v1),
                         ("v2_within_t0", v2),
                         ("v3_logit_jacobian", v3),
                         ("v4_crossval", v4)]:
        t5 = time.time()
        decomp[vname] = run_decomp(
            model, tok, vname, vfull, e_unit, prompts_b, base_b,
            rand_dirs_eval, LAYER, RHO, tool_ids, fin_ids,
            action_id, final_id)
        d = decomp[vname]
        print(f"  [{vname}] cos(e)={d['cos_with_evidence']:+.4f}"
              f" full={d['full']['mean_shift']:+.3f}"
              f" par={d['parallel']['mean_shift']:+.3f}"
              f" perp={d['perpendicular']['mean_shift']:+.3f}"
              f" -> diss={d['dissociation_holds']}"
              f" (mcnemar full p={d['full']['mcnemar_p']:.3g})"
              f" ({time.time()-t5:.0f}s)")

    # ── Save artifacts ──────────────────────────────────────────────────────
    audited = {k: v for k, v in decomp.items() if k != "v1_reference"}
    n_pass = sum(1 for r in audited.values() if r["dissociation_holds"])
    requirement_met = n_pass >= 2
    summary = {
        "config": {
            "layer": LAYER, "rho": RHO, "seed": SEED, "dim": dim,
            "n_t0": int(len(t0)), "n_eval": int(len(eval_prompts)),
            "split_a": int(len(prompts_a)), "split_b": int(len(prompts_b)),
            "n_random": args.n_random,
            "model": "Qwen/Qwen2.5-7B-Instruct",
        },
        "extraction_methods": {
            "v1_reference":
                "h(low_margin) − h(high_margin) over PopQA, "
                "from steering/directions/direction_search_v3_layer20.npz",
            "v2_within_t0":
                f"within-T0 first_action contrast, method={v2_method} "
                f"(n_search={v2_n_search}, n_stop={v2_n_stop})",
            "v3_logit_jacobian":
                f"mean over T0 ({len(t0_prompts)} prompts) of d(margin)/dh_L20 "
                f"at decision token, sign-flipped to match v1 (+ → stop)",
            "v4_crossval":
                f"top-1 of K={args.n_random} random unit-RMS dirs by mean Δmargin "
                f"on split A (n={len(prompts_a)}, ρ={RHO}); evaluated on split B",
        },
        "reference_v1_cos_with_evidence": cos_sim(v1, e_unit),
        "phase1_t0": {
            "n": int(len(t0)),
            "first_action_search": n_search, "first_action_stop": n_stop,
            "first_action_other": n_other,
            "margin_min": float(t0_m.min()), "margin_max": float(t0_m.max()),
            "margin_median": float(np.median(t0_m)),
            "v2_extraction_method": v2_method,
        },
        "v4_extraction": {
            "best_random_idx": v4_top_idx,
            "best_random_mean_shift": float(v4_shifts[v4_top_idx]),
            "split_a_random_shifts": v4_shifts,
        },
        "decomposition": decomp,
        "requirement": {
            "rule": (">=2 of 3 audited variants reproduce parallel-inert "
                     "/ perp-within-15%-of-full"),
            "n_dissociation_holds": n_pass,
            "requirement_met": bool(requirement_met),
        },
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "per_variant_decomposition.json"), "w") as f:
        json.dump(decomp, f, indent=2)
    np.savez(os.path.join(args.output_dir, "directions.npz"),
             v1=v1, v2_within_t0=v2, v3_logit_jacobian=v3,
             v4_crossval=v4, evidence_unit=e_unit)
    print(f"\n[done] requirement_met={requirement_met} ({n_pass}/3)")
    print(f"[done] -> {args.output_dir}/{{summary.json,per_variant_decomposition.json,directions.npz}}")


if __name__ == "__main__":
    main()
