#!/usr/bin/env python3
"""
Action-Direction Construct-Validity Audit (Qwen2.5-7B, L20)
============================================================
Reviewer attack: action_v1 = h(low_margin) − h(high_margin) on PopQA may
conflate extractability with action.

Three robustness extractions, all compared against action_v1 and decomposed
against evidence_dir_L20 per the §8.3 protocol:

  A. action_v2_within_t0  — within-T0 margin contrast (extractability fixed)
  B. action_v3_pca        — PCA on T0 residuals, top component max-correlated
                            with first-action sign(margin)
  C. action_v4_steering   — top-K random directions ranked by signed Δmargin
                            on the §8.3 eval set, orthogonalized vs null mean

For each variant: report cos(v, evidence_dir), cos(v, action_v1), and the
functional decomposition mean shifts.

Hard requirement: ≥2/3 variants reproduce parallel-inert / perp-recovers-full.

Usage: python scripts/action_dir_construct_validity.py
"""

import os, sys, json, argparse, time
import numpy as np
from pathlib import Path

import torch
from scipy.stats import percentileofscore

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS
from steering.hook_utils import SteeringHook
from steering.directions import load_direction

LAYER = 20
RHO = -0.20
SEED = 20260429


def get_margin(logits, tool_ids, fin_ids):
    log_probs = torch.log_softmax(logits, dim=-1)
    return (torch.logsumexp(log_probs[tool_ids], 0) -
            torch.logsumexp(log_probs[fin_ids], 0)).item()


def normalize_rms(d, target=1.0):
    rms = float(np.sqrt(np.mean(d ** 2)))
    return d * (target / rms) if rms > 1e-12 else d


def gen_random_dirs(dim, n, seed):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        d = rng.randn(dim).astype(np.float32)
        out.append(normalize_rms(d, 1.0))
    return out


def build_t0_prompt(tokenizer, rec):
    pb = PromptBuilder(tools=["search", "calculator"])
    steps = [{"action": "search", "action_input": rec["question"],
              "observation": rec["observation"][:1500]}]
    msgs = pb.build_full_prompt(rec["question"], steps)
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)


def build_p0_prompt(tokenizer, question, query, observation):
    pb = PromptBuilder(tools=["search", "calculator"])
    steps = [{"action": "search", "action_input": query,
              "observation": observation[:1500]}]
    msgs = pb.build_full_prompt(question, steps)
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)


def capture_hidden_and_margin(model, tokenizer, prompt, layer, tool_ids, fin_ids):
    """Single forward; capture L{layer} hidden at last token + margin."""
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    captured = {}
    from steering.hook_utils import get_model_layers
    layers = get_model_layers(model)
    target = layers[layer]

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h[0, -1, :].detach().float().cpu().numpy().copy()

    handle = target.register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1, :]
    finally:
        handle.remove()
    m = get_margin(logits, tool_ids, fin_ids)
    return captured["h"], m


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
    return get_margin(logits, tool_ids, fin_ids)


def extract_v2_within_t0(hiddens, margins):
    """Bottom-third (stop-leaning) − Top-third (search-leaning) within T0."""
    order = np.argsort(margins)
    k = len(order) // 3
    bot_idx = order[:k]
    top_idx = order[-k:]
    h_bot = hiddens[bot_idx].mean(axis=0)
    h_top = hiddens[top_idx].mean(axis=0)
    return (h_bot - h_top).astype(np.float32), bot_idx, top_idx


def extract_v3_pca(hiddens, margins, n_components=5):
    """Top PCA component on T0 residuals, signed by max corr with sign(margin)."""
    X = hiddens - hiddens.mean(axis=0, keepdims=True)
    # SVD-based PCA
    U, S, Vt = np.linalg.svd(X.astype(np.float32), full_matrices=False)
    comps = Vt[:n_components]  # (K, d)
    labels = np.sign(margins).astype(np.float32)
    # Avoid degenerate label (all same sign) — fall back to standardised margin
    if len(np.unique(labels)) < 2:
        labels = (margins - margins.mean()) / (margins.std() + 1e-8)
    scores = X @ comps.T  # (N, K)
    # Pick the component most correlated with action label
    best_k, best_abs_r, best_r = 0, -1.0, 0.0
    for k in range(n_components):
        r = float(np.corrcoef(scores[:, k], labels)[0, 1])
        if abs(r) > best_abs_r:
            best_abs_r = abs(r)
            best_k = k
            best_r = r
    v = comps[best_k]
    # Sign so +v matches "stop" direction (negative correlation with margin)
    if best_r > 0:
        v = -v
    return v.astype(np.float32), int(best_k), float(best_r)


def extract_v4_steering(eval_margins_random, random_dirs, top_k=5):
    """Top-K random dirs by signed Δmargin; orth against null mean; unit RMS.

    eval_margins_random: dict {idx: list of Δmargin per prompt}
    random_dirs: list of unit-RMS dirs.
    Convention: under rho=-0.20, +Δmargin means injection of -0.20*d shifted
    margin toward search. Hence d itself points toward "stop" — same sign as v1.
    Rank descending by mean signed Δmargin.
    """
    mean_shifts = np.array([np.mean(eval_margins_random[i])
                            for i in range(len(random_dirs))])
    order = np.argsort(-mean_shifts)  # descending
    top_idx = order[:top_k]
    top_dirs = np.stack([random_dirs[i] for i in top_idx])
    avg = top_dirs.mean(axis=0)
    # Orthogonalize against random null mean (across ALL random dirs)
    null_mean = np.stack(random_dirs).mean(axis=0)
    nm_norm2 = float(np.dot(null_mean, null_mean))
    if nm_norm2 > 1e-12:
        avg = avg - (np.dot(avg, null_mean) / nm_norm2) * null_mean
    return normalize_rms(avg.astype(np.float32), 1.0), top_idx.tolist(), \
        mean_shifts[top_idx].tolist()


def decompose_against(v, e):
    """Return (parallel, perp) of v wrt evidence direction e (any norm)."""
    e_unit = e / (np.linalg.norm(e) + 1e-12)
    par = float(np.dot(v, e_unit)) * e_unit
    perp = v - par
    return par.astype(np.float32), perp.astype(np.float32)


def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=50,
                    help="HotpotQA p0 prompts for variant-C extraction + decomp")
    ap.add_argument("--n-random", type=int, default=30,
                    help="Random dirs for variant-C ranking + null in decomp")
    ap.add_argument("--top-k", type=int, default=5)
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
    print(f"[init] layer={LAYER} rho={RHO} seed={SEED}")
    print(f"[init] n_eval={args.n_eval} n_random={args.n_random} top_k={args.top_k}")

    # ── Load reference directions ────────────────────────────────────────────
    v1, _ = load_direction(
        "steering/directions/direction_search_v3_layer20.npz",
        key="decision_direction_normalized")
    v1 = normalize_rms(v1.astype(np.float32), 1.0)
    e_raw = np.load("results/phase1_probe/probe_direction_l20.npz")["decision_direction"]
    e_unit = (e_raw / np.linalg.norm(e_raw)).astype(np.float32)
    dim = v1.shape[0]
    print(f"[init] dim={dim}  cos(v1,evidence)={cos_sim(v1,e_unit):+.4f}")

    # ── Load model ───────────────────────────────────────────────────────────
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

    # ── PHASE 1: Capture T0 hiddens + margins (variant A, B extraction) ────
    print(f"\n[phase1] capturing T0 hiddens (extractability fixed)")
    pairs = [json.loads(l) for l in open(args.pairs_path)]
    t0 = [r for r in pairs if r["condition"] == "T0"]
    print(f"[phase1] T0 prompts: {len(t0)}")

    t0_hiddens, t0_margins = [], []
    t0_meta = []
    t0_t0 = time.time()
    for i, rec in enumerate(t0):
        prompt = build_t0_prompt(tok, rec)
        h, m = capture_hidden_and_margin(model, tok, prompt, LAYER,
                                         tool_ids, fin_ids)
        t0_hiddens.append(h); t0_margins.append(m)
        t0_meta.append({"sample_id": rec["sample_id"], "margin": float(m)})
        if (i + 1) % 10 == 0:
            print(f"  T0 [{i+1}/{len(t0)}] {time.time()-t0_t0:.0f}s elapsed")
    t0_hiddens = np.stack(t0_hiddens).astype(np.float32)
    t0_margins = np.array(t0_margins, dtype=np.float32)
    print(f"[phase1] margin range: [{t0_margins.min():.2f}, {t0_margins.max():.2f}]"
          f"  median={np.median(t0_margins):+.2f}"
          f"  pct(margin>0)={(t0_margins>0).mean()*100:.0f}%")

    # ── Variant A: within-T0 contrast ───────────────────────────────────────
    v2, bot_idx, top_idx = extract_v2_within_t0(t0_hiddens, t0_margins)
    v2 = normalize_rms(v2, 1.0)
    print(f"[v2] within-T0: |bot|={len(bot_idx)} |top|={len(top_idx)}")

    # ── Variant B: PCA top component ────────────────────────────────────────
    v3, best_k, best_r = extract_v3_pca(t0_hiddens, t0_margins, n_components=5)
    v3 = normalize_rms(v3, 1.0)
    print(f"[v3] PCA: best_component=PC{best_k}  corr_with_sign(margin)={best_r:+.3f}")

    # ── PHASE 2: Build HotpotQA p0 eval prompts (matches §8.3 reference) ────
    print(f"\n[phase2] building HotpotQA p0 eval prompts (matches §8.3)")
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
    print(f"[phase2] {len(eval_prompts)} p0 prompts ready")

    # ── PHASE 3: Random direction sweep (variant-C extraction + null) ───────
    print(f"\n[phase3] random direction sweep (n_random={args.n_random})")
    random_dirs = gen_random_dirs(dim, args.n_random, SEED)
    eval_baseline = []
    p3_t0 = time.time()
    for prompt in eval_prompts:
        eval_baseline.append(measure_margin(model, tok, prompt, None, 0.0,
                                            LAYER, tool_ids, fin_ids))
    print(f"  baseline done ({time.time()-p3_t0:.0f}s)")

    rand_shifts = {i: [] for i in range(args.n_random)}
    p3_t1 = time.time()
    for ri, rd in enumerate(random_dirs):
        for pi, prompt in enumerate(eval_prompts):
            m = measure_margin(model, tok, prompt, rd, RHO, LAYER,
                               tool_ids, fin_ids)
            rand_shifts[ri].append(m - eval_baseline[pi])
        if (ri + 1) % 5 == 0:
            print(f"  random [{ri+1}/{args.n_random}] {time.time()-p3_t1:.0f}s")

    rand_mean = np.array([np.mean(rand_shifts[i]) for i in range(args.n_random)])
    rand_abs_mean = np.array([np.mean(np.abs(rand_shifts[i]))
                              for i in range(args.n_random)])
    print(f"[phase3] random null: signed mean={rand_mean.mean():+.3f}±{rand_mean.std():.3f}"
          f"  |.|={rand_abs_mean.mean():.3f}")

    # ── Variant C: top-K steering-effective ─────────────────────────────────
    v4, v4_top_idx, v4_top_shifts = extract_v4_steering(
        rand_shifts, random_dirs, top_k=args.top_k)
    print(f"[v4] top-{args.top_k} idx={v4_top_idx}  shifts={[f'{s:+.3f}' for s in v4_top_shifts]}")

    # ── PHASE 4: Functional decomposition for each variant ─────────────────
    print(f"\n[phase4] §8.3 functional decomposition for v2, v3, v4")
    variants = {"v2_within_t0": v2, "v3_pca": v3, "v4_steering": v4}
    decomp_results = {}
    p4_t0 = time.time()
    for vname, vfull in variants.items():
        par, perp = decompose_against(vfull, e_unit)
        # Renormalize each component to unit RMS (matches §8.3 protocol)
        full_n = normalize_rms(vfull, 1.0)
        par_n = normalize_rms(par, 1.0) if np.sqrt(np.mean(par**2)) > 1e-8 else par
        perp_n = normalize_rms(perp, 1.0)
        cos_v_e = cos_sim(vfull, e_unit)
        cos_v_v1 = cos_sim(vfull, v1)
        full_shifts, par_shifts, perp_shifts = [], [], []
        for pi, prompt in enumerate(eval_prompts):
            m_full = measure_margin(model, tok, prompt, full_n, RHO, LAYER,
                                    tool_ids, fin_ids)
            m_par = measure_margin(model, tok, prompt, par_n, RHO, LAYER,
                                   tool_ids, fin_ids)
            m_perp = measure_margin(model, tok, prompt, perp_n, RHO, LAYER,
                                    tool_ids, fin_ids)
            full_shifts.append(m_full - eval_baseline[pi])
            par_shifts.append(m_par - eval_baseline[pi])
            perp_shifts.append(m_perp - eval_baseline[pi])
        full_shifts = np.array(full_shifts)
        par_shifts = np.array(par_shifts)
        perp_shifts = np.array(perp_shifts)
        # Dissociation criterion (matches §8.3): parallel within random null,
        # perp recovers full effect (≥80% of full magnitude with same sign).
        par_in_null = (abs(np.mean(np.abs(par_shifts))
                           - rand_abs_mean.mean()) <= 2 * rand_abs_mean.std())
        perp_recovers = (np.sign(np.mean(perp_shifts)) == np.sign(np.mean(full_shifts))
                         and abs(np.mean(perp_shifts))
                         >= 0.80 * abs(np.mean(full_shifts)))
        decomp_results[vname] = {
            "cos_with_evidence": cos_v_e,
            "cos_with_v1": cos_v_v1,
            "parallel_norm_pre_renorm": float(np.sqrt(np.mean(par**2))),
            "full_mean_shift": float(np.mean(full_shifts)),
            "full_abs_mean_shift": float(np.mean(np.abs(full_shifts))),
            "parallel_mean_shift": float(np.mean(par_shifts)),
            "parallel_abs_mean_shift": float(np.mean(np.abs(par_shifts))),
            "perp_mean_shift": float(np.mean(perp_shifts)),
            "perp_abs_mean_shift": float(np.mean(np.abs(perp_shifts))),
            "parallel_pctile_signed": float(percentileofscore(rand_mean,
                                                              np.mean(par_shifts))),
            "parallel_pctile_abs": float(percentileofscore(rand_abs_mean,
                                                           np.mean(np.abs(par_shifts)))),
            "full_pctile_signed": float(percentileofscore(rand_mean,
                                                          np.mean(full_shifts))),
            "parallel_in_random_null": bool(par_in_null),
            "perp_recovers_full": bool(perp_recovers),
            "dissociation_holds": bool(par_in_null and perp_recovers),
        }
        print(f"  [{vname}] cos(e)={cos_v_e:+.4f} cos(v1)={cos_v_v1:+.4f}"
              f" full={np.mean(full_shifts):+.3f} par={np.mean(par_shifts):+.3f}"
              f" perp={np.mean(perp_shifts):+.3f}"
              f" -> diss={decomp_results[vname]['dissociation_holds']}"
              f"  ({time.time()-p4_t0:.0f}s)")

    # ── Save artifacts ──────────────────────────────────────────────────────
    n_pass = sum(1 for r in decomp_results.values() if r["dissociation_holds"])
    requirement_met = n_pass >= 2
    summary = {
        "config": {
            "layer": LAYER, "rho": RHO, "seed": SEED, "dim": dim,
            "n_t0": int(len(t0)), "n_eval": int(len(eval_prompts)),
            "n_random": args.n_random, "top_k": args.top_k,
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "pairs_path": args.pairs_path,
            "baseline_trace": args.baseline_trace,
        },
        "reference_v1": {
            "cos_with_evidence": cos_sim(v1, e_unit),
            "source": "steering/directions/direction_search_v3_layer20.npz",
        },
        "phase1_t0": {
            "margin_min": float(t0_margins.min()),
            "margin_max": float(t0_margins.max()),
            "margin_median": float(np.median(t0_margins)),
            "pct_margin_positive": float((t0_margins > 0).mean()),
        },
        "random_null": {
            "n": args.n_random,
            "signed_mean": float(rand_mean.mean()),
            "signed_std": float(rand_mean.std()),
            "abs_mean": float(rand_abs_mean.mean()),
            "abs_std": float(rand_abs_mean.std()),
            "abs_p97_5": float(np.percentile(rand_abs_mean, 97.5)),
        },
        "v4_extraction": {
            "top_idx": v4_top_idx,
            "top_shifts": v4_top_shifts,
        },
        "variants": decomp_results,
        "requirement": {
            "rule": ">=2 of 3 variants reproduce parallel-inert / perp-recovers-full",
            "n_dissociation_holds": n_pass,
            "requirement_met": bool(requirement_met),
        },
    }
    out_dir = args.output_dir
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(os.path.join(out_dir, "directions.npz"),
             v1=v1, v2_within_t0=v2, v3_pca=v3, v4_steering=v4,
             evidence_unit=e_unit)
    print(f"\n[done] requirement_met={requirement_met} ({n_pass}/3 variants)")
    print(f"[done] saved -> {out_dir}/summary.json, directions.npz")


if __name__ == "__main__":
    main()
