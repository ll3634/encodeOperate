#!/usr/bin/env python3
"""
Exp 5 — Direction-extraction robustness (§16.7 compressed).
Qwen3-32B only.

20 random train/test splits of the direction-extraction data.
Per split: re-extract evidence_dir (probe on train fold) and action_dir
(p10/p90 PopQA contrastive on train fold), then compute:
  - cos(action_dir, evidence_dir)
  - par_natural_over_full: behavioral margin shift ratio at rho=-0.20

All direction-extraction utilities reused from scripts/cross_model_full.py.
All steering utilities reused from scripts/decomposition_ci_hardened_cross_model.py.

Gate: IQR(par_natural_over_full) < 0.02 (2pp). Fail → flag in Limitations.

Usage:
  cd tmc/scripts/e2e_agent
  python scripts/robustness_qwen3_32b.py \
      --peak-layer <L_act from full_results.json> \
      --out results/qwen3_32b_scale_check/robustness
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS                   # noqa: E402
from steering.hook_utils import get_model_layers, SteeringHook, compute_rms  # noqa: E402
from scripts.cross_model_full import (                                   # noqa: E402
    apply_chat_template_safe, collect_step1_states,
    collect_popqa_multilayer, extract_action_dir_from_popqa, train_probe,
)

SEED = 20260502
N_SPLITS = 20
RHO = -0.20

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_margin(model, tok, prompt, layer, vec, alpha, tool_ids, fin_ids, device):
    layers = get_model_layers(model)
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    if vec is None:
        with torch.no_grad():
            logits = model(ids).logits[0, -1, :].float()
    else:
        def hook(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h[0, -1, :] += torch.tensor(vec * alpha, device=h.device, dtype=h.dtype)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        hdl = layers[layer].register_forward_hook(hook)
        with torch.no_grad():
            logits = model(ids).logits[0, -1, :].float()
        hdl.remove()
    lp = torch.log_softmax(logits, dim=-1)
    return (torch.logsumexp(lp[tool_ids], 0) - torch.logsumexp(lp[fin_ids], 0)).item()


def normalize_rms(v, target=1.0):
    rms = float(np.sqrt(np.mean(v ** 2)))
    return v if rms < 1e-12 else (v * (target / rms)).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/home/featurize/work/models/Qwen3-32B")
    ap.add_argument("--labels-path",  default="results/phase1_probe/labels.jsonl")
    ap.add_argument("--baseline-trace",
                    default="results/l20_rho020_n500/baseline_results.jsonl")
    ap.add_argument("--popqa-path",   default="data/popqa/popqa_test.jsonl")
    ap.add_argument("--pairs-path",
                    default="results/extractability_support_toggle/pairs.jsonl")
    ap.add_argument("--n-popqa", type=int, default=200)
    ap.add_argument("--n-n0",    type=int, default=50)
    ap.add_argument("--peak-layer", type=int, required=True,
                    help="Action peak layer (peak_action_layer from full_results.json).")
    ap.add_argument("--out", default="results/qwen3_32b_scale_check/robustness")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[load] {args.model_path}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.eval()
    device = next(model.parameters()).device
    D = model.config.hidden_size
    sqrtD = float(np.sqrt(D))
    print(f"[loaded] {time.time()-t0:.1f}s  layer={args.peak_layer}  D={D}")

    tool_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["tool_call"]]
    fin_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in ACTION_TOKENS["finish"]]
    layers_m = get_model_layers(model)

    # ── Collect full data ONCE ────────────────────────────────────────────────
    print("\n[data] collecting step1 and popqa states...")
    layer = args.peak_layer
    step1_data = collect_step1_states(model, tok, args.labels_path,
                                      args.baseline_trace, [layer])
    popqa_by_layer = collect_popqa_multilayer(model, tok, args.popqa_path,
                                              [layer], n=args.n_popqa)
    popqa_data = popqa_by_layer[layer]
    N_evi = len(step1_data)
    N_popqa = len(popqa_data["margins"])
    X_all = np.array([d["hidden"][layer] for d in step1_data], dtype=np.float32)
    y_all = np.array([d["label"] for d in step1_data], dtype=np.int32)
    print(f"  step1 N={N_evi}  popqa N={N_popqa}")

    # ── N0 steering prompts and baselines ─────────────────────────────────────
    records_all = [json.loads(l) for l in open(args.pairs_path)]
    records_n0  = [r for r in records_all if r.get("condition") == "N0"][:args.n_n0]
    builder = PromptBuilder()
    prompts, rms_arr = [], []
    for rec in records_n0:
        steps = [{"action": "search",
                  "action_input": f"about: {rec['question'][:80]}",
                  "observation": rec["obs"]}]
        msgs   = builder.build_full_prompt(rec["question"], steps)
        prompt = apply_chat_template_safe(tok, msgs)
        prompts.append(prompt)
        # measure h_rms at peak layer
        cap = {}
        def h(m, inp, out, _cap=cap):
            x = out[0] if isinstance(out, tuple) else out
            _cap["v"] = x[0, -1, :].detach().float().cpu().numpy()
        hdl = layers_m[layer].register_forward_hook(h)
        ids = tok.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad(): model(ids)
        hdl.remove()
        rms_arr.append(compute_rms(cap["v"]))
    rms_arr = np.array(rms_arr, dtype=np.float32)
    print(f"  N0 prompts: {len(prompts)}, h_rms mean={rms_arr.mean():.3f}")

    # ── Baselines ─────────────────────────────────────────────────────────────
    baselines = [get_margin(model, tok, p, layer, None, 0.0, tool_ids, fin_ids, device)
                 for p in prompts]
    print(f"  baseline margin mean={np.mean(baselines):+.3f}")

    # ── 20-split loop ─────────────────────────────────────────────────────────
    split_results = []
    for s in range(N_SPLITS):
        rng = np.random.RandomState(SEED + s)
        print(f"\n[split {s+1}/{N_SPLITS}]")
        # Evidence split (70% train)
        ei  = rng.permutation(N_evi)
        tr  = ei[:int(0.7 * N_evi)]
        evidence_dir, _ = train_probe(X_all[tr], y_all[tr])

        # PopQA split (70% train)
        pi  = rng.permutation(N_popqa)
        ptr = pi[:int(0.7 * N_popqa)]
        pd_train = {"margins": [popqa_data["margins"][j] for j in ptr],
                    "hiddens": [popqa_data["hiddens"][j] for j in ptr]}
        action_dir, _, _ = extract_action_dir_from_popqa(pd_train)
        if action_dir is None:
            print(f"  [skip] action_dir extraction failed for split {s}")
            continue

        e_unit  = evidence_dir / np.linalg.norm(evidence_dir)
        a_unit  = action_dir  / np.linalg.norm(action_dir)
        cos_ae  = float(np.dot(a_unit, e_unit))
        par_nat = (float(np.dot(action_dir, e_unit)) * e_unit).astype(np.float32)
        par_nat_L2 = float(np.linalg.norm(par_nat))
        full_rms = normalize_rms(action_dir, 1.0)

        shifts_full, shifts_par = [], []
        for i, (prompt, rms_h) in enumerate(zip(prompts, rms_arr)):
            alpha_rms = RHO * float(rms_h)
            alpha_nat = RHO * float(rms_h) * sqrtD * par_nat_L2
            m_full = get_margin(model, tok, prompt, layer, full_rms, alpha_rms,
                                tool_ids, fin_ids, device)
            # inject par_natural at its natural magnitude
            if par_nat_L2 > 1e-12:
                par_unit = par_nat / par_nat_L2
                m_par  = get_margin(model, tok, prompt, layer, par_unit, alpha_nat,
                                    tool_ids, fin_ids, device)
            else:
                m_par = baselines[i]
            shifts_full.append(m_full - baselines[i])
            shifts_par.append(m_par - baselines[i])

        full_mean = float(np.mean(shifts_full))
        par_mean  = float(np.mean(shifts_par))
        ratio = par_mean / full_mean if abs(full_mean) > 1e-6 else float("nan")
        split_results.append({"split": s, "cos_ae": cos_ae,
                               "par_natural_L2": par_nat_L2,
                               "full_mean": full_mean, "par_mean": par_mean,
                               "par_natural_over_full": ratio})
        print(f"  cos={cos_ae:+.4f}  par_nat_L2={par_nat_L2:.4f}"
              f"  full={full_mean:+.3f}  par={par_mean:+.4f}  ratio={ratio:+.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    cos_vals   = np.array([r["cos_ae"]                for r in split_results])
    ratio_vals = np.array([r["par_natural_over_full"] for r in split_results
                           if not np.isnan(r["par_natural_over_full"])])
    summary = {
        "model": args.model_path,
        "peak_layer": args.peak_layer,
        "n_splits": N_SPLITS,
        "n_completed": len(split_results),
        "rho": RHO,
        "cos_ae": {"median": float(np.median(cos_vals)),
                   "iqr": float(np.percentile(cos_vals, 75) - np.percentile(cos_vals, 25)),
                   "p25": float(np.percentile(cos_vals, 25)),
                   "p75": float(np.percentile(cos_vals, 75))},
        "par_natural_over_full": {"median": float(np.median(ratio_vals)),
                                   "iqr": float(np.percentile(ratio_vals, 75)
                                                - np.percentile(ratio_vals, 25)),
                                   "p25": float(np.percentile(ratio_vals, 25)),
                                   "p75": float(np.percentile(ratio_vals, 75))},
        "gate_iqr_parnat_over_full_lt_2pp": bool(
            float(np.percentile(ratio_vals, 75) - np.percentile(ratio_vals, 25)) < 0.02),
        "split_results": split_results,
    }
    print(f"\n=== Robustness Summary ===")
    print(f"  cos IQR   = {summary['cos_ae']['iqr']:.4f}")
    print(f"  par/full IQR = {summary['par_natural_over_full']['iqr']:.4f}  "
          f"(gate <2pp: {summary['gate_iqr_parnat_over_full_lt_2pp']})")
    out_path = out_dir / "robustness_qwen3_32b.json"
    json.dump(summary, open(out_path, "w"), indent=2)
    print(f"\n[saved] {out_path}")
    if not summary["gate_iqr_parnat_over_full_lt_2pp"]:
        print("[GATE FAIL] IQR(par_natural/full) >= 2pp — flag in Limitations.")


if __name__ == "__main__":
    main()
