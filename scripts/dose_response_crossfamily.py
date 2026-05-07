#!/usr/bin/env python3
"""
Cross-Family Dose-Response (Gemma-2-9B-it L37, Mistral-7B-v0.3 L28)
====================================================================

Mirrors the §18 Qwen sweep (dose_response_gain_ratio.py) at matching coverage,
but uses §20 cross-family directions (no re-extraction → no sign drift) and
the par_natural injection convention (no RMS amplification of evidence/random).

Conditions per model (12 total):
  - action_dir                          (alpha = rho * h_RMS * sqrt(D))
  - evidence_dir at par_natural mag     (alpha = rho * h_RMS * sqrt(D) * |cos|)
  - 10 random unit vectors @ par_nat    (alpha = rho * h_RMS * sqrt(D) * |cos|)

Doses (rho magnitudes): 0.10, 0.20, 0.50.  rho_sign = -1 (continue/search).

Per example (and per (direction, rho) cell):
  - 2nd_search_rate (action_type == "search")
  - parse_success   (1 - parse_failure)
  - em              (vs gold_answer)

Aggregate per condition: mean + bootstrap 95% CI (B=2000).
Saves per-condition rows under {out}/{model}/{label}/rho_{mag}.jsonl
plus {out}/{model}/baseline.jsonl and {out}/{model}_dose_response.json.
"""
import argparse, json, os, sys, time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.prompts import parse_action                                # noqa: E402
from eval.scorers import answer_scorer                                # noqa: E402
from steering.hook_utils import (                                     # noqa: E402
    get_model_layers, SteeringHook, compute_rms,
)
from scripts.eval_extractability_cross_model import (                 # noqa: E402
    build_messages, apply_chat_template_safe, first_action_token,
)


SEED = 20260429
B_BOOT = 2000

MODEL_CFG = {
    "gemma": {
        "model_path": "unsloth/gemma-2-9b-it",
        "layer": 37,
        "directions": "results/gemma_circuit_sanity/exp2_samelayer/directions.npz",
        "evidence_key": "evidence_dir_L37",
        "ids_source": "results/crossfamily_ci_decomposition/gemma/per_example_rows.jsonl",
        "baseline_ref_2sr": 1.00,    # cross_model_behavior_alignment N0 first_search_rate
    },
    "mistral": {
        "model_path": "unsloth/mistral-7b-instruct-v0.3",
        "layer": 28,
        "directions": "results/mistral_circuit_sanity/exp2_samelayer/directions.npz",
        "evidence_key": "evidence_dir_L37",  # filename quirk: actually L28
        "ids_source": "results/crossfamily_ci_decomposition/mistral/per_example_rows.jsonl",
        "baseline_ref_2sr": 0.90,    # cross_model_extractability N0 first_search_rate
    },
    # Qwen3-32B scale-check entry (Exp 4 / §18/§22.1 replication at 32B scale).
    # layer and ids_source are overridable via --peak-layer / --ids-source CLI args;
    # they are filled from results/qwen3_32b_scale_check/ after Exp 1+2 and Exp 3 complete.
    "qwen3_32b": {
        "model_path": "/home/featurize/work/models/Qwen3-32B",
        "layer": None,          # filled via --peak-layer at runtime
        "directions": "results/qwen3_32b_scale_check/directions.npz",
        "evidence_key": "evidence_dir",
        "ids_source": None,     # filled via --ids-source at runtime
        "baseline_ref_2sr": None,
    },
}

PAIRS_PATH = "results/extractability_support_toggle/pairs.jsonl"


# ─── Helpers ────────────────────────────────────────────────────────────────

def jsonl_load(path):
    return [json.loads(l) for l in open(path)]


def jsonl_dump(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def bootstrap_ci(values, n_boot=B_BOOT, ci=95.0, seed=SEED):
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))
    boot = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std(ddof=1)) if n > 1 else 0.0,
        "ci_low":  float(np.percentile(boot, (100 - ci) / 2.0)),
        "ci_high": float(np.percentile(boot, 100 - (100 - ci) / 2.0)),
        "n": int(n),
    }


def load_records_matching_ids(pairs_path, ids_path, condition, max_n):
    """Return N0 records from `pairs_path` whose sample_id appears in `ids_path`,
    in the exact order given by `ids_path`."""
    pairs = {r["sample_id"]: r for r in jsonl_load(pairs_path)
             if r.get("condition") == condition}
    ids_in_order = []
    seen = set()
    for line in open(ids_path):
        sid = json.loads(line)["sample_id"]
        if sid not in seen and sid in pairs:
            ids_in_order.append(sid)
            seen.add(sid)
    out = [pairs[sid] for sid in ids_in_order][:max_n]
    return out


def make_unit_l2_vec(dim, seed):
    rng = np.random.RandomState(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / float(np.linalg.norm(v))


def get_hidden_at_last(model, tok, prompt, layer_idx, device):
    layers = get_model_layers(model)
    cap = {}
    def h(m, i, o):
        x = o[0] if isinstance(o, tuple) else o
        cap["v"] = x[0, -1, :].detach().float().cpu().numpy()
    handle = layers[layer_idx].register_forward_hook(h)
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model(ids)
    handle.remove()
    return cap["v"]


def run_generate(model, tok, prompt, device, max_new_tokens, direction=None,
                 alpha=0.0, layer=20):
    p_ids = tok.encode(prompt, return_tensors="pt",
                       add_special_tokens=False).to(device)
    attn = torch.ones_like(p_ids)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if direction is None:
        with torch.no_grad():
            gen = model.generate(p_ids, attention_mask=attn,
                                 max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=pad_id)
    else:
        with SteeringHook(model, direction, alpha, layer=layer,
                          position=-1, mode="addition", max_interventions=1):
            with torch.no_grad():
                gen = model.generate(p_ids, attention_mask=attn,
                                     max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=pad_id)
    raw = tok.decode(gen[0, p_ids.shape[1]:], skip_special_tokens=True)
    return raw


def parse_one(rec, raw):
    """Mirror eval_extractability_cross_model.run_one's parse logic."""
    parsed = parse_action(raw)
    a2, fa = parsed["action"], parsed["final_answer"]
    pf = (a2 is None and fa is None)
    if a2 and a2.lower() in ("search", "calculator"):
        action_type = "search"
    elif fa is not None:
        action_type = "stop"
    else:
        action_type = None
    fa_first = first_action_token(raw)  # 'search' / 'stop' / 'parse_fail'
    em = None
    if fa is not None and rec.get("gold_answer"):
        gold = rec.get("gold_answers") or [rec["gold_answer"]]
        em = int(answer_scorer(fa, gold, mode="exact")["matched"])
    return {
        "action_type": action_type,
        "first_action_token": fa_first,
        "second_search": int(fa_first == "search"),
        "parse_failure": int(pf),
        "parse_success": int(not pf),
        "em": em if em is not None else 0,
        "em_eligible": int(fa is not None and rec.get("gold_answer") is not None),
        "raw_output": raw[:400],
    }


# ─── Direction loading ──────────────────────────────────────────────────────

def load_directions(npz_path, evidence_key):
    d = np.load(npz_path)
    action = d["action_dir"].astype(np.float32)
    if evidence_key not in d.files:
        for k in ["evidence_dir", "evidence_dir_L37", "evidence_dir_L28"]:
            if k in d.files:
                evidence_key = k; break
    evidence = d[evidence_key].astype(np.float32)
    a_unit = action / np.linalg.norm(action)
    e_unit = evidence / np.linalg.norm(evidence)
    cos_ae = float(np.dot(a_unit, e_unit))
    return a_unit, e_unit, cos_ae


# ─── Main per-model run ─────────────────────────────────────────────────────

def run_model(model_key, args):
    cfg = MODEL_CFG[model_key]
    out_dir = Path(args.out) / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.out) / "run.log"

    def log(msg):
        line = f"[{datetime.now().isoformat()}] {model_key}: {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"=== run_model {model_key} layer=L{cfg['layer']} N={args.n} "
        f"K_random={args.n_random} rhos={args.rhos} ===")

    # ── Load directions ──
    npz = cfg["directions"]
    if not Path(npz).exists():
        raise SystemExit(f"FAIL HARD: missing direction npz for {model_key}: {npz}")
    a_unit, e_unit, cos_ae = load_directions(npz, cfg["evidence_key"])
    D = a_unit.shape[0]; sqrtD = float(np.sqrt(D))
    par_nat_L2 = abs(cos_ae)  # L2 norm of (cos·e_unit) since e_unit is unit-L2
    log(f"directions: D={D} cos(a,e)={cos_ae:+.6f} par_nat_L2={par_nat_L2:.6f}")

    # ── Load N0 prompts matching §20 ids ──
    if not Path(cfg["ids_source"]).exists():
        raise SystemExit(f"FAIL HARD: missing ids file for {model_key}: {cfg['ids_source']}")
    records = load_records_matching_ids(PAIRS_PATH, cfg["ids_source"],
                                        condition=args.condition, max_n=args.n)
    log(f"loaded {len(records)} prompts (cond={args.condition}) "
        f"first_id={records[0]['sample_id'] if records else 'NA'}")
    if len(records) < args.n:
        raise SystemExit(f"FAIL HARD: only {len(records)} prompts available "
                         f"(need {args.n}). Source: {cfg['ids_source']}")

    # ── Build random unit-L2 directions ──
    random_specs = []
    for k in range(args.n_random):
        seed = args.random_seed_base + k
        random_specs.append((f"random_s{seed}", make_unit_l2_vec(D, seed)))

    direction_specs = [("action", a_unit), ("evidence_par_natural", e_unit)] + random_specs

    # ── Load model ──
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    log(f"loading {cfg['model_path']} dtype={args.dtype}")
    tok = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], torch_dtype=dtype, device_map="auto",
        trust_remote_code=True)
    model.eval()
    device = next(model.parameters()).device

    # ── Pre-compute prompts and h_RMS at decision token ──
    prompts, rms_h_arr = [], np.zeros(len(records), dtype=np.float32)
    log("precomputing prompts + h_RMS at decision token ...")
    for i, rec in enumerate(records):
        msgs = build_messages(rec["question"], rec["obs"],
                              prompt_variant="v1", obs_style="factcard")
        prompt_str = apply_chat_template_safe(tok, msgs, add_generation_prompt=True)
        prompts.append(prompt_str)
        h = get_hidden_at_last(model, tok, prompt_str, cfg["layer"], device)
        rms_h_arr[i] = float(compute_rms(h))
    log(f"h_RMS mean={rms_h_arr.mean():.3f} std={rms_h_arr.std():.3f}")

    # ── Baseline (no steering) ──
    bl_path = out_dir / "baseline.jsonl"
    if bl_path.exists() and len(jsonl_load(bl_path)) == len(records):
        bl_rows = jsonl_load(bl_path)
        log(f"baseline: reusing cached ({len(bl_rows)} rows)")
    else:
        bl_rows = []
        t0 = time.time()
        for i, (rec, prompt) in enumerate(zip(records, prompts)):
            raw = run_generate(model, tok, prompt, device, args.max_new_tokens,
                               direction=None, alpha=0.0, layer=cfg["layer"])
            row = parse_one(rec, raw)
            row["sample_id"] = rec["sample_id"]
            bl_rows.append(row)
            if (i + 1) % 10 == 0 or i == len(records) - 1:
                log(f"  baseline [{i+1}/{len(records)}] {time.time()-t0:.1f}s")
        jsonl_dump(bl_path, bl_rows)
    bl_2sr = float(np.mean([r["second_search"] for r in bl_rows]))
    ref_2sr = cfg["baseline_ref_2sr"]
    ref_str = f"{ref_2sr:.3f}" if ref_2sr is not None else "N/A"
    log(f"BASELINE 2nd_search_rate = {bl_2sr:.3f} (ref={ref_str})")

    # Smoke-mode early stop after baseline check (skip if ref not set)
    if ref_2sr is not None:
        diff_pp = abs(bl_2sr - ref_2sr) * 100
        if args.smoke:
            log(f"SMOKE MODE: |Δbaseline|={diff_pp:.1f}pp (threshold 2pp)")
            if diff_pp > 2.0:
                raise SystemExit(
                    f"SMOKE FAILURE: {model_key} baseline 2nd_search_rate={bl_2sr:.3f} "
                    f"differs from §20 ref {ref_2sr:.3f} by {diff_pp:.1f}pp > 2pp"
                )
    else:
        diff_pp = None
        if args.smoke:
            log("SMOKE MODE: no ref baseline set, skip diff check")

    # ── Steered conditions ──
    rho_mags = [float(x) for x in args.rhos.split(",")]
    sign = args.rho_sign
    conditions_out = {}
    summary_jsonp = out_dir / f"{model_key}_dose_response.json"

    summary_out = {
        "model_key": model_key,
        "model": cfg["model_path"],
        "layer": cfg["layer"],
        "config": {
            "n_samples": len(records),
            "rho_magnitudes": rho_mags,
            "rho_sign": sign,
            "n_random": args.n_random,
            "random_seed_base": args.random_seed_base,
            "condition": args.condition,
            "directions_npz": cfg["directions"],
            "ids_source": cfg["ids_source"],
            "max_new_tokens": args.max_new_tokens,
            "B_bootstrap": B_BOOT,
            "alpha_convention": "action: rho*h_RMS*sqrt(D); evidence/random: rho*h_RMS*sqrt(D)*|cos|",
        },
        "geometry": {
            "d_model": int(D),
            "cos_action_evidence": cos_ae,
            "par_natural_L2": float(par_nat_L2),
            "h_rms_mean": float(rms_h_arr.mean()),
            "h_rms_std": float(rms_h_arr.std()),
        },
        "baseline": {
            "2nd_search_rate": bl_2sr,
            "parse_success_rate": float(np.mean([r["parse_success"] for r in bl_rows])),
            "em_rate": float(np.mean([r["em"] for r in bl_rows])),
            "ref_2sr": cfg["baseline_ref_2sr"],
            "diff_pp_vs_ref": diff_pp,
        },
        "conditions": {},
    }

    for label, vec in direction_specs:
        cond_dir = out_dir / label
        cond_dir.mkdir(parents=True, exist_ok=True)
        per_rho = {}
        for mag in rho_mags:
            rho = sign * mag
            rho_path = cond_dir / f"rho_{mag:.2f}.jsonl"
            if rho_path.exists() and len(jsonl_load(rho_path)) == len(records):
                rows = jsonl_load(rho_path)
                log(f"  [{label} rho={rho:+.2f}] reusing cached ({len(rows)} rows)")
            else:
                rows = []
                t0 = time.time()
                for i, (rec, prompt) in enumerate(zip(records, prompts)):
                    h_rms = float(rms_h_arr[i])
                    if label == "action":
                        alpha = rho * h_rms * sqrtD
                    else:
                        alpha = rho * h_rms * sqrtD * par_nat_L2
                    raw = run_generate(model, tok, prompt, device,
                                       args.max_new_tokens, direction=vec,
                                       alpha=alpha, layer=cfg["layer"])
                    row = parse_one(rec, raw)
                    row["sample_id"] = rec["sample_id"]
                    row["alpha"] = alpha
                    row["h_rms"] = h_rms
                    rows.append(row)
                jsonl_dump(rho_path, rows)
                log(f"  [{label} rho={rho:+.2f}] N={len(rows)} t={time.time()-t0:.1f}s "
                    f"2sr={np.mean([r['second_search'] for r in rows]):.3f}")
            two_sr = [r["second_search"] for r in rows]
            ps = [r["parse_success"] for r in rows]
            em = [r["em"] for r in rows]
            per_rho[f"{mag:.2f}"] = {
                "rho_signed": rho,
                "second_search_rate": bootstrap_ci(two_sr),
                "parse_success_rate": bootstrap_ci(ps),
                "em_rate": bootstrap_ci(em),
            }
        summary_out["conditions"][label] = per_rho
        with open(summary_jsonp, "w") as f:
            json.dump(summary_out, f, indent=2, ensure_ascii=False)
        log(f"saved partial summary -> {summary_jsonp}")

    # Free GPU memory before next model
    del model
    torch.cuda.empty_cache()
    return summary_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma", "mistral"],
                    choices=["gemma", "mistral", "qwen3_32b"])
    ap.add_argument("--out", default="results/crossfamily_dose_response")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--random-seed-base", type=int, default=2000)
    ap.add_argument("--rhos", default="0.10,0.20,0.50")
    ap.add_argument("--rho-sign", type=int, default=-1, choices=[-1, 1])
    ap.add_argument("--condition", default="N0")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke mode: forces n=5, n_random=2, single-rho 0.20.")
    # CLI overrides for qwen3_32b whose peak_layer and ids_source are determined
    # at runtime from Exp 1+2 and Exp 3 outputs:
    ap.add_argument("--peak-layer", type=int, default=None,
                    help="Override MODEL_CFG[model].layer (used for qwen3_32b).")
    ap.add_argument("--ids-source", type=str, default=None,
                    help="Override MODEL_CFG[model].ids_source (used for qwen3_32b).")
    args = ap.parse_args()

    # Apply CLI overrides into MODEL_CFG
    for mk in (args.models if isinstance(args.models, list) else [args.models]):
        if args.peak_layer is not None:
            MODEL_CFG[mk]["layer"] = args.peak_layer
        if args.ids_source is not None:
            MODEL_CFG[mk]["ids_source"] = args.ids_source

    if args.smoke:
        args.n = 5
        args.n_random = 2
        args.rhos = "0.20"

    Path(args.out).mkdir(parents=True, exist_ok=True)
    summaries = {}
    for mk in args.models:
        summaries[mk] = run_model(mk, args)

    with open(Path(args.out) / "summary.json", "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                   "args": vars(args),
                   "models": summaries}, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Top-level summary: {Path(args.out) / 'summary.json'}")


if __name__ == "__main__":
    main()
