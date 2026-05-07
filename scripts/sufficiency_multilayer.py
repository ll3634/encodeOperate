#!/usr/bin/env python3
"""Extract multi-layer activations for 1-SF vs 2-SF pairs and run layer-wise sufficiency probe + cosine sweep."""
import sys, json, numpy as np, torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from steering.hook_utils import get_model_layers
from scripts.paired_corruption_analysis import build_prompt
from scripts.probe_sufficiency_synthetic import select_1sf_samples

LAYERS = [12, 16, 20]

def extract_multilayer(model, tokenizer, prompt, layer_indices):
    layers = get_model_layers(model)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    captured = {}
    handles = []
    for li in layer_indices:
        def make_hook(l):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[l] = h[0, -1, :].detach().float().cpu().numpy()
            return hook_fn
        handles.append(layers[li].register_forward_hook(make_hook(li)))
    with torch.no_grad():
        model(input_ids)
    for h in handles:
        h.remove()
    return {l: captured[l] for l in layer_indices}

def main():
    out_dir = Path("results/probe_sufficiency_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = select_1sf_samples(
        "results/l20_rho020_n500/baseline_results.jsonl",
        "data/hotpotqa/hotpot_dev_distractor_v1.json",
        "results/phase1_probe/labels.jsonl",
        max_n=300, seed=42)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True, attn_implementation="eager")
    model.eval()
    print(f"Model loaded. Extracting layers {LAYERS}.")

    data = {f"X_1sf_L{l}": [] for l in LAYERS}
    data.update({f"X_2sf_L{l}": [] for l in LAYERS})
    sids = []

    for s in tqdm(samples, desc="Extracting"):
        try:
            p1 = build_prompt(tokenizer, s["question"], s["query"], s["obs_1sf"])
            p2 = build_prompt(tokenizer, s["question"], s["query"], s["obs_2sf"])
            h1 = extract_multilayer(model, tokenizer, p1, LAYERS)
            h2 = extract_multilayer(model, tokenizer, p2, LAYERS)
            for l in LAYERS:
                data[f"X_1sf_L{l}"].append(h1[l])
                data[f"X_2sf_L{l}"].append(h2[l])
            sids.append(s["sample_id"])
        except Exception as e:
            print(f"  SKIP {s['sample_id']}: {e}")

    arrays = {k: np.array(v, dtype=np.float32) for k, v in data.items()}
    arrays["sample_ids"] = np.array(sids)
    np.savez(str(out_dir / "activations_multilayer.npz"), **arrays)
    print(f"Saved {len(sids)} pairs to {out_dir / 'activations_multilayer.npz'}")

    del model
    torch.cuda.empty_cache()
    run_analysis(arrays, out_dir)


def run_analysis(arrays, out_dir):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from scipy.stats import pearsonr

    # Load directions
    def load_dir(path):
        d = np.load(path)["decision_direction"].astype(np.float64)
        return d / np.linalg.norm(d)

    action_dirs = {}
    for l in LAYERS:
        try:
            action_dirs[l] = load_dir(f"steering/directions/direction_search_v3_layer{l}.npz")
        except:
            action_dirs[l] = None

    evi_dir_l20 = load_dir("results/phase1_probe/probe_direction_l20.npz")

    SEP = "=" * 70
    print(f"\n{SEP}\nLAYER-WISE SUFFICIENCY PROBE + COSINE SWEEP\n{SEP}")

    for l in LAYERS:
        X_1 = arrays[f"X_1sf_L{l}"].astype(np.float64)
        X_2 = arrays[f"X_2sf_L{l}"].astype(np.float64)
        N = len(X_1)
        X = np.vstack([X_1, X_2])
        y = np.array([0]*N + [1]*N)

        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aurocs = cross_val_score(
            LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, solver="lbfgs", random_state=42),
            Xs, y, cv=cv, scoring="roc_auc")
        clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
        clf.fit(Xs, y)
        w = clf.coef_[0] / sc.scale_
        suf_dir = w / np.linalg.norm(w)

        # Evidence probe at this layer
        d_phase1 = np.load("results/phase1_probe/activations_multilayer.npz", allow_pickle=True)
        if f"layer_{l}" in d_phase1.files:
            X_evi = d_phase1[f"layer_{l}"].astype(np.float64)
            y_evi = d_phase1["y"]
            sc_e = StandardScaler()
            clf_e = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
            clf_e.fit(sc_e.fit_transform(X_evi), y_evi)
            w_e = clf_e.coef_[0] / sc_e.scale_
            evi_dir = w_e / np.linalg.norm(w_e)
        else:
            evi_dir = None

        act_dir = action_dirs.get(l)

        print(f"\nLayer {l}:")
        print(f"  Sufficiency AUROC: {aurocs.mean():.3f} +/- {aurocs.std():.3f}  {[f'{s:.3f}' for s in aurocs]}")
        if evi_dir is not None and act_dir is not None:
            print(f"  cos(evi, act):  {np.dot(evi_dir, act_dir):.4f}")
            print(f"  cos(suf, act):  {np.dot(suf_dir, act_dir):.4f}")
            print(f"  cos(evi, suf):  {np.dot(evi_dir, suf_dir):.4f}")
        elif act_dir is not None:
            print(f"  cos(suf, act):  {np.dot(suf_dir, act_dir):.4f}")

        # Paired delta on action_dir
        if act_dir is not None:
            delta = X_2 - X_1
            d_act = delta @ act_dir
            from scipy.stats import ttest_1samp
            t, p = ttest_1samp(d_act, 0)
            print(f"  delta_action: mean={d_act.mean():.4f}, t={t:.2f}, p={p:.2e}")


if __name__ == "__main__":
    main()
