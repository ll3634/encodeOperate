"""
OCFT shared utilities — activation extraction, probe training, decomposition.

Kept in a separate module so the CLI script in ocft_build_directions.py is
short and the heavy GPU/sklearn paths are loaded lazily.
"""

from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np


# ── Action direction loader ─────────────────────────────────────────────────

def load_action_direction(path: str):
    d = np.load(path, allow_pickle=True)
    if "decision_direction" in d:
        v = d["decision_direction"].astype(np.float32)
    else:
        v = d[list(d.keys())[0]].astype(np.float32)
    rms = float(np.sqrt(np.mean(v ** 2)))
    return v, {"rms": rms, "norm": float(np.linalg.norm(v))}


# ── Hotpot cache loader ─────────────────────────────────────────────────────

def extract_hotpot_cache(npz_path: str, labels_path: str, layer: int = 20):
    d = np.load(npz_path, allow_pickle=True)
    X = d[f"layer_{layer}"].astype(np.float32)
    sample_ids = list(d["sample_ids"])
    y_label = d["y"].astype(np.int32)
    label_recs = {json.loads(l)["sample_id"]: json.loads(l)
                  for l in open(labels_path)}
    margins = np.array([float(label_recs[sid]["margin_before"])
                        for sid in sample_ids], dtype=np.float32)
    return {
        "X": X, "sample_ids": sample_ids,
        "y_label": y_label, "margin_before": margins,
    }


# ── Forward-pass extraction (musique + pairs) ───────────────────────────────

_MODEL = None
_TOK = None


def _ensure_model(model_name: str):
    global _MODEL, _TOK
    if _MODEL is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"  [load] {model_name}")
        _TOK = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="eager")
        _MODEL.eval()
    return _MODEL, _TOK


def _extract_last_token_hidden(model, tok, prompt: str, layer: int):
    import torch
    from steering.hook_utils import get_model_layers
    device = next(model.parameters()).device
    layers = get_model_layers(model)
    captured = {}

    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h[0, -1, :].detach().float().cpu().numpy()

    handle = layers[layer].register_forward_hook(hook_fn)
    try:
        ids = tok.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()
    return captured["h"]


def extract_musique_p0(jsonl_path: str, n_max: int, model_name: str,
                       layer: int, build_prompt, parse_train_user):
    model, tok = _ensure_model(model_name)
    X, ids, sources, qs = [], [], [], []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if len(X) >= n_max:
                break
            content = r["prompt_messages"][1]["content"]
            parsed = parse_train_user(content)
            if parsed is None:
                continue
            question, query, observation = parsed
            prompt = build_prompt(tok, question, query, observation)
            X.append(_extract_last_token_hidden(model, tok, prompt, layer))
            ids.append(r["sample_id"])
            sources.append(r.get("source", "?"))
            qs.append(question)
            if len(X) % 25 == 0:
                print(f"    musique [{len(X)}/{n_max}]")
    return {"X": np.array(X, dtype=np.float32),
            "sample_ids": ids, "source": sources, "questions": qs}


def extract_pairs_p0(jsonl_path: str, n_max: int, model_name: str,
                     layer: int, build_prompt):
    model, tok = _ensure_model(model_name)
    X, ids, cond, candp, tlen = [], [], [], [], []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if len(X) >= n_max:
                break
            question = r["question"]
            obs = r.get("observation") or r.get("obs") or ""
            if not obs:
                continue
            prompt = build_prompt(tok, question, question, obs)
            X.append(_extract_last_token_hidden(model, tok, prompt, layer))
            ids.append(r["sample_id"])
            cond.append(r.get("condition", "?"))
            candp.append(bool(r.get("candidate_present", False)))
            tlen.append(int(r.get("token_len", len(obs.split()))))
            if len(X) % 25 == 0:
                print(f"    pairs [{len(X)}/{n_max}]")
    return {
        "X": np.array(X, dtype=np.float32),
        "sample_ids": ids, "condition": np.array(cond),
        "candidate_present": np.array(candp),
        "token_len": np.array(tlen, dtype=np.int32),
    }


# ── Probe training ──────────────────────────────────────────────────────────

def train_probe_on(X: np.ndarray, y: np.ndarray, seed: int = 20260502):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                                 precision_score, recall_score)
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(sss.split(Xs, y))
    clf = LogisticRegression(class_weight="balanced", C=1.0,
                             max_iter=2000, solver="lbfgs", random_state=seed)
    clf.fit(Xs[tr], y[tr])
    yp = clf.predict(Xs[te]); yh = clf.predict_proba(Xs[te])[:, 1]
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y[te], yp)),
        "auroc": float(roc_auc_score(y[te], yh)),
        "precision": float(precision_score(y[te], yp, zero_division=0)),
        "recall": float(recall_score(y[te], yp, zero_division=0)),
        "n_train": int(len(tr)), "n_test": int(len(te)),
    }
    # Re-train on all data for the direction
    clf_all = LogisticRegression(class_weight="balanced", C=1.0,
                                 max_iter=2000, solver="lbfgs",
                                 random_state=seed)
    clf_all.fit(Xs, y)
    w = clf_all.coef_[0] / scaler.scale_
    direction = (w / np.linalg.norm(w)).astype(np.float32)
    metrics["direction"] = direction
    return metrics


# ── Decomposition + save ────────────────────────────────────────────────────

def decompose_and_save(A: np.ndarray, D: np.ndarray, candidate_name: str,
                       steering_dir: str, layer: int = 20):
    D_unit = D / (np.linalg.norm(D) + 1e-12)
    A_par = float(np.dot(A, D_unit)) * D_unit  # 1-D projection
    A_perp = A - A_par
    par_path = Path(steering_dir) / f"direction_decomp_parallel_{candidate_name}_layer{layer}.npz"
    perp_path = Path(steering_dir) / f"direction_decomp_perp_{candidate_name}_layer{layer}.npz"
    np.savez(par_path, decision_direction=A_par.astype(np.float32),
             layer=layer, method="ocft_decomp_parallel",
             component=candidate_name,
             cos_with_probe=float(np.dot(A_par/np.linalg.norm(A_par+1e-12),
                                         D_unit)))
    np.savez(perp_path, decision_direction=A_perp.astype(np.float32),
             layer=layer, method="ocft_decomp_perp",
             component=candidate_name,
             cos_with_probe=float(np.dot(A_perp/np.linalg.norm(A_perp+1e-12),
                                         D_unit)))
    return {"parallel": str(par_path), "perp": str(perp_path),
            "norm_parallel": float(np.linalg.norm(A_par)),
            "norm_perp": float(np.linalg.norm(A_perp)),
            "var_parallel_fraction": float(np.linalg.norm(A_par)**2
                                            / max(np.linalg.norm(A)**2, 1e-12))}
