#!/usr/bin/env python3
"""Distributed Alignment Search (DAS) for evidence_sufficiency -> action_decision.

Tests the high-level causal hypothesis:
    Variable evidence_sufficiency (sufficient/insufficient) causes
    Variable action_decision (search/stop) at L20 of Qwen2.5-7B-Instruct.

Method (per Geiger et al., 2023; Wu et al., 2024):
    Learn a (3584 x k) orthogonal rotation R at L20 such that the interchange
    intervention I(R, base, source) = base + R R^T (source - base) at the last
    prompt token swaps the model's first-token decision (Action vs Final) from
    the base's natural decision to the source's natural decision.

    IIA (interchange intervention accuracy) = Pr[ argmax({Action,Final} logits
    after intervention) == source's natural action ].

Tasks:
    - evidence: base=T0 (W extractable, unsupported, model SEARCHES),
                source=S0 (W extractable, supported,   model STOPS).
                Both directions: (T0->S0) and (S0->T0).
    - extractability (CONTROL): base=N0 (W not extractable, model SEARCHES),
                                source=S0 (W extractable, model STOPS).
                                Both directions.

Baselines per k:
    - random: untrained random orthogonal R (chance-IIA reference).
    - probe (k=1 only): R = unit-normalised L20 evidence probe direction
      (decision_direction in direction_probe_layer20.npz).

Outputs to results/das_evidence_action_alignment/:
    - report.json (top-line aggregate)
    - <task>_k<k>_train_log.json (per-epoch loss/IIA)
"""
import argparse, json, sys, time, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from steering.hook_utils import get_model_layers

MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
HIDDEN = 3584
N_LAYERS = 28
ACTION_ID = 2512   # 'Action'
FINAL_ID  = 19357  # 'Final'
DATA_DIR = Path("tmc/scripts/e2e_agent/data/extractability_train")
PROBE_PATH = Path("tmc/scripts/e2e_agent/steering/directions/direction_probe_layer20.npz")
OUT_DIR = Path("tmc/scripts/e2e_agent/results/das_evidence_action_alignment")
K_VALUES = [1, 2, 4, 8, 16, 32]
N_EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 16
SEED = 17

# action_decision label: source's natural first generated token
LABEL_ACTION = 0   # SEARCH branch
LABEL_FINAL  = 1   # STOP branch
COND_LABEL = {"T0": LABEL_ACTION, "N0": LABEL_ACTION, "S0": LABEL_FINAL}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_examples(condition: str, n_max: int | None = None) -> list[dict]:
    """Load T0/S0/N0 prompts as list of dicts: {sid, messages, action_label}.

    sid is namespaced as f"{condition}/{sample_id}" because the same MuSiQue
    sample_id appears across all three conditions (only the prompt differs);
    a non-namespaced key would silently collide when caches are merged.
    """
    path = DATA_DIR / f"train_{condition}.jsonl"
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out.append({
                "sid": f"{condition}/{r['sample_id']}",
                "messages": r["prompt_messages"],
                "action_label": COND_LABEL[condition],
                "condition": condition,
            })
    if n_max is not None:
        out = out[:n_max]
    return out


def encode_prompt(tok, messages, max_len: int = 1600) -> torch.Tensor:
    """Apply chat template with generation prompt; return input_ids on CPU."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids
    return ids  # (1, T)


@torch.no_grad()
def cache_full_l20(model, tok, examples, device, dtype, fname_cache: Path | None = None):
    """For each example, cache (input_ids, h_at_L20[full_seq], last_idx).

    h_at_L20 is the residual stream OUTPUT of decoder layer LAYER (i.e., the
    input to layer LAYER+1). Stored as fp16 on CPU to save GPU memory.
    """
    if fname_cache is not None and fname_cache.exists():
        cache = torch.load(fname_cache, map_location="cpu")
        if cache and "natural_label" in next(iter(cache.values())):
            return cache
        print(f"  [cache] {fname_cache} missing natural_label; rebuilding", flush=True)

    layers = get_model_layers(model)
    cache = {}

    captured = {}
    def _hook(mod, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h.detach().to(torch.float16).cpu()  # (1,T,H)

    handle = layers[LAYER].register_forward_hook(_hook)
    try:
        t0 = time.time()
        for i, ex in enumerate(examples):
            captured.clear()
            ids = encode_prompt(tok, ex["messages"]).to(device)
            out = model(ids, use_cache=False)
            h = captured["h"][0]  # (T,H)
            T = h.shape[0]
            last_logits = out.logits[0, -1].float()
            a_lg = float(last_logits[ACTION_ID]); f_lg = float(last_logits[FINAL_ID])
            nat_label = LABEL_ACTION if a_lg > f_lg else LABEL_FINAL
            cache[ex["sid"]] = {
                "h_full": h,            # fp16 (T,H)
                "last_idx": T - 1,
                "T": T,
                "input_ids": ids[0].cpu(),
                "action_label": ex["action_label"],   # COND_LABEL-derived (legacy)
                "natural_label": nat_label,           # model's actual top-1 over {A,F}
                "Action_logit": a_lg,
                "Final_logit": f_lg,
                "condition": ex["condition"],
            }
            if (i + 1) % 50 == 0 or (i + 1) == len(examples):
                print(f"  [cache {ex['condition']}] {i+1}/{len(examples)}  "
                      f"({time.time()-t0:.1f}s)", flush=True)
    finally:
        handle.remove()

    if fname_cache is not None:
        fname_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, fname_cache)
    return cache


class UpperStack(torch.nn.Module):
    """Forward through layers[LAYER+1 : N_LAYERS] + final norm + lm_head.

    Takes hidden states output by layer LAYER (shape (B,T,H)) and returns logits
    at the *last* sequence position only (shape (B, vocab)).

    Computes rotary position_embeddings via the underlying Qwen2Model.rotary_emb.
    Causal masking is delegated to SDPA via attention_mask=None for q_len > 1.
    """
    def __init__(self, model):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [get_model_layers(model)[i] for i in range(LAYER + 1, N_LAYERS)]
        )
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.rotary_emb = model.model.rotary_emb

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, H), already output of layer LAYER
        B, T, H = h.shape
        position_ids = torch.arange(T, device=h.device).unsqueeze(0).expand(B, -1)
        position_embeddings = self.rotary_emb(h, position_ids)
        for layer in self.layers:
            out = layer(
                h,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            h = out[0]
        h = self.norm(h)
        last_logits = self.lm_head(h[:, -1, :])  # (B, vocab)
        return last_logits


def build_pairs(base_examples, source_examples, cache,
                base_natural: int, source_natural: int,
                n_pairs: int, rng: random.Random):
    """Return list of (base_sid, source_sid) pairs filtered by NATURAL label.

    Each base must have model's natural top-1 == base_natural; each source must
    have natural top-1 == source_natural. Only meaningful pairs
    (base_natural != source_natural) are informative for IIA: the high-level
    causal model predicts that the post-intervention argmax should match
    source_natural rather than base_natural, and identity-intervention IIA = 0.

    Pairs are sampled without replacement from the Cartesian product
    (eligible_bases x eligible_sources), capped at n_pairs.
    """
    bases = [e for e in base_examples   if cache[e["sid"]]["natural_label"] == base_natural]
    srcs  = [e for e in source_examples if cache[e["sid"]]["natural_label"] == source_natural]
    if not bases or not srcs:
        return []
    all_pairs = [(b["sid"], s["sid"]) for b in bases for s in srcs]
    rng.shuffle(all_pairs)
    return all_pairs[:n_pairs]


class FixedRotationIntervention(torch.nn.Module):
    """Non-trainable: replaces base's projection on R with source's projection.

      output = base + (source - base) @ R @ R^T
    where R is a fixed (HIDDEN, k) matrix with orthonormal columns.
    Used for the random-rotation and probe-direction baselines.
    """
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight.contiguous())

    def forward(self, base: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        diff = source - base                             # (B, HIDDEN)
        rotated_diff = diff @ self.weight                # (B, k)
        return base + rotated_diff @ self.weight.T       # (B, HIDDEN)


class TrainableRotationIntervention(torch.nn.Module):
    """DAS rotation with a free (HIDDEN, k) parameter, orthonormalised via
    differentiable QR at each forward call.

    The orthonormal projector P = R R^T (with R = qr(weight).Q) is rank-k.
    Output: base + (source - base) @ R @ R^T.
    QR is differentiable in PyTorch (autograd through linalg.qr), and unlike
    torch.nn.utils.parametrizations.orthogonal it doesn't zero the gradient
    after the first step.
    """
    def __init__(self, hidden: int, k: int):
        super().__init__()
        # Init: random Gaussian; QR will orthonormalise on the first forward.
        w = torch.randn(hidden, k) / (hidden ** 0.5)
        self.weight = torch.nn.Parameter(w)

    def _orth(self) -> torch.Tensor:
        # QR -> Q has orthonormal columns; absorbs sign convention from R.
        Q, _ = torch.linalg.qr(self.weight, mode="reduced")
        return Q  # (HIDDEN, k)

    def forward(self, base: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        Q = self._orth()
        diff = source - base                             # (B, HIDDEN)
        return base + (diff @ Q) @ Q.T                   # (B, HIDDEN)


def make_intervention(k: int, dtype, device, init_weight: torch.Tensor | None = None):
    """Build an intervention module on `device`, fp32.

    - init_weight is None -> trainable TrainableRotationIntervention(k)
    - init_weight is (HIDDEN, k) -> non-trainable FixedRotationIntervention.
    """
    if init_weight is None:
        itv = TrainableRotationIntervention(HIDDEN, k)
        itv.to(dtype=torch.float32).to(device)
        return itv
    itv = FixedRotationIntervention(init_weight.to(torch.float32))
    itv.to(device)
    return itv


def step_logits(upper_stack: UpperStack, h_base_full: torch.Tensor,
                last_idx: int, intervened_last: torch.Tensor,
                model_dtype: torch.dtype) -> torch.Tensor:
    """Replace h_base_full[last_idx] with intervened_last and run upper stack.

    h_base_full: (T, H) (no batch dim, fp32 on device)
    intervened_last: (H,) fp32 on device, requires_grad through R
    model_dtype: dtype of the model (bf16 / fp16 / fp32) to cast to before
                 the transformer layers (linear projections require dtype match).
    Returns logits at the last position (vocab,) in fp32.
    """
    h_mod = h_base_full.clone()
    h_mod[last_idx] = intervened_last
    # Cast to model dtype at the boundary; cast preserves gradient flow.
    h_mod_dt = h_mod.unsqueeze(0).to(model_dtype)  # (1, T, H)
    out = upper_stack(h_mod_dt)[0]  # (vocab,)
    return out.float()



def eval_iia(upper_stack, intervention, pairs, cache, device, model_dtype):
    """Compute IIA: fraction of pairs where post-intervention argmax over
    {Action, Final} == source's natural label.
    """
    intervention.eval()
    correct = 0
    losses = []
    with torch.no_grad():
        for (b_sid, s_sid) in pairs:
            b = cache[b_sid]; s = cache[s_sid]
            h_base_full = b["h_full"].to(device=device, dtype=torch.float32)
            h_base_last = h_base_full[b["last_idx"]].unsqueeze(0)   # (1, H)
            h_src_last  = s["h_full"][s["last_idx"]].to(
                device=device, dtype=torch.float32).unsqueeze(0)    # (1, H)
            interv = intervention(h_base_last, h_src_last).squeeze(0)  # (H,)
            logits = step_logits(upper_stack, h_base_full, b["last_idx"], interv, model_dtype)
            two = torch.stack([logits[ACTION_ID], logits[FINAL_ID]])
            pred = int(two.argmax().item())
            label = s["natural_label"]
            if pred == label:
                correct += 1
            losses.append(float(F.cross_entropy(two.unsqueeze(0),
                                                torch.tensor([label], device=device)).item()))
    n = len(pairs)
    iia = correct / n if n else float("nan")
    return {"iia": iia, "n": n, "loss_mean": float(np.mean(losses)) if losses else float("nan")}


def train_das(upper_stack, k, train_pairs, test_pairs, cache, device, model_dtype,
              n_epochs=N_EPOCHS, lr=LR, log_path: Path | None = None,
              init_weight: torch.Tensor | None = None, train: bool = True):
    """Train (or just evaluate) a DAS rotation at given k.

    train=False -> no optimization, just init + eval (for random and probe baselines).
    """
    intervention = make_intervention(k, dtype=torch.float32, device=device,
                                     init_weight=init_weight)
    history = []

    if not train:
        eval_train = eval_iia(upper_stack, intervention, train_pairs, cache, device, model_dtype)
        eval_test  = eval_iia(upper_stack, intervention, test_pairs,  cache, device, model_dtype)
        history.append({"epoch": 0, "train": eval_train, "test": eval_test})
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump({"k": k, "trained": False, "history": history},
                      open(log_path, "w"), indent=2)
        return intervention, history

    optim = torch.optim.AdamW(intervention.parameters(), lr=lr)
    rng = random.Random(SEED + k)
    t0 = time.time()
    eval_train0 = eval_iia(upper_stack, intervention, train_pairs, cache, device, model_dtype)
    eval_test0  = eval_iia(upper_stack, intervention, test_pairs,  cache, device, model_dtype)
    history.append({"epoch": 0, "train": eval_train0, "test": eval_test0,
                    "elapsed_s": time.time() - t0})
    print(f"  [k={k}] epoch 0 (init): train_iia={eval_train0['iia']:.3f}  "
          f"test_iia={eval_test0['iia']:.3f}  loss={eval_train0['loss_mean']:.3f}",
          flush=True)

    for epoch in range(1, n_epochs + 1):
        intervention.train()
        order = list(range(len(train_pairs))); rng.shuffle(order)
        ep_loss = 0.0; ep_n = 0
        optim.zero_grad()
        for i_in_batch, idx in enumerate(order):
            b_sid, s_sid = train_pairs[idx]
            b = cache[b_sid]; s = cache[s_sid]
            h_base_full = b["h_full"].to(device=device, dtype=torch.float32)
            h_base_last = h_base_full[b["last_idx"]].unsqueeze(0)
            h_src_last  = s["h_full"][s["last_idx"]].to(
                device=device, dtype=torch.float32).unsqueeze(0)
            interv = intervention(h_base_last, h_src_last).squeeze(0)
            logits = step_logits(upper_stack, h_base_full, b["last_idx"], interv, model_dtype)
            two = torch.stack([logits[ACTION_ID], logits[FINAL_ID]]).unsqueeze(0)
            label = torch.tensor([s["natural_label"]], device=device)
            loss = F.cross_entropy(two, label) / BATCH_SIZE
            loss.backward()
            ep_loss += float(loss.item()) * BATCH_SIZE; ep_n += 1
            if (i_in_batch + 1) % BATCH_SIZE == 0:
                optim.step(); optim.zero_grad()
        if ep_n % BATCH_SIZE != 0:
            optim.step(); optim.zero_grad()
        eval_train = eval_iia(upper_stack, intervention, train_pairs, cache, device, model_dtype)
        eval_test  = eval_iia(upper_stack, intervention, test_pairs,  cache, device, model_dtype)
        history.append({"epoch": epoch, "train_loss_mean": ep_loss / max(ep_n, 1),
                        "train": eval_train, "test": eval_test,
                        "elapsed_s": time.time() - t0})
        print(f"  [k={k}] epoch {epoch:>2d}  train_iia={eval_train['iia']:.3f}  "
              f"test_iia={eval_test['iia']:.3f}  loss={eval_train['loss_mean']:.3f}",
              flush=True)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"k": k, "trained": True, "n_train": len(train_pairs),
                   "n_test": len(test_pairs), "history": history},
                  open(log_path, "w"), indent=2)
    return intervention, history



def make_random_init_weight(k: int, seed: int) -> torch.Tensor:
    """Random orthonormal columns of shape (HIDDEN, k)."""
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(HIDDEN, k, generator=g)
    Q, _ = torch.linalg.qr(M)
    return Q  # (HIDDEN, k) orthonormal columns


def make_probe_init_weight() -> torch.Tensor:
    """Probe direction at L20 as (HIDDEN, 1) unit vector."""
    d = np.load(PROBE_PATH)["decision_direction"].astype(np.float32)
    v = torch.from_numpy(d)
    v = v / v.norm()
    return v.unsqueeze(1)  # (HIDDEN, 1)


def run_task(task_name: str, base_cond: str, source_cond: str, model, tok,
             upper_stack, device, dtype, n_pairs_per_dir: int = 200):
    """Run DAS for a (base_cond, source_cond) pair, both directions.

    Returns dict with per-k results (train + test IIA) for: das, random, probe.
    """
    print(f"\n=== TASK: {task_name}  (base={base_cond} <-> source={source_cond}) ===")
    rng = random.Random(SEED)

    # 1) Load + cache hidden states for both conditions
    base_examples   = load_examples(base_cond)
    source_examples = load_examples(source_cond)
    print(f"  loaded {len(base_examples)} {base_cond} + {len(source_examples)} {source_cond}")

    cache_path_base = OUT_DIR / "cache" / f"{base_cond}_l20.pt"
    cache_path_src  = OUT_DIR / "cache" / f"{source_cond}_l20.pt"
    base_cache = cache_full_l20(model, tok, base_examples, device, dtype, cache_path_base)
    src_cache  = cache_full_l20(model, tok, source_examples, device, dtype, cache_path_src)
    cache = {**base_cache, **src_cache}

    # 2) Natural-label breakdown per condition (model's actual top-1 over {A,F})
    def _count(exs, lab):
        return sum(1 for e in exs if cache[e["sid"]]["natural_label"] == lab)
    print("  natural top-1 over {Action, Final}:")
    for cname, exs in [(base_cond, base_examples), (source_cond, source_examples)]:
        nA = _count(exs, LABEL_ACTION); nF = _count(exs, LABEL_FINAL)
        print(f"    {cname}: Action={nA}/{len(exs)}  Final={nF}/{len(exs)}")

    # 3) Stratified 80/20 split: within each (condition, natural_label) bucket
    # so both train and test contain the rare Action-natural examples.
    def _split_strat(exs, frac=0.8):
        a = [e for e in exs if cache[e["sid"]]["natural_label"] == LABEL_ACTION]
        f = [e for e in exs if cache[e["sid"]]["natural_label"] == LABEL_FINAL]
        rng.shuffle(a); rng.shuffle(f)
        # Ensure at least 1 example in each (train,test) per non-empty bucket.
        def _cut(lst):
            if len(lst) == 0: return [], []
            if len(lst) == 1: return lst, []        # single example -> train only
            n_tr = max(1, min(len(lst) - 1, int(frac * len(lst))))
            return lst[:n_tr], lst[n_tr:]
        a_tr, a_te = _cut(a); f_tr, f_te = _cut(f)
        return a_tr + f_tr, a_te + f_te
    base_tr, base_te = _split_strat(base_examples)
    src_tr,  src_te  = _split_strat(source_examples)

    # 4) Build pairs both swap directions, requiring natural(base) != natural(source).
    # Direction A: base natural=Action, source natural=Final  (test if swap pushes Final)
    train_a = build_pairs(base_tr, src_tr, cache, LABEL_ACTION, LABEL_FINAL,
                          n_pairs_per_dir, rng)
    test_a  = build_pairs(base_te, src_te, cache, LABEL_ACTION, LABEL_FINAL,
                          n_pairs_per_dir, rng)
    # Direction B: base natural=Final, source natural=Action  (test if swap pushes Action)
    # Roles flipped: base drawn from source_examples, source drawn from base_examples.
    train_b = build_pairs(src_tr, base_tr, cache, LABEL_FINAL, LABEL_ACTION,
                          n_pairs_per_dir, rng)
    test_b  = build_pairs(src_te, base_te, cache, LABEL_FINAL, LABEL_ACTION,
                          n_pairs_per_dir, rng)
    train_pairs = train_a + train_b; rng.shuffle(train_pairs)
    test_pairs  = test_a  + test_b;  rng.shuffle(test_pairs)
    print(f"  pairs (dirA = base->Action,src->Final | dirB = base->Final,src->Action):")
    print(f"    train: A={len(train_a)}  B={len(train_b)}  total={len(train_pairs)}")
    print(f"    test : A={len(test_a)}   B={len(test_b)}   total={len(test_pairs)}")
    if not test_pairs:
        print(f"  [skip] no informative test pairs for task {task_name}", flush=True)
        return {"task": task_name, "base_cond": base_cond, "source_cond": source_cond,
                "n_train_pairs": len(train_pairs), "n_test_pairs": 0,
                "skipped": True}

    results = {"task": task_name, "base_cond": base_cond, "source_cond": source_cond,
               "n_train_pairs": len(train_pairs), "n_test_pairs": len(test_pairs),
               "n_train_dirA": len(train_a), "n_train_dirB": len(train_b),
               "n_test_dirA":  len(test_a),  "n_test_dirB":  len(test_b),
               "k_values": K_VALUES,
               "das": {}, "random": {}, "probe": {}}

    # 3) DAS sweep over k
    for k in K_VALUES:
        log_p = OUT_DIR / f"{task_name}_k{k}_das_train.json"
        _, hist = train_das(upper_stack, k, train_pairs, test_pairs, cache, device,
                            dtype, n_epochs=N_EPOCHS, lr=LR, log_path=log_p)
        best = max(hist, key=lambda r: r["test"]["iia"])
        final = hist[-1]
        results["das"][str(k)] = {
            "train_iia_final": final["train"]["iia"],
            "test_iia_final":  final["test"]["iia"],
            "test_iia_best":   best["test"]["iia"],
            "best_epoch":      best["epoch"],
            "loss_final":      final["train"]["loss_mean"],
        }
        print(f"  >> [k={k}] DAS final test IIA = {final['test']['iia']:.3f}  "
              f"(best={best['test']['iia']:.3f} @ ep {best['epoch']})", flush=True)

    # 4) Random rotation baseline (no training) per k
    for k in K_VALUES:
        log_p = OUT_DIR / f"{task_name}_k{k}_random.json"
        Wr = make_random_init_weight(k, seed=SEED + 1000 + k)
        _, hist = train_das(upper_stack, k, train_pairs, test_pairs, cache, device,
                            dtype, log_path=log_p, init_weight=Wr, train=False)
        results["random"][str(k)] = {
            "train_iia": hist[-1]["train"]["iia"],
            "test_iia":  hist[-1]["test"]["iia"],
        }
        print(f"  >> [k={k}] RANDOM   test IIA = {hist[-1]['test']['iia']:.3f}", flush=True)

    # 5) Probe direction baseline (k=1, fixed)
    Wp = make_probe_init_weight()
    log_p = OUT_DIR / f"{task_name}_probe.json"
    _, hist = train_das(upper_stack, 1, train_pairs, test_pairs, cache, device,
                        dtype, log_path=log_p, init_weight=Wp, train=False)
    results["probe"]["1"] = {
        "train_iia": hist[-1]["train"]["iia"],
        "test_iia":  hist[-1]["test"]["iia"],
    }
    print(f"  >> [k=1]  PROBE    test IIA = {hist[-1]['test']['iia']:.3f}", flush=True)

    # Natural baseline: with NO intervention, fraction of test pairs where base's
    # natural first-token prediction matches source's natural label. By
    # construction (natural(base) != natural(source)) this should be exactly 0.
    n_ok = 0
    with torch.no_grad():
        for (b_sid, s_sid) in test_pairs:
            b = cache[b_sid]
            h = b["h_full"].to(device=device, dtype=dtype).unsqueeze(0)
            logits_nat = upper_stack(h)[0].float()
            two = torch.stack([logits_nat[ACTION_ID], logits_nat[FINAL_ID]])
            pred = int(two.argmax().item())
            if pred == cache[s_sid]["natural_label"]:
                n_ok += 1
    results["natural_no_intervention_iia"] = n_ok / len(test_pairs)
    print(f"  >> NATURAL (no intervention) test IIA = "
          f"{results['natural_no_intervention_iia']:.3f}", flush=True)

    return results


def main():
    global N_EPOCHS, LR
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=N_EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--n-pairs-per-dir", type=int, default=200)
    ap.add_argument("--task", choices=["evidence", "extractability", "all"], default="all")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    N_EPOCHS = args.epochs; LR = args.lr

    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"[load] {MODEL_PATH}  dtype={args.dtype}")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model.eval(); device = next(model.parameters()).device
    print(f"[load] device={device}  hidden_size={model.config.hidden_size}  "
          f"n_layers={model.config.num_hidden_layers}")

    upper_stack = UpperStack(model).to(device)
    for p in upper_stack.parameters(): p.requires_grad_(False)

    all_results = {"meta": {"model": MODEL_PATH, "layer": LAYER, "k_values": K_VALUES,
                            "n_epochs": N_EPOCHS, "lr": LR, "seed": SEED,
                            "n_pairs_per_dir": args.n_pairs_per_dir, "dtype": args.dtype}}

    tasks = []
    if args.task in ("evidence", "all"):
        tasks.append(("evidence", "T0", "S0"))
    if args.task in ("extractability", "all"):
        tasks.append(("extractability", "N0", "S0"))

    for tname, base_cond, source_cond in tasks:
        all_results[tname] = run_task(tname, base_cond, source_cond, model, tok,
                                      upper_stack, device, dtype,
                                      n_pairs_per_dir=args.n_pairs_per_dir)

    out = OUT_DIR / "report.json"
    json.dump(all_results, open(out, "w"), indent=2)
    print(f"\n[wrote] {out}")
    print_summary(all_results)


def print_summary(all_results):
    for tname, res in all_results.items():
        if tname == "meta": continue
        print(f"\n=== SUMMARY  task={tname} (base={res['base_cond']}, source={res['source_cond']})")
        print(f"  natural_no_intervention_iia = {res['natural_no_intervention_iia']:.3f}")
        print(f"  {'k':>4s}  {'das_test':>10s}  {'das_best':>10s}  {'random_test':>11s}  {'probe_test':>10s}")
        for k in K_VALUES:
            d = res["das"][str(k)]
            r = res["random"][str(k)]
            p = res["probe"].get("1", {}) if k == 1 else None
            ptxt = f"{p['test_iia']:>10.3f}" if p else f"{'':>10s}"
            print(f"  {k:>4d}  {d['test_iia_final']:>10.3f}  {d['test_iia_best']:>10.3f}  "
                  f"{r['test_iia']:>11.3f}  {ptxt}")


if __name__ == "__main__":
    main()
