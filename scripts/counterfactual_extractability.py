#!/usr/bin/env python3
"""Counterfactual edits + re-run (Part B).

For every wrong-stop sample flagged ``extractable_unsupported`` by Part A,
build three minimally-edited observation variants, re-run Qwen2.5-7B at the
same post-tool decision point, and measure whether the first action flips
from ``stop`` to ``search``.

Conditions:
  base                - re-run with the audit's reconstructed full observation
  replace_W           - swap every alias of W for a typed placeholder
  remove_W            - drop the sentence(s) containing any alias of W
  irrelevant_control  - drop a sentence that contains neither W nor any
                        question entity (matched count to remove_W)

Outputs (results/natural_extractability_audit/):
  counterfactual_base_results.jsonl
  counterfactual_replace_results.jsonl
  counterfactual_remove_results.jsonl
  counterfactual_control_results.jsonl
"""
import argparse, json, re, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.prompts import PromptBuilder, ACTION_TOKENS, parse_action  # noqa: E402
from eval.scorers import answer_scorer  # noqa: E402
from scripts.audit_natural_failures import (  # noqa: E402
    aliases_of, fold, norm, question_entities, split_sentences,
)


# ---------- Edits ---------------------------------------------------------
_PLACEHOLDER_BY_QTYPE = {
    "person":      "an unidentified individual",
    "place":       "an unspecified location",
    "org":         "an unspecified organization",
    "year":        "an unspecified year",
    "number":      "an unspecified amount",
    "title":       "an untitled work",
    "what_other":  "[unspecified]",
    "which_other": "[unspecified]",
    "yesno":       "[unspecified]",
    "other":       "[unspecified]",
}


def _alias_pattern(W):
    """Regex that matches any non-empty alias of W as a whole word, case-insensitive."""
    al = sorted({a for a in aliases_of(W) if a and len(a) >= 2}, key=len, reverse=True)
    if not al:
        return None
    parts = [re.escape(a) for a in al]
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(parts) + r")(?![A-Za-z0-9])",
                      flags=re.IGNORECASE)


def edit_replace_W(observation, W, qtype):
    pat = _alias_pattern(W)
    if pat is None:
        return observation, 0
    placeholder = _PLACEHOLDER_BY_QTYPE.get(qtype, "[unspecified]")
    new_obs, n = pat.subn(placeholder, observation)
    return new_obs, n


def _doc_split(observation):
    """Split observation by '\\n\\n[N] ' boundaries while preserving them.

    Returns list of (header, body) where header is e.g. '[3] Title:' and
    body is the rest of that document. The first chunk may have no header
    if the obs starts mid-text (rare)."""
    if not observation:
        return []
    parts = re.split(r"(?=\[\d+\]\s)", observation.strip())
    docs = []
    for p in parts:
        m = re.match(r"(\[\d+\]\s[^:\n]{0,80}:)\s*(.*)", p, flags=re.DOTALL)
        if m:
            docs.append((m.group(1), m.group(2)))
        else:
            docs.append(("", p))
    return docs


def _join_docs(docs):
    out = []
    for h, b in docs:
        if h:
            out.append(h + " " + b.strip())
        else:
            out.append(b.strip())
    return "\n\n".join(s for s in out if s).strip()


def _contains_alias_text(text, W):
    if not text or not W:
        return False
    pat = _alias_pattern(W)
    if pat is None:
        return False
    return bool(pat.search(text))


def edit_remove_W(observation, W):
    """Drop every sentence (within each document body) that contains an
    alias of W. Keep document headers. Returns (edited_obs, n_removed)."""
    docs = _doc_split(observation)
    n_removed = 0
    new_docs = []
    for h, body in docs:
        sents = split_sentences(body)
        kept = []
        for s in sents:
            if _contains_alias_text(s, W):
                n_removed += 1
            else:
                kept.append(s)
        new_body = " ".join(kept)
        new_docs.append((h, new_body))
    return _join_docs(new_docs), n_removed


def edit_irrelevant_control(observation, W, question, n_to_remove):
    """Drop ``n_to_remove`` sentences that contain neither W nor any
    question entity. Choose deterministically: prefer sentences from
    documents where W does NOT appear (so we don't accidentally make the
    same passage shorter as remove_W). Fall back to any non-W sentence.
    Returns (edited_obs, n_removed)."""
    if n_to_remove <= 0:
        return observation, 0
    docs = _doc_split(observation)
    qents = question_entities(question)

    def is_irrelevant(s):
        if _contains_alias_text(s, W):
            return False
        for q in qents:
            if _contains_alias_text(s, q):
                return False
        return True

    candidates = []
    for di, (_h, body) in enumerate(docs):
        sents = split_sentences(body)
        doc_has_W = any(_contains_alias_text(s, W) for s in sents)
        for si, s in enumerate(sents):
            if is_irrelevant(s):
                candidates.append((doc_has_W, di, si, len(s)))
    candidates.sort(key=lambda x: (x[0], -x[3], x[1], x[2]))
    chosen = set((di, si) for (_, di, si, _) in candidates[:n_to_remove])
    if not chosen:
        return observation, 0

    new_docs = []
    n_removed = 0
    for di, (h, body) in enumerate(docs):
        sents = split_sentences(body)
        kept = []
        for si, s in enumerate(sents):
            if (di, si) in chosen:
                n_removed += 1
            else:
                kept.append(s)
        new_docs.append((h, " ".join(kept)))
    return _join_docs(new_docs), n_removed


# ---------- Inference helpers --------------------------------------------
def _first_action_token(text):
    """search/stop/parse_fail based on whichever marker appears FIRST."""
    if not text:
        return "parse_fail"
    tl = text.lower()
    ai = tl.find("action:")
    fi = tl.find("final answer:")
    if fi == -1:
        fi = tl.find("final:")
    if ai == -1 and fi == -1:
        return "parse_fail"
    if ai == -1:
        return "stop"
    if fi == -1:
        return "search"
    return "search" if ai < fi else "stop"


def _compute_margin(logits, tokenizer):
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    tool_ids = [tokenizer.encode(t, add_special_tokens=False)[0]
                for t in ACTION_TOKENS["tool_call"]
                if tokenizer.encode(t, add_special_tokens=False)]
    fin_ids = [tokenizer.encode(t, add_special_tokens=False)[0]
               for t in ACTION_TOKENS["finish"]
               if tokenizer.encode(t, add_special_tokens=False)]
    tool_lp = torch.logsumexp(log_probs[tool_ids], 0).item() if tool_ids else -100.0
    fin_lp = torch.logsumexp(log_probs[fin_ids], 0).item() if fin_ids else -100.0
    return tool_lp - fin_lp


def run_one(rec, observation, model, tokenizer, builder, device,
            max_new_tokens=256):
    """Re-run a single sample at the post-tool decision point with the
    given (possibly edited) observation. Returns a dict with margin, the
    first action token, parsed action / final_answer, and the raw output."""
    steps = [{
        "action": "search",
        "action_input": rec["first_search_query"],
        "observation": observation,
    }]
    msgs = builder.build_full_prompt(rec["question"], steps)
    prompt_str = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(prompt_str, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    attn = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model(input_ids, attention_mask=attn)
    margin = _compute_margin(out.logits[0, -1, :], tokenizer)

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids, attention_mask=attn,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)

    parsed = parse_action(raw)
    a2 = parsed["action"]
    fa = parsed["final_answer"]
    if a2 and a2.lower() in ("search", "calculator"):
        action_type = "search"
    elif fa is not None:
        action_type = "stop"
    else:
        action_type = None

    fa_first = _first_action_token(raw)
    em = None
    if fa is not None and rec.get("gold_answer"):
        gold = rec["gold_answers"] if rec.get("gold_answers") else [rec["gold_answer"]]
        # use any gold alias; mode=exact (matches CLAUDE.md scoring guidance)
        em = int(any(answer_scorer(fa, g, mode="exact")["matched"] for g in gold))

    # contains_W in the new final_answer (only meaningful when stop)
    W = rec.get("emitted_answer_W") or ""
    contains_W = int(bool(W) and W.lower() in (fa or "").lower()) if fa else 0

    return {
        "sample_id": rec["sample_id"],
        "margin": margin,
        "action_type": action_type,
        "first_action_token": fa_first,
        "action2": a2,
        "final_answer": fa,
        "em": em,
        "contains_W": contains_W,
        "obs_chars": len(observation or ""),
        "raw_output": raw[:400],
    }



# ---------- Main ----------------------------------------------------------
def _build_variants(rec):
    obs = rec["observation_full"]
    W = rec["emitted_answer_W"]
    qtype = rec.get("question_type") or "other"
    obs_replace, n_repl = edit_replace_W(obs, W, qtype)
    obs_remove, n_rem = edit_remove_W(obs, W)
    obs_control, n_ctrl = edit_irrelevant_control(obs, W, rec["question"], n_rem)
    return {
        "base":               (obs,         {"n_edits": 0}),
        "replace_W":          (obs_replace, {"n_edits": n_repl,
                                             "placeholder": _PLACEHOLDER_BY_QTYPE.get(qtype, "[unspecified]")}),
        "remove_W":           (obs_remove,  {"n_edits": n_rem}),
        "irrelevant_control": (obs_control, {"n_edits": n_ctrl}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-raw",
                    default="results/natural_extractability_audit/natural_audit_raw.jsonl")
    ap.add_argument("--out-dir",
                    default="results/natural_extractability_audit")
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of edited samples (smoke test).")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] loading audit raw: {args.audit_raw}")
    audit = [json.loads(l) for l in open(args.audit_raw)]
    targets = [r for r in audit
               if r["category"] == "step1_stop_wrong"
               and r.get("extractable_unsupported")]
    print(f"  total audit records: {len(audit)}; extractable_unsupported wrong-stops: {len(targets)}")
    if args.limit:
        targets = targets[:args.limit]
        print(f"  limit applied: {len(targets)} samples")

    # Pre-build edit variants and skip samples where remove_W actually removed
    # nothing (W detected by alias-fold but our doc-split missed it).
    plans = []
    skipped_no_remove = 0
    for r in targets:
        variants = _build_variants(r)
        if variants["remove_W"][1]["n_edits"] == 0:
            skipped_no_remove += 1
            continue
        plans.append((r, variants))
    print(f"  usable samples (remove_W edited >=1 sentence): {len(plans)} "
          f"(skipped {skipped_no_remove} where W not in any extracted sentence)")

    print(f"[2/4] loading model: {args.model_path}")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  model on {device}")

    builder = PromptBuilder(tools=["search"])

    print(f"[3/4] running {len(plans)} samples x 4 conditions ...")
    out_paths = {
        "base":               out_dir / "counterfactual_base_results.jsonl",
        "replace_W":          out_dir / "counterfactual_replace_results.jsonl",
        "remove_W":           out_dir / "counterfactual_remove_results.jsonl",
        "irrelevant_control": out_dir / "counterfactual_control_results.jsonl",
    }
    files = {k: open(p, "w") for k, p in out_paths.items()}
    t0 = time.time()
    try:
        for i, (rec, variants) in enumerate(plans, 1):
            for cond, (obs, meta) in variants.items():
                res = run_one(rec, obs, model, tok, builder, device,
                              max_new_tokens=args.max_new_tokens)
                res["condition"] = cond
                res["edit_meta"] = meta
                res["question"] = rec["question"]
                res["gold_answer"] = rec["gold_answer"]
                res["question_type"] = rec.get("question_type")
                res["emitted_answer_W"] = rec["emitted_answer_W"]
                res["audit_step1_final_answer"] = rec.get("step1_final_answer")
                res["audit_step1_action"] = rec.get("step1_action")
                files[cond].write(json.dumps(res, ensure_ascii=False) + "\n")
                files[cond].flush()
            if i % 10 == 0 or i == len(plans):
                dt = time.time() - t0
                rate = i / dt if dt > 0 else 0
                eta = (len(plans) - i) / rate if rate > 0 else 0
                print(f"  [{i}/{len(plans)}] elapsed={dt:.0f}s rate={rate:.2f}/s eta={eta:.0f}s")
    finally:
        for f in files.values():
            f.close()

    print(f"[4/4] wrote {len(plans)} records per condition to:")
    for k, p in out_paths.items():
        print(f"  {k:<20s} {p}")


if __name__ == "__main__":
    main()
