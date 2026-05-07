#!/usr/bin/env python3
"""External-validity replication of the N0/T0/S0 extractability-support toggle on
a second benchmark (MuSiQue 2-hop bridge).

Defensive cross-benchmark check, not a new benchmark paper.

Differences vs eval_extractability_cross_model.py:
  - Reads results/second_benchmark_extractability/pairs.jsonl by default.
  - Uses CLEAN_SYSTEM_PROMPT (drops the "Your first word must be Action or Final"
    line). The HotpotQA toggle was rerun with the cleaner prompt; we keep that
    same prompt here so the cross-benchmark comparison is apples-to-apples.
  - Everything else (label_margin, parse_action, run_one's structure, R1 path,
    Robustness A/B variants) is reused from the cross-model script.
"""
import argparse, json, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))   # for `agent.prompts`
sys.path.insert(0, str(_HERE))          # for `eval_extractability_cross_model`
from agent.prompts import TOOL_DESCRIPTIONS                                       # noqa: E402

# Reuse all the math / parsing helpers from the cross-model script.
import eval_extractability_cross_model as xm                                      # noqa: E402
from eval_extractability_cross_model import (                                     # noqa: E402
    label_margin, run_one as _run_one_xm, apply_chat_template_safe,
    is_r1_model, PROMPT_TAILS, to_natural_snippet,
)


# Cleaner system prompt: same two-shape contract as DEFAULT_SYSTEM_PROMPT but
# without the "Your first word must be Action or Final" line, which leaks the
# decision boundary into the prompt and biases the margin.
CLEAN_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using available tools.

Available tools:
{tool_descriptions}

You MUST respond in exactly one of the following formats.

If you need to use a tool:
Action: <tool_name>
Action Input: <input>

If you can answer directly:
Final Answer: <answer>"""


def build_messages_clean(question, observation, prompt_variant="v1", obs_style="factcard"):
    sys_p = CLEAN_SYSTEM_PROMPT.format(
        tool_descriptions="- " + TOOL_DESCRIPTIONS["search"]
    )
    if obs_style == "snippet":
        observation = to_natural_snippet(observation)
    tail = PROMPT_TAILS[prompt_variant]
    user = (
        f"{question}\n\n"
        f"I have already run a search for you.\n"
        f"Tool: search\n"
        f"Tool input: about: {question[:80]}\n"
        f"Tool result:\n{observation}\n\n"
        f"{tail}"
    )
    return [{"role": "system", "content": sys_p},
            {"role": "user",   "content": user}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs",
        default="results/second_benchmark_extractability/pairs.jsonl")
    ap.add_argument("--out", default=None,
        help="Output JSONL path (single-config mode).")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--adapter-path", default=None,
        help="Optional PEFT adapter dir to load on top of --model-path.")
    ap.add_argument("--conditions", nargs="+", default=["N0", "T0", "S0"])
    ap.add_argument("--max-new-tokens", type=int, default=None,
        help="Default 256 for non-R1, 1200 for R1.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt-variant", choices=["v1", "v2", "v3"], default="v1")
    ap.add_argument("--obs-style", choices=["factcard", "snippet"], default="factcard")
    ap.add_argument("--multi-configs", nargs="+", default=None,
        help="Format: 'variant:style:OUT_PATH'. Overrides --out / --prompt-variant / --obs-style.")
    args = ap.parse_args()

    is_r1 = is_r1_model(args.model_path)
    if args.max_new_tokens is None:
        args.max_new_tokens = 1200 if is_r1 else 256

    rows = [json.loads(l) for l in open(args.pairs)]
    rows = [r for r in rows
            if (r.get("condition") or r.get("condition_id")) in args.conditions]
    if args.limit: rows = rows[:args.limit]

    if args.multi_configs:
        configs = []
        for spec in args.multi_configs:
            v, s, p = spec.split(":", 2)
            configs.append((v, s, p))
    else:
        if not args.out:
            ap.error("--out required when --multi-configs not given")
        configs = [(args.prompt_variant, args.obs_style, args.out)]

    print(f"[info] loading {args.model_path}; {len(rows)} records; "
          f"conds={args.conditions}; is_r1={is_r1}; "
          f"max_new_tokens={args.max_new_tokens}; n_configs={len(configs)}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    if args.adapter_path:
        from peft import PeftModel
        print(f"[info] loading adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
    model.eval()
    device = next(model.parameters()).device

    # Monkey-patch the build_messages used inside run_one with the clean version.
    xm.build_messages = build_messages_clean

    for variant, style, out_path in configs:
        out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"[cfg] variant={variant} style={style} -> {out_path}")
        with open(out, "w") as f:
            for i, rec in enumerate(rows, 1):
                row = _run_one_xm(rec, model, tok, device, args.max_new_tokens,
                                  is_r1=is_r1, prompt_variant=variant,
                                  obs_style=style)
                row["dataset"] = "musique_2hop_bridge"
                row["system_prompt"] = "clean_no_first_word"
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                if i % 10 == 0 or i == len(rows):
                    print(f"  [{i}/{len(rows)}] {time.time()-t0:.1f}s")
        print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
