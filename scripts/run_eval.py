#!/usr/bin/env python3
"""
Unified E2E runner: any dataset × any policy → samples.jsonl + summary.json.

Key features over run_popqa_e2e.py:
 - Step-aware JES tau scheduling (--jes-tau-schedule "1:3.0,2+:0.5")
 - Do-no-harm guard (--enable-guard)
 - Supports popqa / gsm8k / math datasets
 - Supports baseline / force_adopt / force_reject / jes / fixed_rho policies

Usage:
  # Direction defaults are dataset-specific (best-practice):
  #   PopQA -> direction_search_v3.npz
  #   GSM8K/MATH -> direction_calculator_v1.npz
  python scripts/run_eval.py \
      --dataset popqa --data-path data/popqa/popqa_test.jsonl \
      --corpus-path data/popqa/corpus.jsonl \
      --policy jes --jes-tau-schedule "1:3.0,2+:0.5" --enable-guard \
      --n-samples 500 --out results/popqa_jes_scheduled
"""
import json, argparse, numpy as np, time
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.popqa import PopQADataset, build_popqa_corpus
from datasets.gsm8k import GSM8KDataset
from datasets.math import MathDataset
from tools.search_tool import SearchTool
from tools.calculator_tool import CalculatorTool
from agent.react_loop import ReActAgent, AgentConfig
from agent.policies import (BaselinePolicy, ForcedPolicy, JESPolicy,
                            FixedRhoPolicy, Policy, SteeringDecision)
from steering.jes import JESConfig, JESController
from steering.directions import load_direction
from eval.unified_output import (convert_episode_to_record, compute_run_summary,
                                 write_records, write_summary, make_run_id,
                                 load_records)


# Best-practice defaults: tool-specific directions
DEFAULT_DIRECTION_BY_DATASET = {
    "popqa": "steering/directions/direction_search_v3.npz",
    "gsm8k": "steering/directions/direction_calculator_v1.npz",
    "math": "steering/directions/direction_calculator_v1.npz",
}


def default_direction_for_dataset(dataset: str) -> str:
    if dataset not in DEFAULT_DIRECTION_BY_DATASET:
        raise ValueError(f"No default direction registered for dataset={dataset!r}")
    return DEFAULT_DIRECTION_BY_DATASET[dataset]


# ---------------------------------------------------------------------------
# Step-aware JES policy + do-no-harm guard
# ---------------------------------------------------------------------------

def parse_tau_schedule(spec: str) -> dict:
    """Parse '1:3.0,2+:0.5' → {1: 3.0, 'rest': 0.5}."""
    schedule = {}
    for part in spec.split(","):
        k, v = part.split(":")
        k = k.strip()
        v = float(v.strip())
        if k.endswith("+"):
            schedule["rest"] = v
        else:
            schedule[int(k)] = v
    return schedule


class StepAwareJESPolicy(Policy):
    """JES with per-step tau schedule + optional do-no-harm guard."""

    def __init__(self, base_config: JESConfig, direction: np.ndarray,
                 tau_schedule: dict = None,
                 guard_enabled: bool = False, guard_threshold: float = -1.0):
        self._base = base_config
        self._direction = direction
        self._tau_schedule = tau_schedule or {}
        self.guard_enabled = guard_enabled
        self.guard_threshold = guard_threshold
        self.step = 0
        self.guard_triggered_steps = []

    @property
    def name(self) -> str:
        return "jes"

    def reset_episode(self):
        self.step = 0
        self.guard_triggered_steps = []

    def _tau_for_step(self, step: int) -> float:
        if step in self._tau_schedule:
            return self._tau_schedule[step]
        return self._tau_schedule.get("rest", self._base.tau)

    def decide(self, margin_fn, target_side, hidden_rms, direction_rms):
        self.step += 1
        tau = self._tau_for_step(self.step)

        # Cache wrapper to avoid redundant forward passes
        _cache = {}
        def cfn(rho):
            key = round(rho, 8)
            if key not in _cache:
                _cache[key] = margin_fn(rho)
            return _cache[key]

        m0 = cfn(0.0)

        # Do-no-harm guard: if step>1 and model strongly prefers finish → pass through
        if self.guard_enabled and self.step > 1 and m0 < self.guard_threshold:
            self.guard_triggered_steps.append(self.step)
            return SteeringDecision(
                rho=0.0, alpha=0.0, policy_name="jes", m_before=m0,
                details={"guard_triggered": True, "step": self.step,
                         "tau_effective": tau, "m_before": m0,
                         "already_satisfied": False, "saturated": False})

        cfg = JESConfig(tau=tau, eps=self._base.eps, max_rho=self._base.max_rho,
                        slope_min=self._base.slope_min, alpha_max=self._base.alpha_max)
        ctrl = JESController(cfg, self._direction, hidden_rms, direction_rms)
        res = ctrl.compute_steering(cfn, target_side)

        details = res.to_dict()
        details["step"] = self.step
        details["tau_effective"] = tau
        details["guard_triggered"] = False
        return SteeringDecision(rho=res.rho_used, alpha=res.alpha_used,
                                policy_name="jes", m_before=res.m_before,
                                details=details)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    print(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True)
    mdl.eval()
    return mdl, tok


# ---------------------------------------------------------------------------
# Dataset + tool helpers
# ---------------------------------------------------------------------------

def load_dataset_and_tools(args):
    """Return (samples, tools_dict) based on --dataset."""
    if args.dataset == "popqa":
        if not args.data_path:
            raise ValueError("--data-path is required for popqa")
        ds = PopQADataset(args.data_path)
        if args.pop_limit is not None:
            samples = ds.get_subset_by_popularity(args.n_samples,
                                                  max_pop=args.pop_limit, seed=args.seed)
        else:
            samples = ds.get_subset(args.n_samples, seed=args.seed)
        corpus_path = Path(args.corpus_path)
        if not corpus_path.exists():
            build_popqa_corpus(args.data_path, str(corpus_path))
        search = SearchTool(str(corpus_path), top_k=3)
        calc = CalculatorTool()
        tools = {"search": search, "calculator": calc}
        gold_fn = lambda s: s.answers  # list of acceptable answers
    elif args.dataset == "gsm8k":
        gsm_hard = getattr(args, 'gsm_hard', False)
        ds = GSM8KDataset(args.data_path, gsm_hard=gsm_hard)
        samples = ds.get_subset(args.n_samples, seed=args.seed)
        calc = CalculatorTool()
        tools = {"calculator": calc}
        gold_fn = lambda s: s.answer  # single numeric string
    elif args.dataset == "math":
        ds = MathDataset(args.data_path)
        samples = ds.get_subset(args.n_samples, seed=args.seed)
        calc = CalculatorTool()
        tools = {"calculator": calc}
        gold_fn = lambda s: s.answer
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    variant = f"{args.dataset}(GSM-Hard)" if getattr(args, 'gsm_hard', False) else args.dataset
    print(f"Selected {len(samples)} {variant} samples")
    return samples, tools, gold_fn


# ---------------------------------------------------------------------------
# Policy creation
# ---------------------------------------------------------------------------

def create_policy(args, direction, jes_config):
    """Create policy from CLI args."""
    if args.policy == "baseline":
        return BaselinePolicy()
    elif args.policy == "force_adopt":
        return ForcedPolicy(force_adopt=True)
    elif args.policy == "force_reject":
        return ForcedPolicy(force_adopt=False)
    elif args.policy == "fixed_rho":
        return FixedRhoPolicy(rho=args.fixed_rho)
    elif args.policy == "jes":
        tau_sched = {}
        if args.jes_tau_schedule:
            tau_sched = parse_tau_schedule(args.jes_tau_schedule)
        return StepAwareJESPolicy(
            base_config=jes_config,
            direction=direction,
            tau_schedule=tau_sched,
            guard_enabled=args.enable_guard,
            guard_threshold=args.guard_threshold,
        )
    else:
        raise ValueError(f"Unknown policy: {args.policy}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified E2E runner: dataset × policy → samples.jsonl + summary.json")
    parser.add_argument("--dataset", choices=["popqa", "gsm8k", "math"], required=True)
    parser.add_argument("--data-path", default=None,
                        help="Dataset JSONL path. For gsm8k/math, omit (or pass '') to load from HuggingFace.")
    parser.add_argument("--corpus-path", default="data/popqa/corpus.jsonl")
    parser.add_argument(
        "--direction-path",
        default=None,
        help=(
            "Direction NPZ path. Default is dataset-specific: "
            "popqa->direction_search_v3, gsm8k/math->direction_calculator_v1."
        ),
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--policy", required=True,
                        choices=["baseline", "force_adopt", "force_reject", "jes", "fixed_rho"])
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    # JES params
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--rho-max", type=float, default=0.75)
    parser.add_argument("--jes-tau-schedule", default=None,
                        help='Step-aware tau schedule, e.g. "1:3.0,2+:0.5"')
    parser.add_argument("--enable-guard", action="store_true",
                        help="Enable do-no-harm guard")
    parser.add_argument("--guard-threshold", type=float, default=-1.0,
                        help="Margin threshold for guard (default: -1.0)")
    # Fixed rho
    parser.add_argument("--fixed-rho", type=float, default=0.3,
                        help="rho value for fixed_rho policy")
    # PopQA-specific
    parser.add_argument("--pop-limit", type=int, default=None)
    # GSM8K-specific
    parser.add_argument("--gsm-hard", action="store_true",
                        help="Use GSM-Hard (large-number variant) for GSM8K dataset")
    # Corruption (optional)
    parser.add_argument("--corruption-p", type=float, default=0.0,
                        help="Corruption probability for tool outputs")
    parser.add_argument("--corruption-mode", default="random",
                        choices=["random", "empty", "noise", "counterfactual"])
    # Scoring
    parser.add_argument("--score-mode", default=None,
                        choices=["numeric", "any", "exact"],
                        help="Answer scoring mode. Default is dataset-aware: "
                             "numeric for gsm8k/math, any for popqa.")
    # Output
    parser.add_argument("--out", required=True, help="Output directory (not file path)")
    args = parser.parse_args()

    # Resolve dataset-aware score_mode default
    if args.score_mode is None:
        args.score_mode = "numeric" if args.dataset in ("gsm8k", "math") else "any"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)  # out is a directory

    # Load dataset + tools (does not require model)
    samples, tools, gold_fn = load_dataset_and_tools(args)

    # Select direction (best-practice dataset-specific default)
    direction_path = args.direction_path or default_direction_for_dataset(args.dataset)
    if args.direction_path is None:
        print(f"[run_eval] Using default direction for {args.dataset}: {direction_path}")

    # Load model + direction
    model, tokenizer = load_model_and_tokenizer(args.model)
    direction, _ = load_direction(direction_path)
    direction_rms = float(np.sqrt(np.mean(direction ** 2)))

    # Apply corruption if requested
    if args.corruption_p > 0:
        from tools.corruption import CorruptionWrapper, CorruptionConfig
        for tname in list(tools):
            cfg = CorruptionConfig(probability=args.corruption_p,
                                   mode=args.corruption_mode, seed=args.seed)
            tools[tname] = CorruptionWrapper(tools[tname], cfg)

    # Agent
    agent_config = AgentConfig(
        max_steps=args.max_steps, layer=args.layer, position=args.position,
        score_mode=args.score_mode)
    agent = ReActAgent(model, tokenizer, tools, agent_config,
                       direction=direction, direction_rms=direction_rms)

    jes_config = JESConfig(tau=args.tau, eps=args.eps, max_rho=args.rho_max)
    jes_params = {"tau": args.tau, "eps": args.eps, "rho_max": args.rho_max,
                  "layer": args.layer, "pos": args.position,
                  "direction_path": direction_path}
    if args.jes_tau_schedule:
        jes_params["tau_schedule"] = args.jes_tau_schedule
    if args.enable_guard:
        jes_params["guard_threshold"] = args.guard_threshold

    policy = create_policy(args, direction, jes_config)
    target_side = "positive"  # "should use tool"

    # ---- Run ----
    run_id = make_run_id(args.dataset, args.policy, len(samples), args.seed)

    # Optional paired context (for correct regression/rescue accounting and flags)
    baseline_records = None
    force_adopt_records = None
    bl_ok_by_id = {}
    fa_ok_by_id = {}
    if args.policy != "baseline":
        bl_path = out / "baseline.jsonl"
        if bl_path.exists():
            baseline_records = load_records(str(bl_path))
            bl_ok_by_id = {
                r["sample_id"]: bool(r.get("is_correct", False))
                for r in baseline_records
            }

        # Only needed for other policies' flags/subset breakdown.
        fa_path = out / "force_adopt.jsonl"
        if args.policy != "force_adopt" and fa_path.exists():
            force_adopt_records = load_records(str(fa_path))
            fa_ok_by_id = {
                r["sample_id"]: bool(r.get("is_correct", False))
                for r in force_adopt_records
            }

    records = []
    print(f"\n{'='*50}  {args.policy} on {args.dataset}  {'='*50}")
    for sample in tqdm(samples, desc=args.policy):
        # Reset per-episode state for step-aware policy
        if hasattr(policy, "reset_episode"):
            policy.reset_episode()
        # Reset corruption RNG for deterministic per-sample corruption
        for t in tools.values():
            if hasattr(t, "reset_for_sample"):
                t.reset_for_sample(str(sample.id))

        gold = gold_fn(sample)
        result = agent.run(
            question=sample.question, policy=policy,
            gold_answer=gold, episode_id=sample.id,
            target_side=target_side)
        ep = result.to_dict()

        sid = str(sample.id)
        rec = convert_episode_to_record(
            ep, run_id=run_id, dataset=args.dataset,
            jes_params=jes_params if args.policy == "jes" else None,
            baseline_success=bl_ok_by_id.get(sid) if baseline_records else None,
            # Only meaningful when comparing a policy to force_adopt.
            force_adopt_success=fa_ok_by_id.get(sid) if force_adopt_records else None,
        )
        # Enrich with guard info
        if hasattr(policy, "guard_triggered_steps") and policy.guard_triggered_steps:
            rec["guard_triggered_steps"] = list(policy.guard_triggered_steps)
        records.append(rec)

    # ---- Output ----
    write_records(records, str(out / f"{args.policy}.jsonl"))

    # For subset breakdown, we want the *current* force_adopt records.
    force_adopt_for_summary = records if args.policy == "force_adopt" else force_adopt_records
    summ = compute_run_summary(
        records,
        baseline_records=baseline_records,
        force_adopt_records=force_adopt_for_summary,
    )
    # Extra JES stats
    if args.policy == "jes":
        steering_count = sum(
            1 for r in records for d in r.get("decision_trace", [])
            if d.get("rho", 0) != 0)
        total_decisions = sum(len(r.get("decision_trace", [])) for r in records)
        summ["steering_rate"] = steering_count / total_decisions if total_decisions else 0
        # Guard metrics:
        # - episode_rate: how often guard triggered at least once in an episode
        # - step_rate: how often guard triggered per decision point
        n_guard_eps = sum(1 for r in records if r.get("guard_triggered_steps"))
        n_guard_steps = sum(len(r.get("guard_triggered_steps") or []) for r in records)
        summ["guard_trigger_episode_rate"] = n_guard_eps / len(records) if records else 0
        summ["guard_trigger_step_rate"] = n_guard_steps / total_decisions if total_decisions else 0
        # Back-compat: keep the old key but make semantics explicit.
        summ["guard_trigger_rate"] = summ["guard_trigger_episode_rate"]
    write_summary(summ, str(out / f"{args.policy}_summary.json"))

    print(f"\nResults: {out}/{args.policy}.jsonl  ({len(records)} samples)")
    print(f"Success: {summ['success_rate']:.1%}  "
          f"AvgTokens: {summ['avg_tokens_total']:.0f}  "
          f"AvgToolCalls: {summ['avg_tool_calls']:.2f}")


if __name__ == "__main__":
    main()

