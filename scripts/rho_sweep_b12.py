#!/usr/bin/env python3
"""Step 2 of the candidate-lock-depth diagnostic: rho dose-response on B1+B2.

Runs 8 steered conditions (A3 dose sweep + random/evidence-parallel controls) on a
fixed 50-B1 + 50-B2 stratified subset of the natural-failure audit. Baseline is
reused from results/l20_rho020_n500/baseline_results.jsonl (FreeGen, no override).

Sign convention: rho is given as positive magnitude; negative is applied
internally so steering pushes toward search (matches A3 main-result convention).
alpha_max is raised to 64 so the rho dose isn't artificially clipped.
"""
from __future__ import annotations
import os, sys, json, time, random
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

sys.path.insert(0, '.')
from tqdm import tqdm
from agent.react_loop import ReActAgent, AgentConfig
from agent.policies_verify import FixedRhoSteerPolicy
from datasets.hotpotqa import HotpotQADataset
from tools.search_tool import SearchTool
from steering.directions import load_direction
from scripts.run_verify_critical_pipeline import run_episode
from scripts.steering_by_bucket import assign_bucket

ROOT = Path('results/l20_rho020_n500')
AUDIT_PATH = Path('results/natural_extractability_audit/natural_audit_raw.jsonl')
OUT_DIR = Path('results/candidate_lock_depth')
DIRECTIONS = {
    'a3':     'steering/directions/direction_search_v3_layer20.npz',
    'random': 'steering/directions/direction_random_seed42.npz',
    'evpar':  'steering/directions/direction_decomp_parallel_layer20.npz',
}
CONDITIONS = [
    ('a3', 0.20), ('a3', 0.50), ('a3', 1.00), ('a3', 1.50),
    ('random', 1.00), ('random', 1.50),
    ('evpar', 1.00), ('evpar', 1.50),
]
LAYER = 20
ALPHA_MAX = 64.0
SEED = 42
N_PER_BUCKET = 50
MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'


def select_b12(audit, base_ids):
    rng = random.Random(SEED)
    b1, b2 = [], []
    for sid in sorted(base_ids):
        if sid not in audit: continue
        b = assign_bucket(audit[sid])
        if b == 1: b1.append(sid)
        elif b == 2: b2.append(sid)
    return sorted(rng.sample(b1, N_PER_BUCKET)), sorted(rng.sample(b2, N_PER_BUCKET))


def _stats(records, audit, bucket_of):
    n = len(records)
    if n == 0: return {}
    n_search = n_2nd = n_commit = n_em = n_pf = n_fmt = 0
    mb_list, ma_list = [], []
    for r in records:
        steps = r.get('steps') or []
        actions = [s.get('action') for s in steps]
        if any(a == 'search' for a in actions): n_search += 1
        if sum(1 for a in actions if a == 'search') >= 2: n_2nd += 1
        if any(s.get('parse_failure_reason') for s in steps): n_pf += 1
        ans = (r.get('final_answer') or '').strip().lower()
        au = audit.get(r['sample_id']) or {}
        W = (au.get('emitted_answer_W') or '').strip().lower()
        if W and ans and (ans == W or W in ans or ans in W):
            n_commit += 1
        if r.get('em_correct'): n_em += 1
        # format drift: any generation step whose text has no Action:/Final Answer: anchor
        # (action is None signals the parser couldn't extract either)
        for s in steps:
            txt = s.get('raw_model_text') or ''
            if txt and s.get('action') is None and s.get('final_answer') is None \
                    and 'Action:' not in txt and 'Final Answer:' not in txt:
                n_fmt += 1; break
        if len(steps) > 1:
            stg = steps[1].get('steering') or {}
            mb = stg.get('m_before')
            if mb is None: mb = steps[1].get('margin_before')
            ma = stg.get('m_after')
            if mb is not None: mb_list.append(float(mb))
            if ma is not None: ma_list.append(float(ma))
    out = {
        'n': n,
        'search_rate': n_search / n,
        'second_search_rate': n_2nd / n,
        'commit_W_rate': n_commit / n,
        'em_correct_rate': n_em / n,
        'parse_failure_rate': n_pf / n,
        'format_drift_rate': n_fmt / n,
        'mean_m_before': float(np.mean(mb_list)) if mb_list else None,
        'mean_m_after':  float(np.mean(ma_list)) if ma_list else None,
    }
    if mb_list and ma_list and len(mb_list) == len(ma_list):
        out['mean_delta_margin'] = float(np.mean(np.array(ma_list) - np.array(mb_list)))
    return out


def aggregate(records, audit, bucket_of):
    return {
        'overall': _stats(records, audit, bucket_of),
        'b1': _stats([r for r in records if bucket_of.get(r['sample_id']) == 1], audit, bucket_of),
        'b2': _stats([r for r in records if bucket_of.get(r['sample_id']) == 2], audit, bucket_of),
    }


def print_table(summaries):
    keys = ['n', 'search_rate', 'second_search_rate', 'commit_W_rate', 'em_correct_rate',
            'parse_failure_rate', 'format_drift_rate', 'mean_m_before', 'mean_m_after']
    print()
    print(f'{"condition":<20} {"n":>4} {"sr":>6} {"2sr":>6} {"cW":>6} {"em":>6} {"pf":>5} {"fmt":>5} {"mB":>6} {"mA":>6}')
    for tag, agg in summaries.items():
        s = agg['overall']
        print(f'{tag:<20} {s["n"]:>4} {s["search_rate"]*100:>5.1f}% {s["second_search_rate"]*100:>5.1f}% '
              f'{s["commit_W_rate"]*100:>5.1f}% {s["em_correct_rate"]*100:>5.1f}% '
              f'{s["parse_failure_rate"]*100:>4.1f}% {s["format_drift_rate"]*100:>4.1f}% '
              f'{(s["mean_m_before"] or 0):>+6.2f} {(s["mean_m_after"] or 0):>+6.2f}')


def apply_stopping_rule(summaries):
    # baseline forces step-1 search, so search_rate is always 100%.
    # the meaningful behavioral output is whether the agent re-searches after the
    # first observation (second_search_rate).
    metric = 'second_search_rate'
    base = summaries['baseline']['overall'][metric]
    a3_high = max(summaries[t]['overall'][metric]
                  for t in ['a3_rho1.00', 'a3_rho1.50'])
    rnd_high = max(summaries[t]['overall'][metric]
                   for t in ['random_rho1.00', 'random_rho1.50'])
    ep_high = max(summaries[t]['overall'][metric]
                  for t in ['evpar_rho1.00', 'evpar_rho1.50'])
    pf_a3 = max(summaries[t]['overall']['parse_failure_rate']
                for t in ['a3_rho1.00', 'a3_rho1.50'])
    deep_basin_confirmed = (a3_high - rnd_high >= 0.20
                            and a3_high - ep_high >= 0.20
                            and pf_a3 < 0.20)
    print(f'\nStopping rule (metric=second_search_rate): base={base:.1%} '
          f'A3_high={a3_high:.1%} rnd_high={rnd_high:.1%} ep_high={ep_high:.1%} '
          f'pf_a3={pf_a3:.1%} -> deep_basin_confirmed={deep_basin_confirmed}')
    return deep_basin_confirmed


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = {r['sample_id']: r for r in (json.loads(l) for l in open(AUDIT_PATH))}
    base_results = {r['sample_id']: r for r in (json.loads(l) for l in open(ROOT/'baseline_results.jsonl'))}
    sids_b1, sids_b2 = select_b12(audit, base_results.keys())
    bucket_of = {sid: 1 for sid in sids_b1}
    bucket_of.update({sid: 2 for sid in sids_b2})
    target_ids = sorted(set(sids_b1) | set(sids_b2))
    print(f'Selected B1={len(sids_b1)} B2={len(sids_b2)} (total={len(target_ids)})')

    json.dump({'b1': sids_b1, 'b2': sids_b2, 'bucket_of': bucket_of, 'seed': SEED,
               'alpha_max': ALPHA_MAX, 'layer': LAYER},
              open(OUT_DIR/'selected_ids_b12.json', 'w'), indent=2)

    print('Loading model...')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    model.eval()

    print('Loading dataset...')
    dataset = HotpotQADataset('data/hotpotqa/hotpot_dev_distractor_v1.json')
    sample_by_id = {s.id: s for s in dataset.samples}
    samples = [sample_by_id[sid] for sid in target_ids]
    assert len(samples) == len(target_ids)

    search_tool = SearchTool(corpus_path='data/hotpotqa/corpus.jsonl')
    config = AgentConfig(max_steps=5, max_tokens_per_step=256, temperature=0.0,
                         layer=LAYER, tools=['search'], score_mode='exact')
    first_dir, _ = load_direction(DIRECTIONS['a3'], normalize_rms=1.0)
    first_rms = float(np.sqrt(np.mean(first_dir ** 2)))
    agent = ReActAgent(model=model, tokenizer=tok, tools={'search': search_tool},
                       config=config, direction=first_dir, direction_rms=first_rms)

    raw_path = OUT_DIR / 'rho_sweep_b12.jsonl'
    raw_f = open(raw_path, 'w')
    summaries = {}

    bl_subset = [base_results[sid] for sid in target_ids]
    for r in bl_subset:
        raw_f.write(json.dumps({'condition': 'baseline', 'direction': None, 'rho': 0.0,
                                'sample_id': r['sample_id'], 'bucket': bucket_of[r['sample_id']],
                                'episode': r}, ensure_ascii=False) + '\n')
    summaries['baseline'] = aggregate(bl_subset, audit, bucket_of)
    print(f'baseline overall: {summaries["baseline"]["overall"]}')

    t0 = time.time()
    for dir_name, rho_mag in CONDITIONS:
        cond_tag = f'{dir_name}_rho{rho_mag:.2f}'
        print(f'\n=== {cond_tag} (elapsed {(time.time()-t0)/60:.1f} min) ===')
        direction, _ = load_direction(DIRECTIONS[dir_name], normalize_rms=1.0)
        agent.direction = direction
        agent.direction_rms = float(np.sqrt(np.mean(direction ** 2)))
        rho = -float(rho_mag)  # negative = push toward search
        policy = FixedRhoSteerPolicy(rho=rho, steer_step=1, alpha_max=ALPHA_MAX,
                                     decision_only=True)
        cond_results = []
        for s in tqdm(samples, desc=cond_tag):
            r = run_episode(agent, s, policy, 'exact')
            cond_results.append(r)
            raw_f.write(json.dumps({'condition': cond_tag, 'direction': dir_name,
                                    'rho': rho, 'alpha_max': ALPHA_MAX,
                                    'sample_id': s.id, 'bucket': bucket_of[s.id],
                                    'episode': r}, ensure_ascii=False) + '\n')
            raw_f.flush()
        summaries[cond_tag] = aggregate(cond_results, audit, bucket_of)
        print(f'  overall: search={summaries[cond_tag]["overall"]["search_rate"]:.1%} '
              f'commitW={summaries[cond_tag]["overall"]["commit_W_rate"]:.1%} '
              f'em={summaries[cond_tag]["overall"]["em_correct_rate"]:.1%} '
              f'pf={summaries[cond_tag]["overall"]["parse_failure_rate"]:.1%}')

    raw_f.close()

    summary_obj = {
        'timestamp': datetime.now().isoformat(),
        'config': {'layer': LAYER, 'alpha_max': ALPHA_MAX, 'seed': SEED,
                   'n_b1': N_PER_BUCKET, 'n_b2': N_PER_BUCKET,
                   'directions': DIRECTIONS, 'conditions': CONDITIONS,
                   'sign_convention': 'negative rho pushes toward search'},
        'summaries': summaries,
    }
    json.dump(summary_obj, open(OUT_DIR/'rho_sweep_b12_summary.json', 'w'),
              indent=2, default=str)
    print_table(summaries)
    apply_stopping_rule(summaries)
    print(f'\nWrote {raw_path} and rho_sweep_b12_summary.json')


if __name__ == '__main__':
    main()
