#!/usr/bin/env python3
"""Step 3 of the candidate-lock-depth diagnostic: anchor x A3 interaction.

For each of the 100 stratified B1+B2 samples used in Step 2, run the agent
under 7 conditions:

  (1) original                           (no steering)
  (2) original          + A3 rho=0.20
  (3) original          + A3 rho=0.50
  (4) replace_W                          (no steering)
  (5) replace_W         + A3 rho=0.20
  (6) replace_W         + A3 rho=0.50
  (7) irrelevant_control + A3 rho=0.50

The edited observation is injected as the FIRST observation returned to the
agent at step 1 (the model's natural first search query is preserved).
Subsequent searches (if any) defer to the real BM25 SearchTool.

Sign convention: rho is given as positive magnitude; negative is applied
internally so steering pushes toward search.
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
from agent.policies_verify import FixedRhoSteerPolicy, FreeGenBaselinePolicy
from datasets.hotpotqa import HotpotQADataset
from tools.search_tool import SearchTool
from steering.directions import load_direction
from scripts.run_verify_critical_pipeline import run_episode
from scripts.counterfactual_extractability import (
    edit_replace_W, edit_irrelevant_control, edit_remove_W, _PLACEHOLDER_BY_QTYPE,
)

AUDIT_PATH = Path('results/natural_extractability_audit/natural_audit_raw.jsonl')
SEL_PATH = Path('results/candidate_lock_depth/selected_ids_b12.json')
OUT_DIR = Path('results/candidate_lock_depth')
DIRECTION_PATH = 'steering/directions/direction_search_v3_layer20.npz'
LAYER = 20
ALPHA_MAX = 64.0
MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'

# (edit_kind, policy_dir, rho_magnitude) -- None policy means baseline (no steering)
CONDITIONS = [
    ('original',           None, 0.0),
    ('original',           'a3', 0.20),
    ('original',           'a3', 0.50),
    ('replace_W',          None, 0.0),
    ('replace_W',          'a3', 0.20),
    ('replace_W',          'a3', 0.50),
    ('irrelevant_control', 'a3', 0.50),
]


class ScriptedFirstObsSearchTool:
    """Return a pre-set scripted observation on the FIRST __call__; defer to
    the wrapped real SearchTool for any subsequent calls. Reset per sample."""

    def __init__(self, real_tool):
        self.real = real_tool
        self.scripted = None
        self._n_calls = 0
        self.first_call_query = None

    def reset(self, scripted_first_obs):
        self.scripted = scripted_first_obs
        self._n_calls = 0
        self.first_call_query = None

    def __call__(self, query):
        self._n_calls += 1
        if self._n_calls == 1 and self.scripted is not None:
            self.first_call_query = query
            return self.scripted
        return self.real(query)


def build_variants(audit_rec):
    """Return dict[edit_kind] -> (edited_obs, edit_meta)."""
    obs = audit_rec.get('observation_full') or ''
    W = audit_rec.get('emitted_answer_W') or ''
    qtype = audit_rec.get('question_type') or 'other'
    placeholder = _PLACEHOLDER_BY_QTYPE.get(qtype, '[unspecified]')
    obs_r, n_r = edit_replace_W(obs, W, qtype)
    obs_d, n_d = edit_remove_W(obs, W)
    obs_c, n_c = edit_irrelevant_control(obs, W, audit_rec.get('question', ''),
                                          max(1, n_d) if n_d else 1)
    return {
        'original':           (obs,   {'n_edits': 0, 'placeholder': None}),
        'replace_W':          (obs_r, {'n_edits': n_r, 'placeholder': placeholder}),
        'irrelevant_control': (obs_c, {'n_edits': n_c, 'placeholder': None}),
        # remove_W kept for optional / inspection only
        'remove_W':           (obs_d, {'n_edits': n_d, 'placeholder': None}),
    }


def _stats(records, audit, bucket_of, placeholder_of):
    n = len(records)
    if n == 0: return {}
    n_search = n_2nd = n_commit_W = n_contains_gold = n_contains_R = 0
    n_em = n_pf = n_fmt = 0
    mb_list, ma_list = [], []
    for r in records:
        sid = r['sample_id']
        steps = r.get('steps') or []
        actions = [s.get('action') for s in steps]
        if any(a == 'search' for a in actions): n_search += 1
        if sum(1 for a in actions if a == 'search') >= 2: n_2nd += 1
        if any(s.get('parse_failure_reason') for s in steps): n_pf += 1
        ans = (r.get('final_answer') or '').strip().lower()
        au = audit.get(sid) or {}
        W = (au.get('emitted_answer_W') or '').strip().lower()
        gold = (au.get('gold_answer') or '').strip().lower()
        if W and ans and (ans == W or W in ans or ans in W):
            n_commit_W += 1
        if gold and ans and (ans == gold or gold in ans or ans in gold):
            n_contains_gold += 1
        R = (placeholder_of.get(sid) or '').strip().lower()
        if R and ans and R in ans:
            n_contains_R += 1
        if r.get('em_correct'): n_em += 1
        for s in steps:
            txt = s.get('raw_model_text') or ''
            if txt and s.get('action') is None and s.get('final_answer') is None \
                    and 'Action:' not in txt and 'Final Answer:' not in txt:
                n_fmt += 1; break
        if len(steps) > 1:
            stg = steps[1].get('steering') or {}
            mb = stg.get('m_before') if stg else None
            if mb is None: mb = steps[1].get('margin_before')
            ma = stg.get('m_after') if stg else None
            if mb is not None: mb_list.append(float(mb))
            if ma is not None: ma_list.append(float(ma))
    return {
        'n': n, 'search_rate': n_search/n, 'second_search_rate': n_2nd/n,
        'commit_W_rate': n_commit_W/n, 'contains_gold_rate': n_contains_gold/n,
        'contains_R_rate': n_contains_R/n, 'em_correct_rate': n_em/n,
        'parse_failure_rate': n_pf/n, 'format_drift_rate': n_fmt/n,
        'mean_m_before': float(np.mean(mb_list)) if mb_list else None,
        'mean_m_after':  float(np.mean(ma_list)) if ma_list else None,
        'mean_delta_margin': (float(np.mean(np.array(ma_list) - np.array(mb_list)))
                              if mb_list and ma_list and len(mb_list)==len(ma_list) else None),
    }



def aggregate(records, audit, bucket_of, placeholder_of):
    return {
        'overall': _stats(records, audit, bucket_of, placeholder_of),
        'b1': _stats([r for r in records if bucket_of.get(r['sample_id']) == 1],
                     audit, bucket_of, placeholder_of),
        'b2': _stats([r for r in records if bucket_of.get(r['sample_id']) == 2],
                     audit, bucket_of, placeholder_of),
    }


def evaluate_success_criteria(summaries):
    """Print A/B/C interpretation per the prompt's success criteria."""
    g = lambda tag, k: summaries[tag]['overall'][k]
    print('\n=== Success criteria ===')
    # A: replace_W + A3 rho=0.20 vs original + A3 rho=0.20  ->  search up or margin up
    a_lhs_2sr = g('replace_W_a3_rho0.20', 'second_search_rate')
    a_rhs_2sr = g('original_a3_rho0.20',  'second_search_rate')
    a_lhs_dm  = g('replace_W_a3_rho0.20', 'mean_delta_margin') or 0.0
    a_rhs_dm  = g('original_a3_rho0.20',  'mean_delta_margin') or 0.0
    A_pass = (a_lhs_2sr - a_rhs_2sr >= 0.10) or (a_lhs_dm - a_rhs_dm >= 1.0)
    print(f'A. replace_W+A3@0.20 vs original+A3@0.20: '
          f'2sr {a_rhs_2sr:.1%} -> {a_lhs_2sr:.1%} (Δ={a_lhs_2sr-a_rhs_2sr:+.1%}); '
          f'Δmargin {a_rhs_dm:+.2f} -> {a_lhs_dm:+.2f}  =>  pass={A_pass}')

    # B: replace_W + A3 rho=0.50 vs original + A3 rho=0.50  ->  contains_W down OR gold/2sr up
    b_lhs_W = g('replace_W_a3_rho0.50', 'commit_W_rate')
    b_rhs_W = g('original_a3_rho0.50',  'commit_W_rate')
    b_lhs_g = g('replace_W_a3_rho0.50', 'contains_gold_rate')
    b_rhs_g = g('original_a3_rho0.50',  'contains_gold_rate')
    b_lhs_2 = g('replace_W_a3_rho0.50', 'second_search_rate')
    b_rhs_2 = g('original_a3_rho0.50',  'second_search_rate')
    B_pass = ((b_rhs_W - b_lhs_W) >= 0.10) or ((b_lhs_g - b_rhs_g) >= 0.05) \
             or ((b_lhs_2 - b_rhs_2) >= 0.05)
    print(f'B. replace_W+A3@0.50 vs original+A3@0.50: '
          f'commit_W {b_rhs_W:.1%} -> {b_lhs_W:.1%} (Δ={b_lhs_W-b_rhs_W:+.1%}); '
          f'gold {b_rhs_g:.1%} -> {b_lhs_g:.1%}; 2sr {b_rhs_2:.1%} -> {b_lhs_2:.1%}  =>  pass={B_pass}')

    # C: irrelevant_control + A3 rho=0.50 should NOT match replace_W + A3 rho=0.50
    c_ctl_W = g('irrelevant_control_a3_rho0.50', 'commit_W_rate')
    c_ctl_g = g('irrelevant_control_a3_rho0.50', 'contains_gold_rate')
    C_specificity = ((b_rhs_W - b_lhs_W) - (b_rhs_W - c_ctl_W)) >= 0.05 \
                    or ((b_lhs_g - b_rhs_g) - (c_ctl_g - b_rhs_g)) >= 0.03
    print(f'C. irrelevant_control+A3@0.50: commit_W={c_ctl_W:.1%} gold={c_ctl_g:.1%}; '
          f'replace_W effect bigger than control? specificity={C_specificity}')

    return {'A': A_pass, 'B': B_pass, 'C': C_specificity}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = {r['sample_id']: r for r in (json.loads(l) for l in open(AUDIT_PATH))}
    sel = json.load(open(SEL_PATH))
    sids_b1, sids_b2 = sel['b1'], sel['b2']
    bucket_of = {sid: 1 for sid in sids_b1}
    bucket_of.update({sid: 2 for sid in sids_b2})
    target_ids = sorted(set(sids_b1) | set(sids_b2))
    print(f'Step 3 sample set: B1={len(sids_b1)} B2={len(sids_b2)} (total={len(target_ids)})')

    print('Pre-building edit variants ...')
    variants = {sid: build_variants(audit[sid]) for sid in target_ids}
    placeholder_of = {sid: variants[sid]['replace_W'][1].get('placeholder') for sid in target_ids}
    n_repl_failed = sum(1 for sid in target_ids if variants[sid]['replace_W'][1]['n_edits'] == 0)
    print(f'  replace_W failed (no W alias matched in obs): {n_repl_failed}/{len(target_ids)}')
    edit_summary = {
        sid: {k: v[1] for k, v in variants[sid].items()} for sid in target_ids
    }
    json.dump(edit_summary, open(OUT_DIR/'anchor_interaction_edit_meta.json', 'w'), indent=2)

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

    real_search = SearchTool(corpus_path='data/hotpotqa/corpus.jsonl')
    scripted_tool = ScriptedFirstObsSearchTool(real_search)
    config = AgentConfig(max_steps=5, max_tokens_per_step=256, temperature=0.0,
                         layer=LAYER, tools=['search'], score_mode='exact')
    direction, _ = load_direction(DIRECTION_PATH, normalize_rms=1.0)
    direction_rms = float(np.sqrt(np.mean(direction ** 2)))
    agent = ReActAgent(model=model, tokenizer=tok, tools={'search': scripted_tool},
                       config=config, direction=direction, direction_rms=direction_rms)

    raw_path = OUT_DIR / 'anchor_interaction.jsonl'
    raw_f = open(raw_path, 'w')
    summaries = {}
    t0 = time.time()
    try:
        for edit_kind, dir_name, rho_mag in CONDITIONS:
            if dir_name is None:
                cond_tag = f'{edit_kind}_none'
                policy_factory = lambda: FreeGenBaselinePolicy()
            else:
                cond_tag = f'{edit_kind}_{dir_name}_rho{rho_mag:.2f}'
                rho = -float(rho_mag)
                policy_factory = lambda r=rho: FixedRhoSteerPolicy(
                    rho=r, steer_step=1, alpha_max=ALPHA_MAX, decision_only=True)
            print(f'\n=== {cond_tag} (elapsed {(time.time()-t0)/60:.1f} min) ===')
            cond_results = []
            for s in tqdm(samples, desc=cond_tag):
                edited_obs, edit_meta = variants[s.id][edit_kind]
                scripted_tool.reset(scripted_first_obs=edited_obs)
                policy = policy_factory()
                r = run_episode(agent, s, policy, 'exact')
                r['_edit_kind'] = edit_kind
                r['_edit_meta'] = edit_meta
                r['_first_query_to_scripted_tool'] = scripted_tool.first_call_query
                cond_results.append(r)
                raw_f.write(json.dumps({
                    'condition': cond_tag, 'edit_kind': edit_kind,
                    'direction': dir_name, 'rho': (None if dir_name is None else -rho_mag),
                    'alpha_max': ALPHA_MAX, 'sample_id': s.id,
                    'bucket': bucket_of[s.id],
                    'edit_meta': edit_meta, 'episode': r,
                }, ensure_ascii=False) + '\n')
                raw_f.flush()
            summaries[cond_tag] = aggregate(cond_results, audit, bucket_of, placeholder_of)
            o = summaries[cond_tag]['overall']
            print(f'  overall: 2sr={o["second_search_rate"]:.1%} commitW={o["commit_W_rate"]:.1%} '
                  f'gold={o["contains_gold_rate"]:.1%} R={o["contains_R_rate"]:.1%} '
                  f'em={o["em_correct_rate"]:.1%} pf={o["parse_failure_rate"]:.1%}')
    finally:
        raw_f.close()

    summary_obj = {
        'timestamp': datetime.now().isoformat(),
        'config': {'layer': LAYER, 'alpha_max': ALPHA_MAX,
                   'n_b1': len(sids_b1), 'n_b2': len(sids_b2),
                   'direction': DIRECTION_PATH, 'conditions': CONDITIONS,
                   'sign_convention': 'negative rho pushes toward search'},
        'summaries': summaries,
    }
    json.dump(summary_obj, open(OUT_DIR/'anchor_interaction_summary.json', 'w'),
              indent=2, default=str)
    evaluate_success_criteria(summaries)
    print(f'\nWrote {raw_path} and anchor_interaction_summary.json')


if __name__ == '__main__':
    main()
