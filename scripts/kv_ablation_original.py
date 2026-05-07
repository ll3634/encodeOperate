#!/usr/bin/env python3
"""Experiment ③: KV2 ablation on `original` (un-edited) observation.

Tests whether ablating KV group 2 at attn_L18 on the natural (W-bearing)
observation reduces commitment to W -- i.e. whether the L18 KV2 routing is
the circuit-level cause of the behavioral commitment that `replace_W`
reverses by removing W from the input.

Same N=100 sample set and same scripted first observation as Step 3
`original` condition. Conditions:

  baseline             no intervention                       (= original_none)
  kv2_ablate           L18 KV2 alpha=0.0 at decision point
  kv0_ablate           L18 KV0 alpha=0.0 (specificity control)
  kv2_amplify          L18 KV2 alpha=2.0 (positive control)
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '.')
from agent.react_loop import ReActAgent, AgentConfig
from agent.policies_verify import FreeGenBaselinePolicy, KVGroupScalingPolicy
from datasets.hotpotqa import HotpotQADataset
from tools.search_tool import SearchTool
from steering.directions import load_direction
from scripts.run_verify_critical_pipeline import run_episode
from scripts.anchor_interaction import (
    ScriptedFirstObsSearchTool, build_variants, aggregate,
    AUDIT_PATH, SEL_PATH, MODEL_NAME,
)

OUT_DIR = Path('results/kv_ablation_original')
DIRECTION_PATH = 'steering/directions/direction_search_v3_layer20.npz'
LAYER = 20
ALPHA_MAX = 64.0

# (cond_tag, layer, kv_group, alpha)  None means baseline (no intervention)
CONDITIONS = [
    ('baseline',    None, None, None),
    ('kv2_ablate',  18,   2,    0.0),
    ('kv0_ablate',  18,   0,    0.0),
    ('kv2_amplify', 18,   2,    2.0),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = {r['sample_id']: r for r in (json.loads(l) for l in open(AUDIT_PATH))}
    sel = json.load(open(SEL_PATH))
    sids_b1, sids_b2 = sel['b1'], sel['b2']
    bucket_of = {sid: 1 for sid in sids_b1}
    bucket_of.update({sid: 2 for sid in sids_b2})
    target_ids = sorted(set(sids_b1) | set(sids_b2))
    print(f'sample set: B1={len(sids_b1)} B2={len(sids_b2)} (total={len(target_ids)})')

    print('Pre-building variants (for original observation only) ...')
    variants = {sid: build_variants(audit[sid]) for sid in target_ids}
    placeholder_of = {sid: variants[sid]['replace_W'][1].get('placeholder') for sid in target_ids}

    print('Loading model ...')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    model.eval()

    print('Loading dataset ...')
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

    raw_path = OUT_DIR / 'kv_ablation_original.jsonl'
    raw_f = open(raw_path, 'w')
    summaries = {}
    t0 = time.time()
    try:
        for cond_tag, layer, kv_group, alpha in CONDITIONS:
            if layer is None:
                policy_factory = lambda: FreeGenBaselinePolicy()
            else:
                policy_factory = lambda L=layer, G=kv_group, A=alpha: KVGroupScalingPolicy(
                    layer=L, kv_group=G, alpha=A)
            print(f'\n=== {cond_tag} (elapsed {(time.time()-t0)/60:.1f} min) ===')
            cond_results = []
            for s in tqdm(samples, desc=cond_tag):
                edited_obs, edit_meta = variants[s.id]['original']
                scripted_tool.reset(scripted_first_obs=edited_obs)
                policy = policy_factory()
                r = run_episode(agent, s, policy, 'exact')
                r['_edit_kind'] = 'original'
                r['_first_query_to_scripted_tool'] = scripted_tool.first_call_query
                cond_results.append(r)
                raw_f.write(json.dumps({
                    'condition': cond_tag, 'layer': layer, 'kv_group': kv_group,
                    'alpha': alpha, 'sample_id': s.id,
                    'bucket': bucket_of[s.id], 'episode': r,
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
                   'conditions': [list(c) for c in CONDITIONS],
                   'sample_set': 'Step 3 B1+B2 (selected_ids_b12.json)',
                   'first_obs': 'original (un-edited audit observation)'},
        'summaries': summaries,
    }
    json.dump(summary_obj, open(OUT_DIR/'kv_ablation_original_summary.json', 'w'),
              indent=2, default=str)
    print(f'\nWrote {raw_path} and kv_ablation_original_summary.json')


if __name__ == '__main__':
    main()
