#!/usr/bin/env python3
"""Projection analysis at the decision point.

Re-uses existing Step 3 (anchor_interaction) and controlled_lgm prompts; runs
ONE forward pass per (sample, edit_kind) capturing the L20 last-token residual,
then projects onto:

  - action_dir   = direction_search_v3_layer20.npz['decision_direction_normalized']
  - evidence_dir = phase1_probe/probe_direction_l20.npz['decision_direction']

Outputs:
  results/projection_analysis/step3_projections.jsonl      (per-record)
  results/projection_analysis/lgm_projections.jsonl        (per-record)
  results/projection_analysis/summary.json
  results/projection_analysis/report.md

Zero new generation. ~500 forward passes total.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '.')
from agent.prompts import PromptBuilder

MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
LAYER = 20
ACTION_DIR_PATH = 'steering/directions/direction_search_v3_layer20.npz'
EVIDENCE_DIR_PATH = 'results/phase1_probe/probe_direction_l20.npz'

ANCHOR_JSONL = Path('results/candidate_lock_depth/anchor_interaction.jsonl')
AUDIT_JSONL = Path('results/natural_extractability_audit/natural_audit_raw.jsonl')
SEL_PATH = Path('results/candidate_lock_depth/selected_ids_b12.json')

LGM_PAIRS = Path('results/controlled_lgm/pairs.jsonl')
LGM_EVAL = Path('results/controlled_lgm/eval_results.jsonl')

OUT_DIR = Path('results/projection_analysis')


def load_directions():
    d_act = np.load(ACTION_DIR_PATH)
    action_dir = d_act['decision_direction_normalized'].astype(np.float32)
    action_dir = action_dir / np.linalg.norm(action_dir)

    d_ev = np.load(EVIDENCE_DIR_PATH)
    evidence_dir = d_ev['decision_direction'].astype(np.float32)
    evidence_dir = evidence_dir / np.linalg.norm(evidence_dir)

    cos = float(np.dot(action_dir, evidence_dir))
    print(f'[dir] cos(action, evidence) = {cos:+.4f}')
    return action_dir, evidence_dir


class L20Capture:
    """Forward hook on model.model.layers[LAYER] capturing last-token residual."""

    def __init__(self, model, layer_idx):
        self.model = model
        self.layer = layer_idx
        self._h_last = None
        self._handle = None

    def __enter__(self):
        layers = self.model.model.layers

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self._h_last = h[0, -1, :].detach().float().cpu().numpy()
        self._handle = layers[self.layer].register_forward_hook(hook)
        return self

    def __exit__(self, *args):
        if self._handle is not None:
            self._handle.remove()

    @property
    def h_last(self):
        return self._h_last


def build_prompt(tokenizer, builder, question, first_query, observation):
    history = [{
        'action': 'search',
        'action_input': first_query,
        'observation': observation,
    }]
    messages = builder.build_full_prompt(question, history)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    return prompt


def project_one(model, tokenizer, prompt, action_dir, evidence_dir, device):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with L20Capture(model, LAYER) as cap:
        with torch.no_grad():
            model(input_ids)
    h = cap.h_last
    return {
        'proj_action': float(np.dot(h, action_dir)),
        'proj_evidence': float(np.dot(h, evidence_dir)),
        'h_norm': float(np.linalg.norm(h)),
        'h_rms': float(np.sqrt(np.mean(h ** 2))),
    }


# -------------------------- Step 3 --------------------------

def collect_step3_unique_prompts():
    """Return dict[(sid, edit_kind)] -> dict(question, first_query, obs)."""
    audit = {r['sample_id']: r for r in (json.loads(l) for l in open(AUDIT_JSONL))}
    sel = json.load(open(SEL_PATH))
    target = sorted(set(sel['b1']) | set(sel['b2']))
    bucket = {sid: 1 for sid in sel['b1']}
    bucket.update({sid: 2 for sid in sel['b2']})

    out = {}
    for line in open(ANCHOR_JSONL):
        r = json.loads(line)
        sid = r['sample_id']
        if sid not in target:
            continue
        edit_kind = r['edit_kind']
        if edit_kind not in ('original', 'replace_W', 'irrelevant_control'):
            continue
        key = (sid, edit_kind)
        if key in out:
            continue
        ep = r['episode']
        steps = ep.get('steps') or []
        if not steps:
            continue
        s0 = steps[0]
        if s0.get('action') != 'search' or not s0.get('observation'):
            continue
        question = ep['question']
        first_query = ep.get('_first_query_to_scripted_tool') or s0.get('action_input')
        obs = s0['observation']
        out[key] = {
            'sample_id': sid,
            'edit_kind': edit_kind,
            'bucket': bucket[sid],
            'question': question,
            'first_query': first_query,
            'observation': obs,
            'gold_answer': audit.get(sid, {}).get('gold_answer'),
            'emitted_W': audit.get(sid, {}).get('emitted_answer_W'),
        }
    return out


def run_step3(model, tokenizer, builder, device, action_dir, evidence_dir):
    prompts = collect_step3_unique_prompts()
    print(f'[step3] unique prompts: {len(prompts)}')
    out_path = OUT_DIR / 'step3_projections.jsonl'
    records = []
    t0 = time.time()
    with open(out_path, 'w') as f:
        for i, (key, info) in enumerate(prompts.items()):
            prompt = build_prompt(tokenizer, builder, info['question'],
                                  info['first_query'], info['observation'])
            p = project_one(model, tokenizer, prompt, action_dir, evidence_dir, device)
            rec = {**info, **p}
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            records.append(rec)
            if (i + 1) % 25 == 0 or i + 1 == len(prompts):
                dt = time.time() - t0
                print(f'  [{i+1}/{len(prompts)}] {dt:.1f}s')
    print(f'[step3] wrote {out_path}')
    return records


# -------------------------- controlled_lgm --------------------------

def run_lgm(model, tokenizer, builder, device, action_dir, evidence_dir):
    pairs = [json.loads(l) for l in open(LGM_PAIRS)]
    # Match the eval prompt convention (run_local_answerability_eval-style):
    # query = f"about: {question[:80]}".
    print(f'[lgm] records: {len(pairs)}')
    out_path = OUT_DIR / 'lgm_projections.jsonl'
    records = []
    t0 = time.time()
    with open(out_path, 'w') as f:
        for i, p in enumerate(pairs):
            q = p['question']
            first_query = f'about: {q[:80]}'
            obs = p['obs']
            prompt = build_prompt(tokenizer, builder, q, first_query, obs)
            proj = project_one(model, tokenizer, prompt, action_dir, evidence_dir, device)
            rec = {
                'sample_id': p['sample_id'],
                'condition_id': p['condition_id'],
                'schema': p.get('schema'),
                'V': p.get('V'),
                'gold_answer': p.get('gold_answer'),
                'answer_present': p.get('answer_present'),
                'global_sufficiency_verified': p.get('global_sufficiency_verified'),
                'feat': p.get('feat'),
                **proj,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            records.append(rec)
            if (i + 1) % 50 == 0 or i + 1 == len(pairs):
                dt = time.time() - t0
                print(f'  [{i+1}/{len(pairs)}] {dt:.1f}s')
    print(f'[lgm] wrote {out_path}')
    return records


# -------------------------- analysis --------------------------

def paired_ttest(d):
    """Return (t, df, p) for one-sample t-test on differences `d` (paired)."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    if n < 2:
        return None, None, None
    m = float(np.mean(d))
    s = float(np.std(d, ddof=1))
    if s == 0:
        return None, n - 1, None
    t = m / (s / np.sqrt(n))
    # two-sided p via normal approx (n is moderate; report z-based p)
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return t, n - 1, p


def wilcoxon_signed(d):
    """Wilcoxon signed-rank test (manual, no scipy dep)."""
    d = np.asarray(d, dtype=np.float64)
    nz = d[d != 0]
    n = len(nz)
    if n < 2:
        return None, None
    abs_d = np.abs(nz)
    order = np.argsort(abs_d)
    ranks = np.empty(n)
    # average-rank for ties
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_d[order[j + 1]] == abs_d[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_pos = float(np.sum(ranks[nz > 0]))
    w_neg = float(np.sum(ranks[nz < 0]))
    w = min(w_pos, w_neg)
    # Normal approx
    mu = n * (n + 1) / 4.0
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return w, None
    z = (w - mu) / sigma
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return float(w), float(p)


def analyze_step3(records):
    by_kind = defaultdict(dict)
    for r in records:
        by_kind[r['edit_kind']][r['sample_id']] = r

    sids_common = sorted(set(by_kind['original']) & set(by_kind['replace_W']))
    print(f'[step3-analysis] paired (original, replace_W) n={len(sids_common)}')

    paired = {
        'd_action': [],          # original - replace_W on action_dir
        'd_evidence': [],        # original - replace_W on evidence_dir
        'orig_action': [], 'rW_action': [],
        'orig_evidence': [], 'rW_evidence': [],
    }
    for sid in sids_common:
        a = by_kind['original'][sid]
        b = by_kind['replace_W'][sid]
        paired['d_action'].append(a['proj_action'] - b['proj_action'])
        paired['d_evidence'].append(a['proj_evidence'] - b['proj_evidence'])
        paired['orig_action'].append(a['proj_action'])
        paired['rW_action'].append(b['proj_action'])
        paired['orig_evidence'].append(a['proj_evidence'])
        paired['rW_evidence'].append(b['proj_evidence'])

    t_a, df_a, p_a = paired_ttest(paired['d_action'])
    t_e, df_e, p_e = paired_ttest(paired['d_evidence'])
    w_a, wp_a = wilcoxon_signed(paired['d_action'])
    w_e, wp_e = wilcoxon_signed(paired['d_evidence'])

    summary = {
        'n_paired': len(sids_common),
        'mean_orig_action':   float(np.mean(paired['orig_action'])),
        'mean_rW_action':     float(np.mean(paired['rW_action'])),
        'mean_orig_evidence': float(np.mean(paired['orig_evidence'])),
        'mean_rW_evidence':   float(np.mean(paired['rW_evidence'])),
        'mean_d_action':      float(np.mean(paired['d_action'])),
        'mean_d_evidence':    float(np.mean(paired['d_evidence'])),
        'sd_d_action':        float(np.std(paired['d_action'], ddof=1)),
        'sd_d_evidence':      float(np.std(paired['d_evidence'], ddof=1)),
        'paired_ttest_action':   {'t': t_a, 'df': df_a, 'p': p_a},
        'paired_ttest_evidence': {'t': t_e, 'df': df_e, 'p': p_e},
        'wilcoxon_action':       {'w': w_a, 'p': wp_a},
        'wilcoxon_evidence':     {'w': w_e, 'p': wp_e},
    }

    # Specificity vs irrelevant_control
    if by_kind.get('irrelevant_control'):
        sids_ic = sorted(set(by_kind['original']) & set(by_kind['irrelevant_control']))
        d_ic_action = [by_kind['original'][s]['proj_action']
                       - by_kind['irrelevant_control'][s]['proj_action'] for s in sids_ic]
        d_ic_evidence = [by_kind['original'][s]['proj_evidence']
                         - by_kind['irrelevant_control'][s]['proj_evidence'] for s in sids_ic]
        t_ica, _, p_ica = paired_ttest(d_ic_action)
        t_ice, _, p_ice = paired_ttest(d_ic_evidence)
        summary['n_paired_ic'] = len(sids_ic)
        summary['mean_d_ic_action'] = float(np.mean(d_ic_action))
        summary['mean_d_ic_evidence'] = float(np.mean(d_ic_evidence))
        summary['paired_ttest_ic_action'] = {'t': t_ica, 'p': p_ica}
        summary['paired_ttest_ic_evidence'] = {'t': t_ice, 'p': p_ice}

    return summary


def analyze_lgm(records):
    by_cond = defaultdict(list)
    for r in records:
        by_cond[r['condition_id']].append(r)
    summary = {}
    for cond in ['B0', 'B1', 'C0', 'D0']:
        rs = by_cond.get(cond, [])
        if not rs:
            continue
        a = [r['proj_action'] for r in rs]
        e = [r['proj_evidence'] for r in rs]
        summary[cond] = {
            'n': len(rs),
            'mean_action': float(np.mean(a)),
            'sd_action': float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
            'mean_evidence': float(np.mean(e)),
            'sd_evidence': float(np.std(e, ddof=1)) if len(e) > 1 else 0.0,
        }

    # Paired contrasts B0 vs C0 and B0 vs D0 (sample_id matched)
    by_cond_sid = {c: {r['sample_id']: r for r in rs} for c, rs in by_cond.items()}
    contrasts = {}
    for ref, alt in [('B0', 'C0'), ('B0', 'D0'), ('C0', 'D0')]:
        sids = sorted(set(by_cond_sid.get(ref, {})) & set(by_cond_sid.get(alt, {})))
        d_a = [by_cond_sid[ref][s]['proj_action'] - by_cond_sid[alt][s]['proj_action'] for s in sids]
        d_e = [by_cond_sid[ref][s]['proj_evidence'] - by_cond_sid[alt][s]['proj_evidence'] for s in sids]
        t_a, _, p_a = paired_ttest(d_a)
        t_e, _, p_e = paired_ttest(d_e)
        contrasts[f'{ref}_vs_{alt}'] = {
            'n': len(sids),
            'mean_d_action': float(np.mean(d_a)) if d_a else None,
            'mean_d_evidence': float(np.mean(d_e)) if d_e else None,
            'paired_ttest_action':   {'t': t_a, 'p': p_a},
            'paired_ttest_evidence': {'t': t_e, 'p': p_e},
        }
    summary['contrasts'] = contrasts
    return summary


def write_report(step3_sum, lgm_sum, dir_meta):
    md = ['# Projection analysis at decision point (L20)',
          '',
          '> One forward pass per (sample, edit_kind). Capture L20 last-token residual; ',
          'project onto action_dir (`direction_search_v3_layer20`, normalized) and ',
          'evidence_dir (`phase1_probe/probe_direction_l20`, normalized).',
          '',
          f'cos(action_dir, evidence_dir) = **{dir_meta["cos"]:+.4f}**',
          '',
          '## Step 3 — anchor (original) vs replace_W',
          '',
          f'paired n = {step3_sum["n_paired"]}',
          '',
          '| direction | mean(original) | mean(replace_W) | mean Δ (orig − rW) | sd(Δ) | paired t (df) | p (z-approx) | Wilcoxon p |',
          '|---|---:|---:|---:|---:|---:|---:|---:|',
          f'| action_dir   | {step3_sum["mean_orig_action"]:+.3f} | {step3_sum["mean_rW_action"]:+.3f} | '
          f'**{step3_sum["mean_d_action"]:+.3f}** | {step3_sum["sd_d_action"]:.3f} | '
          f'{step3_sum["paired_ttest_action"]["t"]:+.2f} ({step3_sum["paired_ttest_action"]["df"]}) | '
          f'{step3_sum["paired_ttest_action"]["p"]:.2g} | '
          f'{step3_sum["wilcoxon_action"]["p"]:.2g} |',
          f'| evidence_dir | {step3_sum["mean_orig_evidence"]:+.3f} | {step3_sum["mean_rW_evidence"]:+.3f} | '
          f'**{step3_sum["mean_d_evidence"]:+.3f}** | {step3_sum["sd_d_evidence"]:.3f} | '
          f'{step3_sum["paired_ttest_evidence"]["t"]:+.2f} ({step3_sum["paired_ttest_evidence"]["df"]}) | '
          f'{step3_sum["paired_ttest_evidence"]["p"]:.2g} | '
          f'{step3_sum["wilcoxon_evidence"]["p"]:.2g} |']
    if 'mean_d_ic_action' in step3_sum:
        md += ['',
               '### Specificity control (original vs irrelevant_control)',
               '',
               f'paired n = {step3_sum["n_paired_ic"]}',
               '',
               '| direction | mean Δ (orig − ic) | paired t | p |',
               '|---|---:|---:|---:|',
               f'| action_dir   | {step3_sum["mean_d_ic_action"]:+.3f} | '
               f'{step3_sum["paired_ttest_ic_action"]["t"]:+.2f} | '
               f'{step3_sum["paired_ttest_ic_action"]["p"]:.2g} |',
               f'| evidence_dir | {step3_sum["mean_d_ic_evidence"]:+.3f} | '
               f'{step3_sum["paired_ttest_ic_evidence"]["t"]:+.2f} | '
               f'{step3_sum["paired_ttest_ic_evidence"]["p"]:.2g} |']
    md += ['',
           '## controlled_lgm — per condition',
           '',
           '| cond | n | mean(action) ± sd | mean(evidence) ± sd |',
           '|---|---:|---:|---:|']
    for c in ['B0', 'B1', 'C0', 'D0']:
        if c not in lgm_sum: continue
        s = lgm_sum[c]
        md.append(f'| {c} | {s["n"]} | {s["mean_action"]:+.3f} ± {s["sd_action"]:.3f} | '
                  f'{s["mean_evidence"]:+.3f} ± {s["sd_evidence"]:.3f} |')
    md += ['',
           '### Paired contrasts (matched sample_id)',
           '',
           '| contrast | n | Δ action | t (action) | p (action) | Δ evidence | t (evidence) | p (evidence) |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for k, v in lgm_sum.get('contrasts', {}).items():
        md.append(f'| {k} | {v["n"]} | {v["mean_d_action"]:+.3f} | '
                  f'{v["paired_ttest_action"]["t"]:+.2f} | {v["paired_ttest_action"]["p"]:.2g} | '
                  f'{v["mean_d_evidence"]:+.3f} | '
                  f'{v["paired_ttest_evidence"]["t"]:+.2f} | {v["paired_ttest_evidence"]["p"]:.2g} |')
    md += ['',
           '## Sign conventions',
           '',
           '- action_dir: extracted as `h_high_margin_mean − h_low_margin_mean`, where margin = log P(tool) − log P(finish). '
           'Higher projection → more "search/tool" preference, lower → more "finish/commit" preference.',
           '- evidence_dir: linear-probe coefficient for label 0 (no supporting doc) vs label 1 (≥1 doc). '
           'Sign: positive projection → "more sufficient" by the probe.',
           '']
    out_path = OUT_DIR / 'report.md'
    out_path.write_text('\n'.join(md))
    print(f'[report] wrote {out_path}')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    action_dir, evidence_dir = load_directions()
    cos_ae = float(np.dot(action_dir, evidence_dir))

    print(f'[model] loading {MODEL_NAME} (bfloat16)')
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    model.eval()
    device = next(model.parameters()).device
    builder = PromptBuilder()

    step3_records = run_step3(model, tok, builder, device, action_dir, evidence_dir)
    lgm_records = run_lgm(model, tok, builder, device, action_dir, evidence_dir)

    step3_sum = analyze_step3(step3_records)
    lgm_sum = analyze_lgm(lgm_records)

    out = {
        'config': {
            'model': MODEL_NAME, 'layer': LAYER,
            'action_dir': ACTION_DIR_PATH, 'evidence_dir': EVIDENCE_DIR_PATH,
            'cos_action_evidence': cos_ae,
        },
        'step3': step3_sum,
        'lgm': lgm_sum,
    }
    json.dump(out, open(OUT_DIR / 'summary.json', 'w'), indent=2)
    print('[summary] wrote', OUT_DIR / 'summary.json')

    write_report(step3_sum, lgm_sum, {'cos': cos_ae})

    print('\n=== Step 3 (original − replace_W, paired) ===')
    print(f"  Δ action_dir   = {step3_sum['mean_d_action']:+.3f}  "
          f"(t={step3_sum['paired_ttest_action']['t']:+.2f}, p={step3_sum['paired_ttest_action']['p']:.2g})")
    print(f"  Δ evidence_dir = {step3_sum['mean_d_evidence']:+.3f}  "
          f"(t={step3_sum['paired_ttest_evidence']['t']:+.2f}, p={step3_sum['paired_ttest_evidence']['p']:.2g})")
    print('\n=== controlled_lgm ===')
    for c in ['B0', 'B1', 'C0', 'D0']:
        if c not in lgm_sum: continue
        s = lgm_sum[c]
        print(f'  {c}: n={s["n"]}  action={s["mean_action"]:+.3f}±{s["sd_action"]:.3f}  '
              f'evidence={s["mean_evidence"]:+.3f}±{s["sd_evidence"]:.3f}')


if __name__ == '__main__':
    main()
