#!/usr/bin/env python3
"""Steering-by-bucket analysis (Task 1).

Joins the natural-failure audit (per-sample W/gold/observation) with the A3 N=500
baseline and steered runs, classifies each baseline wrong-stop into one of five
behavioural buckets, and reports rescue / regression statistics per bucket.

Buckets (over baseline wrong-stops only):
  1: W in obs, gold not in obs                           (extractable family)
  2: W in obs AND gold in obs                            (gold available, model picked W)
  3: W ~ gold (substring overlap)                       (EM false negative)
  4: W not in obs, gold not in obs                       (retrieval failure)
  5: W not in obs, gold in obs                           (prior override)
  0: no W extracted / parse failure                      (other)
"""
from __future__ import annotations
import json, sys, os
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, '.')
from scripts.audit_natural_failures import _contains_alias

ROOT = Path('results/l20_rho020_n500')
AUDIT_PATH = Path('results/natural_extractability_audit/natural_audit_raw.jsonl')
BASE_PATH = ROOT / 'baseline_results.jsonl'
STEER_PATH = ROOT / 'v3_L20' / 'jes_tau0.20_mr0.20.jsonl'
OUT_PATH = Path('results/natural_extractability_audit/steering_by_bucket.json')


def assign_bucket(audit_rec: dict) -> int:
    cat = audit_rec.get('category')
    if cat != 'step1_stop_wrong':
        return -1  # not a baseline wrong-stop; ignore for rescue analysis
    W = (audit_rec.get('emitted_answer_W') or '').strip()
    gold = (audit_rec.get('gold_answer') or '').strip()
    obs = audit_rec.get('observation_full') or ''
    if not W:
        return 0
    Wn, Gn = W.lower(), gold.lower()
    if gold and (Wn == Gn or Wn in Gn or Gn in Wn):
        return 3  # EM false negative
    in_obs_W = bool(audit_rec.get('W_in_observation'))
    in_obs_G = bool(gold) and _contains_alias(obs, gold)
    if in_obs_W and not in_obs_G:
        return 1
    if in_obs_W and in_obs_G:
        return 2
    if not in_obs_W and not in_obs_G:
        return 4
    if not in_obs_W and in_obs_G:
        return 5
    return 0


def first_action(rec: dict) -> str | None:
    steps = rec.get('steps') or []
    if not steps:
        return None
    return steps[0].get('action')


def took_second_search(rec: dict) -> bool:
    steps = rec.get('steps') or []
    return any((s.get('action') == 'search') for s in steps[1:])


def main():
    audit = {r['sample_id']: r for r in (json.loads(l) for l in open(AUDIT_PATH))}
    base = {r['sample_id']: r for r in (json.loads(l) for l in open(BASE_PATH))}
    steer = {r['sample_id']: r for r in (json.loads(l) for l in open(STEER_PATH))}
    ids = sorted(set(audit) & set(base) & set(steer))
    assert len(ids) == 500, f'expected 500 joined samples, got {len(ids)}'

    bucket_of = {sid: assign_bucket(audit[sid]) for sid in ids}
    bucket_counts = Counter(bucket_of.values())

    # Per-bucket stats (only over baseline wrong-stops, i.e. bucket >= 0)
    rows = {}
    for b in [1, 2, 3, 4, 5, 0]:
        sids = [sid for sid in ids if bucket_of[sid] == b]
        n = len(sids)
        rescued = []     # baseline wrong, steered correct
        regressed = []   # baseline correct, steered wrong (n/a here since base wrong)
        steer_search = 0  # steered model issues 2nd search after decision
        steer_changed = 0  # steered final_answer differs from baseline
        steer_correct = 0
        for sid in sids:
            b_em = bool(base[sid].get('em_correct'))
            s_em = bool(steer[sid].get('em_correct'))
            assert not b_em, 'bucket samples should all be baseline wrong'
            if s_em:
                rescued.append(sid)
                steer_correct += 1
            if took_second_search(steer[sid]):
                steer_search += 1
            b_fa = (base[sid].get('final_answer') or '').strip().lower()
            s_fa = (steer[sid].get('final_answer') or '').strip().lower()
            if b_fa != s_fa:
                steer_changed += 1
        # rescued via search
        rescued_via_search = sum(1 for sid in rescued if took_second_search(steer[sid]))
        rows[b] = {
            'n': n,
            'rescued': len(rescued),
            'rescued_via_search': rescued_via_search,
            'rescue_rate': (len(rescued) / n) if n else 0.0,
            'steer_2nd_search': steer_search,
            'steer_2nd_search_rate': (steer_search / n) if n else 0.0,
            'final_answer_changed': steer_changed,
            'rescued_ids': rescued[:5],
        }

    # Aggregated 1+2 vs 4+5 contrast
    from scipy.stats import fisher_exact
    n12 = rows[1]['n'] + rows[2]['n']; r12 = rows[1]['rescued'] + rows[2]['rescued']
    n45 = rows[4]['n'] + rows[5]['n']; r45 = rows[4]['rescued'] + rows[5]['rescued']
    s12 = rows[1]['steer_2nd_search'] + rows[2]['steer_2nd_search']
    s45 = rows[4]['steer_2nd_search'] + rows[5]['steer_2nd_search']
    rescue_table = [[r12, n12 - r12], [r45, n45 - r45]]
    search_table = [[s12, n12 - s12], [s45, n45 - s45]]
    or_resc, p_resc = fisher_exact(rescue_table, alternative='greater')
    or_srch, p_srch = fisher_exact(search_table, alternative='greater')

    summary = {
        'bucket_counts': dict(bucket_counts),
        'rows': rows,
        'aggregate_12_vs_45': {
            'rescue': {'n12': n12, 'r12': r12, 'n45': n45, 'r45': r45,
                       'OR': or_resc, 'p_one_sided': p_resc, 'table': rescue_table},
            'search_increase': {'n12': n12, 's12': s12, 'n45': n45, 's45': s45,
                                'OR': or_srch, 'p_one_sided': p_srch, 'table': search_table},
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2, default=str))

    # Print human-readable table
    print(f'Total joined samples: {len(ids)}')
    print(f'Bucket counts (over all 500): {dict(bucket_counts)}')
    print()
    print(f'{"bkt":>3} {"n":>4} {"resc":>5} {"r/n":>7} {"viaS":>5} {"2ndS":>5} {"2S/n":>7} {"FAchg":>6}')
    for b in [1, 2, 3, 4, 5, 0]:
        r = rows[b]
        print(f'{b:>3} {r["n"]:>4} {r["rescued"]:>5} {r["rescue_rate"]*100:>6.1f}% {r["rescued_via_search"]:>5} '
              f'{r["steer_2nd_search"]:>5} {r["steer_2nd_search_rate"]*100:>6.1f}% {r["final_answer_changed"]:>6}')
    print()
    print(f'Aggregate bucket1+2 (in-context candidate commitment) vs bucket4+5 (retrieval/prior):')
    print(f'  Rescue: 1+2 = {r12}/{n12} ({r12/n12*100:.1f}%)  vs  4+5 = {r45}/{n45} ({r45/n45*100:.1f}%)  '
          f'Fisher OR={or_resc:.3g}, one-sided p={p_resc:.4g}')
    print(f'  Search-incr: 1+2 = {s12}/{n12} ({s12/n12*100:.1f}%)  vs  4+5 = {s45}/{n45} ({s45/n45*100:.1f}%)  '
          f'Fisher OR={or_srch:.3g}, one-sided p={p_srch:.4g}')
    print(f'\nWrote {OUT_PATH}')


if __name__ == '__main__':
    main()
