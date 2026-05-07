#!/usr/bin/env python3
"""Step 1 of the candidate-lock-depth diagnostic.

Reads the existing baseline + A3-steered runs (N = 500, ρ = 0.20) and the
natural-failure audit, joins them by sample_id, classifies each baseline
wrong-stop into B1..B5, and reports per-bucket statistics on the
search-vs-final action margin at the post-observation decision token:

  - baseline margin        (m_before; threshold = 0; positive ⇒ prefer search)
  - A3 m_after             (post-steering margin)
  - delta = m_after - m_before
  - distance to threshold  = -m_before for stop-anchored cases
  - 2nd-search rate        baseline (always 0) and A3
  - rescue rate            (EM correct after steering)
  - rescued vs non-rescued margin distributions

H0 (A3 unrelated to candidate commitment) vs H1 (candidate commitment is a
deeper basin). Diagnostic outputs are written to
results/candidate_lock_depth/margin_by_bucket.json.
"""
from __future__ import annotations
import json, sys, os, statistics
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '.')
from scripts.audit_natural_failures import _contains_alias  # noqa: E402
from scripts.steering_by_bucket import assign_bucket  # noqa: E402

ROOT = Path('results/l20_rho020_n500')
AUDIT_PATH = Path('results/natural_extractability_audit/natural_audit_raw.jsonl')
BASE_PATH = ROOT / 'baseline_results.jsonl'
STEER_PATH = ROOT / 'v3_L20' / 'jes_tau0.20_mr0.20.jsonl'
OUT_DIR = Path('results/candidate_lock_depth')
OUT_PATH = OUT_DIR / 'margin_by_bucket.json'


def step1_margins(rec_steer: dict, rec_base: dict) -> tuple | None:
    """Return (m_before, m_after, delta, steer_action) at step 1, or None."""
    if len(rec_steer.get('steps') or []) < 2 or len(rec_base.get('steps') or []) < 2:
        return None
    steer_st1 = rec_steer['steps'][1]
    base_st1 = rec_base['steps'][1]
    s = steer_st1.get('steering') or {}
    m_before = s.get('m_before')
    m_after = s.get('m_after')
    if m_before is None or m_after is None:
        # fall back to baseline margin_before
        m_before = base_st1.get('margin_before')
        m_after = steer_st1.get('margin_before')
        if m_before is None or m_after is None:
            return None
    return float(m_before), float(m_after), float(m_after - m_before), steer_st1.get('action')


def quartiles(xs):
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    def q(p):
        i = max(0, min(n - 1, int(round(p * (n - 1)))))
        return xs[i]
    return {'min': xs[0], 'q1': q(0.25), 'median': xs[n // 2], 'mean': sum(xs) / n,
            'q3': q(0.75), 'max': xs[-1], 'n': n}


def main():
    audit = {r['sample_id']: r for r in (json.loads(l) for l in open(AUDIT_PATH))}
    base = {r['sample_id']: r for r in (json.loads(l) for l in open(BASE_PATH))}
    steer = {r['sample_id']: r for r in (json.loads(l) for l in open(STEER_PATH))}
    ids = sorted(set(audit) & set(base) & set(steer))

    per_bucket = defaultdict(lambda: {
        'm_before': [], 'm_after': [], 'delta': [],
        'rescued_m_before': [], 'rescued_delta': [],
        'nonresc_m_before': [], 'nonresc_delta': [],
        'flipped_m_before': [], 'flipped_delta': [],  # action flipped to search
        'noflip_m_before': [], 'noflip_delta': [],
        'n_total': 0, 'n_with_margin': 0,
        'n_steer_search': 0, 'n_rescued': 0,
    })

    for sid in ids:
        b = assign_bucket(audit[sid])
        if b < 0:
            continue
        per_bucket[b]['n_total'] += 1
        mres = step1_margins(steer[sid], base[sid])
        if mres is None:
            continue
        m_b, m_a, dlt, st_act = mres
        flipped = (st_act == 'search')
        rescued = bool(steer[sid].get('em_correct'))
        d = per_bucket[b]
        d['n_with_margin'] += 1
        d['m_before'].append(m_b); d['m_after'].append(m_a); d['delta'].append(dlt)
        if flipped:
            d['n_steer_search'] += 1
            d['flipped_m_before'].append(m_b); d['flipped_delta'].append(dlt)
        else:
            d['noflip_m_before'].append(m_b); d['noflip_delta'].append(dlt)
        if rescued:
            d['n_rescued'] += 1
            d['rescued_m_before'].append(m_b); d['rescued_delta'].append(dlt)
        else:
            d['nonresc_m_before'].append(m_b); d['nonresc_delta'].append(dlt)

    summary = {'per_bucket': {}, 'aggregate_12_vs_45': {}}
    for b in [1, 2, 3, 4, 5, 0]:
        if b not in per_bucket: continue
        d = per_bucket[b]
        summary['per_bucket'][b] = {
            'n_total': d['n_total'], 'n_with_margin': d['n_with_margin'],
            'n_steer_search': d['n_steer_search'], 'n_rescued': d['n_rescued'],
            'baseline_2nd_search_rate': 0.0,
            'a3_2nd_search_rate': d['n_steer_search'] / max(1, d['n_with_margin']),
            'rescue_rate': d['n_rescued'] / max(1, d['n_total']),
            'm_before': quartiles(d['m_before']),
            'm_after': quartiles(d['m_after']),
            'delta': quartiles(d['delta']),
            'rescued_m_before': quartiles(d['rescued_m_before']),
            'nonresc_m_before': quartiles(d['nonresc_m_before']),
            'flipped_m_before': quartiles(d['flipped_m_before']),
            'noflip_m_before':  quartiles(d['noflip_m_before']),
            'flipped_delta':    quartiles(d['flipped_delta']),
            'noflip_delta':     quartiles(d['noflip_delta']),
        }

    # Aggregate B1+B2 vs B4+B5
    def agg(bs, key):
        out = []
        for bb in bs:
            out.extend(per_bucket[bb][key])
        return out

    from scipy.stats import mannwhitneyu
    g12 = agg([1, 2], 'm_before'); g45 = agg([4, 5], 'm_before')
    d12 = agg([1, 2], 'delta');    d45 = agg([4, 5], 'delta')
    flip12 = sum(per_bucket[bb]['n_steer_search'] for bb in [1, 2])
    n12   = sum(per_bucket[bb]['n_with_margin']    for bb in [1, 2])
    flip45 = sum(per_bucket[bb]['n_steer_search'] for bb in [4, 5])
    n45   = sum(per_bucket[bb]['n_with_margin']    for bb in [4, 5])
    u_mb, p_mb = mannwhitneyu(g12, g45, alternative='less')   # 1+2 more negative?
    u_dl, p_dl = mannwhitneyu(d12, d45, alternative='two-sided')
    summary['aggregate_12_vs_45'] = {
        'baseline_margin_quartiles_12': quartiles(g12),
        'baseline_margin_quartiles_45': quartiles(g45),
        'mw_baseline_margin_12_lt_45': {'U': float(u_mb), 'p_one_sided': float(p_mb)},
        'delta_quartiles_12': quartiles(d12),
        'delta_quartiles_45': quartiles(d45),
        'mw_delta_two_sided': {'U': float(u_dl), 'p': float(p_dl)},
        'a3_2nd_search_rate_12': flip12 / max(1, n12),
        'a3_2nd_search_rate_45': flip45 / max(1, n45),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f'\nWrote {OUT_PATH}')


if __name__ == '__main__':
    main()
