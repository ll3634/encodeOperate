"""Recompute normalized projection cells/deltas from step1_*.jsonl rows."""
import json, math, sys, argparse
from collections import defaultdict

def load_rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def per_cond(rows):
    bc = defaultdict(list)
    for r in rows:
        nh = r['norm_h_act']
        if nh and nh > 0:
            bc[r['condition']].append((r['sample_id'], r['proj_action_at_Lact']/nh))
    return bc

def cell(pairs):
    vs = [v for _, v in pairs]
    n = len(vs)
    if n == 0: return {'n': 0}
    mn = sum(vs)/n
    var = sum((v-mn)**2 for v in vs)/(n-1) if n > 1 else 0.0
    return {'n': n, 'mean': mn, 'std': math.sqrt(var)}

def paired(a, b):
    am = dict(a); bm = dict(b)
    ks = sorted(set(am) & set(bm))
    diffs = [bm[k] - am[k] for k in ks]
    n = len(diffs)
    if n == 0: return None
    mn = sum(diffs)/n
    var = sum((d-mn)**2 for d in diffs)/(n-1) if n > 1 else 0.0
    sd = math.sqrt(var)
    cd = mn/sd if sd > 0 else float('nan')
    return {'n_pairs': n, 'mean_delta': mn, 'paired_d': cd}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--c2-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--L-act', type=int, required=True)
    ap.add_argument('--model', default='unknown')
    args = ap.parse_args()

    results = {}
    for ds in ('hotpotqa', 'musique'):
        rows = load_rows(f'{args.c2_dir}/step1_{ds}.jsonl')
        bc = per_cond(rows)
        cells = {c: cell(bc[c]) for c in ('N0', 'T0', 'S0')}
        deltas = {
            'T0_minus_N0': paired(bc['N0'], bc['T0']),
            'S0_minus_N0': paired(bc['N0'], bc['S0']),
            'S0_minus_T0': paired(bc['T0'], bc['S0']),
        }
        results[ds] = {'cells': cells, 'deltas': deltas}
        print(f'\n=== {ds} (L{args.L_act}) ===')
        for c, s in cells.items():
            print(f"  {c}: n={s['n']}  mean={s['mean']:+.5f}  std={s['std']:.5f}")
        for k, v in deltas.items():
            print(f"  {k}: Δ_norm={v['mean_delta']:+.5f}  paired_d={v['paired_d']:+.3f}")
    with open(args.out, 'w') as f:
        json.dump({'model': args.model, 'L_act': args.L_act, 'datasets': results}, f, indent=2)
    print(f'\n[save] {args.out}')

if __name__ == '__main__':
    main()
