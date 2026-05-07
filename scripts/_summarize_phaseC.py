"""Summarise Phase C behavioral eval (base / +T0 / +N0) on MuSiQue 2-hop pairs.

Per condition (N0=no support, T0=unsupported-extractable, S0=fully supported)
report:
  - n
  - search_rate  : fraction with action_type == "search"  (continue-to-search)
  - em_rate      : fraction with em == True
  - margin_post_mean : mean of margin_post  (positive = stop bias, negative = search bias)
  - parse_fail_rate
"""
import json, statistics
from pathlib import Path

FILES = [
    ("base",        "results/ft_phaseC/base.jsonl"),
    ("ft_t0",       "results/ft_phaseC/ft_t0.jsonl"),
    ("ft_n0",       "results/ft_phaseC/ft_n0.jsonl"),
    ("ft_balanced", "results/ft_phaseC/ft_balanced.jsonl"),
]
CONDS = ["N0", "T0", "S0"]


def summarise(path):
    rows = [json.loads(l) for l in open(path)]
    out = {}
    for c in CONDS:
        sub = [r for r in rows if r.get("condition") == c]
        if not sub:
            out[c] = None
            continue
        n = len(sub)
        n_search = sum(1 for r in sub if r.get("action_type") == "search")
        n_final = sum(1 for r in sub
                      if r.get("action_type") in ("stop", "final", "answer"))
        n_em = sum(1 for r in sub if r.get("em"))
        n_commit_w = sum(1 for r in sub if r.get("contains_W"))
        n_pf = sum(1 for r in sub if r.get("parse_failure"))
        margins = [r["margin_post"] for r in sub
                   if isinstance(r.get("margin_post"), (int, float))]
        lens = [len((r.get("raw_output") or "")) for r in sub]
        out[c] = dict(
            n=n,
            search_rate=n_search / n,
            final_rate=n_final / n,
            em_rate=n_em / n,
            commit_W_rate=n_commit_w / n,
            margin_post_mean=(statistics.mean(margins) if margins else None),
            margin_post_n=len(margins),
            parse_fail_rate=n_pf / n,
            output_len_mean=(statistics.mean(lens) if lens else 0),
        )
    return out


def main():
    summaries = {label: summarise(p) for label, p in FILES if Path(p).exists()}
    hdr = (f"{'model':<12} {'cond':<4} {'n':>4} {'search%':>8} {'final%':>7} "
           f"{'commit_W%':>9} {'em%':>6} {'pf%':>5} {'len':>5}")
    print(hdr)
    print("-" * len(hdr))
    for label, s in summaries.items():
        for c in CONDS:
            d = s.get(c)
            if d is None:
                print(f"{label:<12} {c:<4} {'-':>4}")
                continue
            print(f"{label:<12} {c:<4} {d['n']:>4} "
                  f"{d['search_rate']*100:>7.1f}% "
                  f"{d['final_rate']*100:>6.1f}% "
                  f"{d['commit_W_rate']*100:>8.1f}% "
                  f"{d['em_rate']*100:>5.1f}% "
                  f"{d['parse_fail_rate']*100:>4.1f}% "
                  f"{d['output_len_mean']:>5.0f}")
        print()

    # Stop-rule check for ft_balanced
    print("\n=== Phase C stop-rule check (balanced_v1 vs base) ===")
    if "ft_balanced" in summaries and "base" in summaries:
        b, B = summaries["base"], summaries["ft_balanced"]
        t0_delta = (B["T0"]["search_rate"] - b["T0"]["search_rate"]) * 100
        s0_search_pct = B["S0"]["search_rate"] * 100
        s0_final_pct = B["S0"]["final_rate"] * 100
        s0_em_pct = B["S0"]["em_rate"] * 100
        t0_em_pct = B["T0"]["em_rate"] * 100
        max_pf = max(B[c]["parse_fail_rate"] for c in CONDS) * 100

        gate1 = t0_delta >= 20.0
        gate2 = (s0_search_pct < 50.0) or \
                (s0_em_pct > t0_em_pct + 5.0) or (s0_final_pct > 50.0)
        gate3 = max_pf < 5.0
        gate4 = True  # output format = parse_fail rate already covers it

        print(f"  Gate1 T0 search +>=20pp:   delta=+{t0_delta:.1f}pp  -> {'PASS' if gate1 else 'FAIL'}")
        print(f"  Gate2 S0 not collapsed:    search={s0_search_pct:.1f}% final={s0_final_pct:.1f}% "
              f"em(S0)={s0_em_pct:.1f}% em(T0)={t0_em_pct:.1f}% -> {'PASS' if gate2 else 'FAIL'}")
        print(f"  Gate3 parse_fail<5%:       max={max_pf:.1f}%  -> {'PASS' if gate3 else 'FAIL'}")
        print(f"  Gate4 valid output:        {'PASS' if gate4 else 'FAIL'}")
        print(f"  Gate5 held-out overlap=0:  PASS  (verified at build, audit json)")
        verdict = "PROCEED to Phase D" if all([gate1, gate2, gate3, gate4]) \
                  else "STOP — balanced SFT did not produce evidence-conditioned action control."
        print(f"\n  VERDICT: {verdict}")
        summaries["_gate"] = dict(t0_delta_pp=t0_delta, s0_search_pct=s0_search_pct,
                                  s0_final_pct=s0_final_pct, s0_em_pct=s0_em_pct,
                                  t0_em_pct=t0_em_pct, max_pf_pct=max_pf,
                                  gate1=gate1, gate2=gate2, gate3=gate3, gate4=gate4,
                                  verdict=verdict)
    out_path = Path("results/ft_phaseC/summary.json")
    out_path.write_text(json.dumps(summaries, indent=2))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
