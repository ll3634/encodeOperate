#!/usr/bin/env python3
"""Cross-task transfer figure: margin diagnostic predicts steering success.

Three panels:
  (A) p0 logit margin distribution on T0 — codesearch vs SQL.
  (B) T0 commit_W vs |rho| on both surfaces (overlay + steering budget band).
  (C) Per-template SQL ρ=-0.60 rescue count vs mean p0 margin (predicted vs observed).
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("tmc/scripts/e2e_agent/results")

CS_BASE = ROOT / "nonqa_react_codesearch/20260501_051848_react_codesearch_n60_main_v1"
CS_SWEEP = {
    0.20: ROOT / "nonqa_react_codesearch/20260501_053757_react_codesearch_n60_transfer_steer_L20_pos0p200",
    0.0:  CS_BASE,
    -0.10: ROOT / "nonqa_react_codesearch/20260501_053616_react_codesearch_n60_transfer_steer_L20_neg0p100",
    -0.20: ROOT / "nonqa_react_codesearch/20260501_053436_react_codesearch_n60_transfer_steer_L20_neg0p200",
    -0.30: ROOT / "nonqa_react_codesearch/20260501_053256_react_codesearch_n60_transfer_steer_L20_neg0p300",
    -0.60: ROOT / "nonqa_react_codesearch/20260501_054045_react_codesearch_n60_transfer_steer_L20_neg0p600",
    -1.00: ROOT / "nonqa_react_codesearch/20260501_054225_react_codesearch_n60_transfer_steer_L20_neg1p000",
    -2.00: ROOT / "nonqa_react_codesearch/20260501_054403_react_codesearch_n60_transfer_steer_L20_neg2p000"}

SQL_BASE = ROOT / "nonqa_react_sql/20260501_062532_react_sql_n60_baseline"
SQL_SWEEP = {
    0.0: SQL_BASE,
    -0.20: ROOT / "nonqa_react_sql/20260501_063537_react_sql_n60_transfer_steer_L20_neg0p200",
    -0.30: ROOT / "nonqa_react_sql/20260501_064129_react_sql_n60_transfer_steer_L20_neg0p300",
    -0.60: ROOT / "nonqa_react_sql/20260501_064730_react_sql_n60_transfer_steer_L20_neg0p600"}

CS_MARGIN = ROOT / "nonqa_react_codesearch/margin_diag_T0/per_item.json"
SQL_MARGIN = ROOT / "nonqa_react_sql/margin_diag_T0/per_item.json"


def commit_rates(d: Path):
    rows = [json.loads(l) for l in (d / "parsed_outputs.jsonl").open()]
    by = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    out = {}
    for c, rs in by.items():
        n = len(rs)
        out[c] = {
            "commit_W": sum(r["commit_W"] for r in rs) / n,
            "commit_W_se": np.sqrt(sum(r["commit_W"] for r in rs) / n
                                    * (1 - sum(r["commit_W"] for r in rs) / n) / n),
            "commit_correct": sum(r.get("commit_correct", 0) for r in rs) / n,
            "n": n}
    return out


def main():
    cs_marg = [r["margin"] for r in json.load(CS_MARGIN.open())]
    sql_marg_per_item = json.load(SQL_MARGIN.open())
    sql_marg = [r["margin"] for r in sql_marg_per_item]
    sql_marg_by_item = {r["item_id"]: r["margin"] for r in sql_marg_per_item}

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # === Panel A: margin distributions =====================================
    ax = axes[0]
    bins = np.arange(-18, 5, 1)
    ax.hist(cs_marg, bins=bins, alpha=0.55, color="#c44e52",
            edgecolor="white", label=f"code-search  μ={np.mean(cs_marg):+.2f}")
    ax.hist(sql_marg, bins=bins, alpha=0.55, color="#4c72b0",
            edgecolor="white", label=f"SQL  μ={np.mean(sql_marg):+.2f}")
    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.axvspan(-2.76, 2.76, color="grey", alpha=0.15,
               label=r"reachable @ $|\rho|=0.60$")
    ax.set_xlabel("p0 logit margin (Action − Final)")
    ax.set_ylabel("# items (of 60)")
    ax.set_title("(A) Pre-intervention margin: surface heterogeneity")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)

    # === Panel B: commit_W vs rho =========================================
    ax = axes[1]
    cs_rhos = sorted(CS_SWEEP)
    cs_y = [commit_rates(CS_SWEEP[r])["T0"]["commit_W"] for r in cs_rhos]
    cs_e = [commit_rates(CS_SWEEP[r])["T0"]["commit_W_se"] for r in cs_rhos]
    cs_s0 = [commit_rates(CS_SWEEP[r])["S0"]["commit_correct"] for r in cs_rhos]

    sql_rhos = sorted(SQL_SWEEP)
    sql_y = [commit_rates(SQL_SWEEP[r])["T0"]["commit_W"] for r in sql_rhos]
    sql_e = [commit_rates(SQL_SWEEP[r])["T0"]["commit_W_se"] for r in sql_rhos]
    sql_s0 = [commit_rates(SQL_SWEEP[r])["S0"]["commit_correct"] for r in sql_rhos]

    ax.errorbar(cs_rhos, cs_y, yerr=cs_e, fmt="o-", color="#c44e52",
                label="code-search T0 commit_W", capsize=3)
    ax.errorbar(sql_rhos, sql_y, yerr=sql_e, fmt="s-", color="#4c72b0",
                label="SQL T0 commit_W", capsize=3)
    ax.plot(cs_rhos, cs_s0, "o:", color="#c44e52", alpha=0.45,
            label="code-search S0 correct (control)")
    ax.plot(sql_rhos, sql_s0, "s:", color="#4c72b0", alpha=0.45,
            label="SQL S0 correct (control)")
    ax.set_xlabel(r"steering $\rho$  (negative = continue)")
    ax.set_ylabel("rate")
    ax.set_title("(B) QA A3 transfer sweep")
    ax.set_xlim(-2.1, 0.3)
    ax.set_ylim(-0.03, 1.03)
    ax.axhline(0, color="k", lw=0.4)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="center right")

    # === Panel C: per-template SQL @ rho=-0.60 ============================
    ax = axes[2]
    base_rows = [json.loads(l) for l in (SQL_BASE / "parsed_outputs.jsonl").open()]
    rho_rows = [json.loads(l) for l in (SQL_SWEEP[-0.60] / "parsed_outputs.jsonl").open()]
    raw_rows = [json.loads(l) for l in (SQL_BASE / "raw_generations.jsonl").open()]
    tmpl = {r["item_id"]: r.get("template_key") for r in raw_rows}
    base_by = {(r["item_id"], r["condition"]): r for r in base_rows}
    rho_by = {(r["item_id"], r["condition"]): r for r in rho_rows}

    by_tmpl = defaultdict(lambda: {"resc": 0, "n": 0, "marg": []})
    for (iid, c), r in base_by.items():
        if c != "T0":
            continue
        t = tmpl[iid]
        rr = rho_by[(iid, "T0")]["commit_W"]
        if r["commit_W"] == 1 and rr == 0:
            by_tmpl[t]["resc"] += 1
        by_tmpl[t]["n"] += 1
        by_tmpl[t]["marg"].append(sql_marg_by_item[iid])

    tmpls = sorted(by_tmpl, key=lambda k: np.mean(by_tmpl[k]["marg"]), reverse=True)
    xs = [np.mean(by_tmpl[t]["marg"]) for t in tmpls]
    ys = [by_tmpl[t]["resc"] for t in tmpls]
    ns = [by_tmpl[t]["n"] for t in tmpls]
    ax.scatter(xs, ys, s=[80] * len(xs), color="#4c72b0", edgecolor="black", zorder=3)
    for x, y, t, n in zip(xs, ys, tmpls, ns):
        ax.annotate(f"{t}\n({y}/{n})", (x, y), xytext=(6, 6),
                    textcoords="offset points", fontsize=8)
    ax.axvspan(-2.76, 2.76, color="grey", alpha=0.18,
               label=r"reachable @ $|\rho|=0.60$")
    ax.set_xlabel("template mean p0 margin")
    ax.set_ylabel(r"# items rescued at $\rho=-0.60$  (of 12)")
    ax.set_title("(C) SQL: rescue follows the margin budget")
    ax.set_xlim(-13, 4)
    ax.set_ylim(-0.5, 11.5)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle("Cross-task transfer of the QA A3 steering direction", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path("tmc/scripts/e2e_agent/results/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "cross_task_transfer.png", dpi=160)
    fig.savefig(out / "cross_task_transfer.pdf")
    print(f"[fig] wrote {out/'cross_task_transfer.png'}")


if __name__ == "__main__":
    main()
