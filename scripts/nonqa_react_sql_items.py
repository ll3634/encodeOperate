#!/usr/bin/env python3
"""
Items for the multi-turn ReAct SQL data-analysis surface.

The agent answers a business analytics question via `execute_query`. The
prefilled scratchpad already executed one SQL whose Observation is a result
table. In T0 the prefilled SQL computes the WRONG metric (e.g., COUNT instead
of COUNT/COUNT) and the result table has a plausible rank-1 entity W; the
correct answer is a different entity that needs a different query. In S0 the
prefilled SQL is the CORRECT metric and the rank-1 entity is the true answer.
In N0 the result is empty (no W). In IC the result is aggregated by a
different (non-entity) dimension (e.g., monthly), so no W is present.

Five template families: rate_vs_count, total_vs_per_unit, revenue_vs_margin,
absolute_vs_growth, gross_vs_net.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple


# Each template defines: question, schema description, trap/correct SQL,
# the prefilled Thought, the column header for trap/correct/IC tables, the
# IC dimension values, the period label, and the entity pools (12 per
# template, so 5 * 12 = 60 unique items).
TEMPLATES: List[dict] = [
    {
        "key": "rate_vs_count",
        "period": "Q3 2025",
        "date_lo": "2025-07-01", "date_hi": "2025-09-30",
        "question": "Which product category had the highest return rate in {period}?",
        "schema": ("Use the `orders` and `returns_log` tables in the analytics "
                   "database. The `orders` table has columns: order_id, category, "
                   "order_date, quantity. The `returns_log` table has columns: "
                   "return_id, order_id, category, return_date, reason."),
        "dim_label": "category",
        "trap_metric_col": "total_returns",
        "correct_metric_col": "return_rate",
        "trap_unit": "int", "correct_unit": "pct",
        "trap_thought": "I need to find the product category with the highest return rate in {period}. Let me start by looking at returns by category.",
        "correct_thought": "I need to find the product category with the highest return rate in {period}. I'll join returns with orders and compute the rate.",
        "trap_sql": "SELECT category, COUNT(*) as total_returns FROM returns_log WHERE return_date BETWEEN '{date_lo}' AND '{date_hi}' GROUP BY category ORDER BY total_returns DESC",
        "correct_sql": "SELECT r.category, COUNT(r.return_id) * 100.0 / COUNT(DISTINCT o.order_id) as return_rate FROM returns_log r JOIN orders o ON r.category = o.category WHERE r.return_date BETWEEN '{date_lo}' AND '{date_hi}' GROUP BY r.category ORDER BY return_rate DESC",
        "ic_dim_label": "month",
        "ic_dim_values": ["2025-07", "2025-08", "2025-09"],
        "trap_value_range": (200, 1500),
        "correct_value_range": (3.0, 15.0),
        "ic_value_range": (700, 1100),
    },
    {
        "key": "total_vs_per_unit",
        "period": "March 2025",
        "date_lo": "2025-03-01", "date_hi": "2025-03-31",
        "question": "Which supplier had the highest shipping cost per order in {period}?",
        "schema": ("Use the `shipments` table in the logistics database. Columns: "
                   "shipment_id, supplier, order_id, ship_date, shipping_cost."),
        "dim_label": "supplier",
        "trap_metric_col": "total_shipping_cost",
        "correct_metric_col": "cost_per_order",
        "trap_unit": "usd", "correct_unit": "usd_decimal",
        "trap_thought": "I need to find the supplier with the highest shipping cost per order in {period}. Let me query the total shipping cost by supplier first.",
        "correct_thought": "I need to find the supplier with the highest shipping cost per order in {period}. I'll compute the per-order average.",
        "trap_sql": "SELECT supplier, SUM(shipping_cost) as total_shipping_cost FROM shipments WHERE ship_date BETWEEN '{date_lo}' AND '{date_hi}' GROUP BY supplier ORDER BY total_shipping_cost DESC",
        "correct_sql": "SELECT supplier, SUM(shipping_cost) / COUNT(DISTINCT order_id) as cost_per_order FROM shipments WHERE ship_date BETWEEN '{date_lo}' AND '{date_hi}' GROUP BY supplier ORDER BY cost_per_order DESC",
        "ic_dim_label": "week",
        "ic_dim_values": ["2025-03 W1", "2025-03 W2", "2025-03 W3", "2025-03 W4"],
        "trap_value_range": (8000, 45000),
        "correct_value_range": (12.0, 38.0),
        "ic_value_range": (9000, 13000),
    },
    {
        "key": "revenue_vs_margin",
        "period": "FY2025",
        "date_lo": "2025-01-01", "date_hi": "2025-12-31",
        "question": "Which region had the best profit margin in {period}?",
        "schema": ("Use the `sales_ledger` table in the finance database. Columns: "
                   "txn_id, region, txn_date, revenue, cost_of_goods."),
        "dim_label": "region",
        "trap_metric_col": "total_revenue",
        "correct_metric_col": "profit_margin_pct",
        "trap_unit": "usd_m", "correct_unit": "pct",
        "trap_thought": "I need to find the region with the best profit margin in {period}. Let me first see revenue by region.",
        "correct_thought": "I need to find the region with the best profit margin in {period}. I'll compute (revenue - cost) / revenue per region.",
        "trap_sql": "SELECT region, SUM(revenue) / 1000000.0 as total_revenue FROM sales_ledger WHERE txn_date BETWEEN '{date_lo}' AND '{date_hi}' GROUP BY region ORDER BY total_revenue DESC",
        "correct_sql": "SELECT region, (SUM(revenue) - SUM(cost_of_goods)) * 100.0 / SUM(revenue) as profit_margin_pct FROM sales_ledger WHERE txn_date BETWEEN '{date_lo}' AND '{date_hi}' GROUP BY region ORDER BY profit_margin_pct DESC",
        "ic_dim_label": "quarter",
        "ic_dim_values": ["2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4"],
        "trap_value_range": (40.0, 220.0),
        "correct_value_range": (8.0, 32.0),
        "ic_value_range": (45.0, 75.0),
    },
    {
        "key": "absolute_vs_growth",
        "period": "Q4 2025",
        "date_lo": "2025-10-01", "date_hi": "2025-12-31",
        "question": "Which product line showed the strongest growth in {period} compared to the prior quarter?",
        "schema": ("Use the `product_sales` table in the analytics database. "
                   "Columns: product_line, period_start, units_sold, revenue."),
        "dim_label": "product_line",
        "trap_metric_col": "current_revenue",
        "correct_metric_col": "qoq_growth_pct",
        "trap_unit": "usd_k", "correct_unit": "pct_signed",
        "trap_thought": "I need to find the product line with the strongest growth in {period}. Let me start by listing current-quarter revenue by product line.",
        "correct_thought": "I need to find the product line with the strongest growth in {period}. I'll compute (current - prior) / prior per product line.",
        "trap_sql": "SELECT product_line, SUM(revenue) / 1000.0 as current_revenue FROM product_sales WHERE period_start BETWEEN '{date_lo}' AND '{date_hi}' GROUP BY product_line ORDER BY current_revenue DESC",
        "correct_sql": "WITH cur AS (SELECT product_line, SUM(revenue) AS r FROM product_sales WHERE period_start BETWEEN '{date_lo}' AND '{date_hi}' GROUP BY product_line), prev AS (SELECT product_line, SUM(revenue) AS r FROM product_sales WHERE period_start BETWEEN '2025-07-01' AND '2025-09-30' GROUP BY product_line) SELECT cur.product_line, (cur.r - prev.r) * 100.0 / prev.r AS qoq_growth_pct FROM cur JOIN prev USING(product_line) ORDER BY qoq_growth_pct DESC",
        "ic_dim_label": "month",
        "ic_dim_values": ["2025-10", "2025-11", "2025-12"],
        "trap_value_range": (180.0, 1400.0),
        "correct_value_range": (5.0, 65.0),
        "ic_value_range": (320.0, 480.0),
    },
    {
        "key": "gross_vs_net",
        "period": "2025",
        "date_lo": "2025-01-01", "date_hi": "2025-12-31",
        "question": "Which sales rep generated the most net revenue in {period}?",
        "schema": ("Use the `deals` table in the CRM database. Columns: "
                   "deal_id, sales_rep, close_date, gross_revenue, discount, refund."),
        "dim_label": "sales_rep",
        "trap_metric_col": "gross_revenue_k",
        "correct_metric_col": "net_revenue_k",
        "trap_unit": "usd_k", "correct_unit": "usd_k",
        "trap_thought": "I need to find the sales rep with the most net revenue in {period}. Let me start by querying gross revenue per rep.",
        "correct_thought": "I need to find the sales rep with the most net revenue in {period}. I'll subtract discounts and refunds from gross.",
        "trap_sql": "SELECT sales_rep, SUM(gross_revenue) / 1000.0 as gross_revenue_k FROM deals WHERE close_date BETWEEN '{date_lo}' AND '{date_hi}' GROUP BY sales_rep ORDER BY gross_revenue_k DESC",
        "correct_sql": "SELECT sales_rep, (SUM(gross_revenue) - SUM(discount) - SUM(refund)) / 1000.0 as net_revenue_k FROM deals WHERE close_date BETWEEN '{date_lo}' AND '{date_hi}' GROUP BY sales_rep ORDER BY net_revenue_k DESC",
        "ic_dim_label": "quarter",
        "ic_dim_values": ["2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4"],
        "trap_value_range": (320.0, 1900.0),
        "correct_value_range": (260.0, 1600.0),
        "ic_value_range": (380.0, 620.0),
    },
]



# Entity pools per template. Each entry is a 4-tuple of entity names where
# index 0 is W (trap rank-1 by trap metric) and index 1 is the correct answer
# (rank-1 by correct metric in S0). Indices 2-3 are distractors. All entities
# are synthetic to avoid parametric-knowledge short-circuits.
ENTITY_POOLS: Dict[str, List[Tuple[str, ...]]] = {
    "rate_vs_count": [
        ("Meridian Electronics", "Novus Apparel", "Aldon Home", "Crestford Sports"),
        ("Halcyon Audio", "Sable Knitwear", "Pinegrove Decor", "Tideway Outdoor"),
        ("Voltarc Devices", "Wovenline", "Bramble & Co", "Lockstep Athletics"),
        ("Quanta Optics", "Ferncloth", "Hearthwell", "Trailborne"),
        ("Stratus PC Parts", "Lambentwear", "Glenfield Living", "Riverbend Gear"),
        ("Helion Cameras", "Briarweave", "Oakshade Decor", "Steelyard Sports"),
        ("Nimbus Wearables", "Marrowknit", "Quillburn Home", "Foreshore Fitness"),
        ("Pyxis Smart Home", "Cordwell Threads", "Aubernook", "Vanguard Outdoor"),
        ("Cinder Audio", "Petalwoven", "Whitstable Home", "Northpine Athletics"),
        ("Lumen Displays", "Argentknit", "Cobblestone Living", "Surfridge Sports"),
        ("Solstice Phones", "Ravenstitch", "Mossvale Decor", "Crestward Outdoor"),
        ("Vesper Computing", "Tinsel Apparel", "Brookhollow Home", "Highmark Athletics"),
    ],
    "total_vs_per_unit": [
        ("GlobalFreight Logistics", "Quickship Couriers", "Cedar Express", "Vantage Cargo"),
        ("Atlas Hauling", "PinPoint Delivery", "Marlow Transit", "Beacon Freight"),
        ("Continental Shipping", "Pivot Couriers", "Rowanline", "Summit Cargo"),
        ("Sentinel Logistics", "Lattice Express", "Granite Transit", "Foxglen Freight"),
        ("Liberty Cargo", "Switchpoint Couriers", "Hartwell Lines", "Newgate Logistics"),
        ("Pacific Forward", "Kestrel Express", "Ironwood Transit", "Birchgate Cargo"),
        ("Empire Hauling", "Verdant Couriers", "Ridgepath Lines", "Foundry Freight"),
        ("Cardinal Shipping", "Quill Express", "Stonefield Transit", "Highwater Cargo"),
        ("Coastal Logistics", "Spry Couriers", "Holloway Lines", "Aspen Freight"),
        ("Frontier Freight", "Pulsar Express", "Maplecroft Transit", "Kingsmere Cargo"),
        ("Universal Cargo", "Tessera Couriers", "Ashbourne Lines", "Heron Freight"),
        ("Premier Hauling", "Vantage Express", "Wexford Transit", "Brightwater Cargo"),
    ],
    "revenue_vs_margin": [
        ("North America", "Asia-Pacific", "Europe", "Latin America"),
        ("US-East", "MENA", "EU-West", "Brazil"),
        ("Greater China", "Nordics", "ANZ", "Sub-Saharan Africa"),
        ("Western US", "Southeast Asia", "Central Europe", "Andean Region"),
        ("Canada", "Indian Subcontinent", "Iberia", "Caribbean"),
        ("US-Central", "Japan & Korea", "Benelux", "Southern Cone"),
        ("Pacific Northwest", "Greater Mekong", "Eastern Europe", "Caribbean Basin"),
        ("New England", "Oceania", "DACH", "Mexico & CA"),
        ("US-South", "South Asia", "France & Italy", "Brazil & Cone"),
        ("Mid-Atlantic", "Greater ASEAN", "British Isles", "Mercosur"),
        ("Mountain West", "Greater Japan", "CEE", "Central America"),
        ("Great Lakes", "Greater India", "Iberia & France", "Caribbean Rim"),
    ],
    "absolute_vs_growth": [
        ("Enterprise Suite", "Starter Plan", "Pro Tier", "Legacy Edition"),
        ("Vanguard Platform", "Spark Lite", "Studio Pro", "Heritage Bundle"),
        ("Citadel Cloud", "Lumen Free", "Atlas Pro", "Classic Box"),
        ("Pinnacle Server", "Beacon Lite", "Forge Pro", "Vintage Pack"),
        ("Apex Workstation", "Ember Mini", "Loom Pro", "Origin Edition"),
        ("Summit Database", "Pebble Lite", "Crucible Pro", "Heirloom Box"),
        ("Monarch Suite", "Whisk Free", "Vault Pro", "Legacy Bundle"),
        ("Sentinel Cloud", "Mote Lite", "Beacon Pro", "Classic Suite"),
        ("Olympus Platform", "Glint Mini", "Anvil Pro", "Heritage Pack"),
        ("Titan Workstation", "Drift Lite", "Quill Pro", "Vintage Edition"),
        ("Magnate Server", "Pulse Free", "Quarry Pro", "Origin Box"),
        ("Imperial Suite", "Twig Lite", "Forge Studio", "Heirloom Pack"),
    ],
    "gross_vs_net": [
        ("Jordan Blake", "Morgan Chen", "Avery Patel", "Riley Okafor"),
        ("Casey Nguyen", "Drew Mancini", "Sky Tanaka", "Reese Adamou"),
        ("Logan Park", "Quinn Esposito", "Sage Kapoor", "Hayden Marchetti"),
        ("Parker Devine", "Rowan Velasco", "Indie Brunner", "Emerson Yates"),
        ("Taylor Halloran", "Kit Petrov", "Marlowe Singh", "Bellamy Cruz"),
        ("Alex Brennan", "Sam Lindqvist", "Jules Kowalski", "Carter Mwangi"),
        ("Spencer Goh", "Reagan Ferro", "Lennox Iyer", "Briar Holm"),
        ("Sloane Tanaka", "Ellis Kohli", "Wren Castelli", "Auden Petrov"),
        ("Hollis Park", "Murphy Vargas", "Dexter Halim", "Sutton Acharya"),
        ("Beckett Ng", "Frankie Solberg", "Linden Quan", "Pacey Tomic"),
        ("Indigo Reyes", "Marlow Faltin", "Robin Tahir", "Kennedy Olstad"),
        ("Sutton Hines", "Hadley Larsen", "Camryn Doshi", "Tatum Rosso"),
    ],
}



def _fmt_value(v: float, unit: str) -> str:
    if unit == "int":
        return f"{int(round(v)):,}"
    if unit == "pct":
        return f"{v:.1f}%"
    if unit == "pct_signed":
        return f"{v:+.1f}%"
    if unit == "usd":
        return f"${int(round(v)):,}"
    if unit == "usd_decimal":
        return f"${v:.2f}"
    if unit == "usd_m":
        return f"${v:.1f}M"
    if unit == "usd_k":
        return f"${v:.1f}K"
    return str(v)


def _table(header_dim: str, header_val: str, rows: List[Tuple[str, str]],
           note: str = "") -> str:
    dim_w = max(len(header_dim), max((len(r[0]) for r in rows), default=0))
    val_w = max(len(header_val), max((len(r[1]) for r in rows), default=0))
    sep = f"|{'-'*(dim_w+2)}|{'-'*(val_w+2)}|"
    head = f"| {header_dim.ljust(dim_w)} | {header_val.ljust(val_w)} |"
    body = "\n".join(f"| {r[0].ljust(dim_w)} | {r[1].rjust(val_w)} |" for r in rows)
    out = "Query executed successfully.\n\n" + head + "\n" + sep + "\n" + body + \
          f"\n\n{len(rows)} rows returned."
    if note:
        out += "\n" + note
    return out


def _empty_table(header_dim: str, header_val: str, note: str) -> str:
    sep = f"|{'-'*(len(header_dim)+2)}|{'-'*(len(header_val)+2)}|"
    head = f"| {header_dim} | {header_val} |"
    return ("Query executed successfully.\n\n" + head + "\n" + sep +
            "\n\n0 rows returned.\n" + note)


def _gen_descending_values(rng: random.Random, n: int, lo: float, hi: float,
                           is_int: bool) -> List[float]:
    """Return n values strictly descending in [lo, hi]. Spread roughly evenly
    with some jitter so the trap rank-1 entity is clearly highest."""
    span = hi - lo
    base = [hi - span * (i / max(n - 1, 1)) for i in range(n)]
    out = []
    for i, b in enumerate(base):
        jitter = span * 0.04 * (rng.random() - 0.5)
        v = max(lo, min(hi, b + jitter))
        out.append(int(round(v)) if is_int else round(v, 2))
    # Re-sort to enforce strict descending after jitter.
    out.sort(reverse=True)
    return out


@dataclass
class SQLItem:
    item_id: str
    template_key: str
    question: str
    schema: str
    period: str
    date_lo: str
    date_hi: str
    trap_sql: str
    correct_sql: str
    trap_thought: str
    correct_thought: str
    dim_label: str
    trap_metric_col: str
    correct_metric_col: str
    W_entity: str           # trap rank-1 entity (commit_W target in T0/N0/IC)
    correct_entity: str     # rank-1 in S0 by correct metric (commit_correct target)
    other_entities: List[str]
    obs_T0: str
    obs_N0: str
    obs_S0: str
    obs_IC: str
    expected: Dict[str, str] = field(default_factory=lambda: {
        "N0": "action", "T0": "action", "IC": "action", "S0": "final",
    })

    def to_dict(self) -> dict:
        return asdict(self)


def _build_item(item_id: str, idx_in_template: int, tmpl: dict,
                rng: random.Random) -> SQLItem:
    pool = ENTITY_POOLS[tmpl["key"]]
    entities = pool[idx_in_template % len(pool)]
    W = entities[0]
    correct = entities[1]
    others = list(entities[2:])

    # T0 rows: W is rank 1 by trap metric. Place correct in a middle rank
    # (rank 2 or 3) so it's still in the table but not at the top.
    t0_order = [W, others[0], correct, others[1]] if len(others) >= 2 else [W, correct]
    n_t0 = len(t0_order)
    t0_vals = _gen_descending_values(
        rng, n_t0, *tmpl["trap_value_range"],
        is_int=(tmpl["trap_unit"] == "int"),
    )
    t0_rows = [(t0_order[i], _fmt_value(t0_vals[i], tmpl["trap_unit"]))
               for i in range(n_t0)]

    # S0 rows: correct is rank 1 by correct metric. W is at the bottom.
    s0_order = [correct, others[0], others[1], W] if len(others) >= 2 else [correct, W]
    n_s0 = len(s0_order)
    s0_vals = _gen_descending_values(
        rng, n_s0, *tmpl["correct_value_range"], is_int=False,
    )
    s0_rows = [(s0_order[i], _fmt_value(s0_vals[i], tmpl["correct_unit"]))
               for i in range(n_s0)]

    # N0: empty result with a partition note.
    n0_note = (f"(0 rows returned \u2014 no records matched the WHERE filter "
               f"on {tmpl['date_lo']}..{tmpl['date_hi']}. Partition `archive_"
               f"{tmpl['date_lo'][:4]}` may need to be attached.)")

    # IC: same trap metric column, but aggregated by a non-entity dim.
    ic_dim_vals = list(tmpl["ic_dim_values"])
    n_ic = len(ic_dim_vals)
    ic_vals = _gen_descending_values(
        rng, n_ic, *tmpl["ic_value_range"],
        is_int=(tmpl["trap_unit"] == "int"),
    )
    rng.shuffle(ic_vals)
    ic_rows = [(ic_dim_vals[i], _fmt_value(ic_vals[i], tmpl["trap_unit"]))
               for i in range(n_ic)]

    obs_T0 = _table(tmpl["dim_label"], tmpl["trap_metric_col"], t0_rows)
    obs_S0 = _table(tmpl["dim_label"], tmpl["correct_metric_col"], s0_rows)
    obs_N0 = _empty_table(tmpl["dim_label"], tmpl["trap_metric_col"], n0_note)
    obs_IC = _table(tmpl["ic_dim_label"], tmpl["trap_metric_col"], ic_rows)

    period = tmpl["period"]
    return SQLItem(
        item_id=item_id, template_key=tmpl["key"],
        question=tmpl["question"].format(period=period),
        schema=tmpl["schema"], period=period,
        date_lo=tmpl["date_lo"], date_hi=tmpl["date_hi"],
        trap_sql=tmpl["trap_sql"].format(
            date_lo=tmpl["date_lo"], date_hi=tmpl["date_hi"]),
        correct_sql=tmpl["correct_sql"].format(
            date_lo=tmpl["date_lo"], date_hi=tmpl["date_hi"]),
        trap_thought=tmpl["trap_thought"].format(period=period),
        correct_thought=tmpl["correct_thought"].format(period=period),
        dim_label=tmpl["dim_label"],
        trap_metric_col=tmpl["trap_metric_col"],
        correct_metric_col=tmpl["correct_metric_col"],
        W_entity=W, correct_entity=correct, other_entities=others,
        obs_T0=obs_T0, obs_N0=obs_N0, obs_S0=obs_S0, obs_IC=obs_IC,
    )


def build_items(n_items: int = 60, seed: int = 20260501) -> List[SQLItem]:
    """Round-robin across templates so the first 5 items cover all families."""
    rng = random.Random(seed)
    n_per_template = max(len(p) for p in ENTITY_POOLS.values())
    n_max = len(TEMPLATES) * n_per_template
    if n_items > n_max:
        raise ValueError(f"n_items {n_items} exceeds {n_max} unique combos")
    items = []
    for i in range(n_items):
        tmpl = TEMPLATES[i % len(TEMPLATES)]
        idx_in_template = i // len(TEMPLATES)
        items.append(_build_item(f"sql_{i+1:03d}", idx_in_template, tmpl, rng))
    return items


def verify_item_invariants(it: SQLItem) -> List[str]:
    errs = []
    if it.W_entity == it.correct_entity:
        errs.append(f"{it.item_id}: W and correct entity coincide")
    if it.W_entity not in it.obs_T0:
        errs.append(f"{it.item_id}: W not in obs_T0")
    if it.correct_entity not in it.obs_S0:
        errs.append(f"{it.item_id}: correct entity not in obs_S0")
    if it.W_entity in it.obs_N0:
        errs.append(f"{it.item_id}: W leaks into obs_N0")
    if it.W_entity in it.obs_IC:
        errs.append(f"{it.item_id}: W leaks into obs_IC")
    if it.correct_entity in it.obs_N0:
        errs.append(f"{it.item_id}: correct entity leaks into obs_N0")
    if it.correct_entity in it.obs_IC:
        errs.append(f"{it.item_id}: correct entity leaks into obs_IC")
    # Rank-1 check for T0: the first data row must contain W.
    t0_lines = [ln for ln in it.obs_T0.splitlines() if ln.startswith("|") and "---" not in ln]
    if len(t0_lines) >= 2 and it.W_entity not in t0_lines[1]:
        errs.append(f"{it.item_id}: W not at rank-1 in obs_T0")
    s0_lines = [ln for ln in it.obs_S0.splitlines() if ln.startswith("|") and "---" not in ln]
    if len(s0_lines) >= 2 and it.correct_entity not in s0_lines[1]:
        errs.append(f"{it.item_id}: correct entity not at rank-1 in obs_S0")
    L_T0, L_IC = len(it.obs_T0), len(it.obs_IC)
    if L_IC < 0.4 * L_T0 or L_IC > 1.8 * L_T0:
        errs.append(f"{it.item_id}: IC length {L_IC} out of 0.4-1.8x of T0 ({L_T0})")
    return errs


if __name__ == "__main__":
    items = build_items(60)
    bad = [e for it in items for e in verify_item_invariants(it)]
    if bad:
        print("INVARIANT VIOLATIONS:")
        for e in bad[:30]:
            print(" ", e)
    else:
        print(f"Built {len(items)} items. All invariants pass.")
    it = items[0]
    print(f"\n--- {it.item_id}: {it.template_key} ---")
    print(f"question: {it.question}")
    print(f"schema:   {it.schema}")
    print(f"W:        {it.W_entity}")
    print(f"correct:  {it.correct_entity}")
    print(f"trap_sql: {it.trap_sql}")
    for tag, obs in [("N0", it.obs_N0), ("T0", it.obs_T0),
                     ("IC", it.obs_IC), ("S0", it.obs_S0)]:
        print(f"\n[{tag}] (len={len(obs)})\n{obs}")
