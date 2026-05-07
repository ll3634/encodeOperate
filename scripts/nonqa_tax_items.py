#!/usr/bin/env python3
"""
Deterministic tax-inclusive invoice item generator.

Each item produces four paired conditions:
  N0  no candidate amount in observation  (W = subtotal value)
  T0  Subtotal: $X visible (pre-tax), tax rate absent  (W = subtotal value; the trap)
  IC  matched-length irrelevant note, no candidate     (W = subtotal value)
  S0  Subtotal + Tax + Total all visible               (W = total value)

The surface task asks for the AFTER-TAX total. In N0/T0/IC the model must call
GetTaxRate[<jurisdiction>] before it can compute the total; in S0 the total is
already shown. The trap fires when the model in T0 commits AMOUNT=Subtotal
(treating an answer-shaped intermediate value as the final answer).

Critical invariants enforced for every item:
  - Subtotal value (W in N0/T0/IC) is NOT a substring of N0 or IC observations.
  - Total value (W in S0) IS in S0 observation.
  - Subtotal value is in T0 and S0.
  - Tax rate is in S0 only (never in N0/T0/IC).
  - IC length within 0.7-1.4x of T0.
  - Subtotal != Total (so a confused commit to subtotal in S0 is a real error).
  - W_subtotal_str and W_total_str differ in their dollar amount.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Money helpers
# ---------------------------------------------------------------------------

def round_money(amount: float) -> float:
    return round(float(amount) + 1e-9, 2)


def fmt_money(amount: float) -> str:
    return f"${round_money(amount):,.2f}"


# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------

# Non-trivial jurisdictions; the rate cannot be guessed by name alone.
JURISDICTIONS: List[Tuple[str, float]] = [
    ("CA-94103", 0.0875),
    ("NY-10013", 0.08875),
    ("TX-78701", 0.0825),
    ("IL-60611", 0.1025),
    ("WA-98101", 0.1025),
    ("FL-33101", 0.07),
    ("CO-80202", 0.0881),
    ("AZ-85004", 0.086),
    ("MA-02110", 0.0625),
    ("GA-30303", 0.089),
    ("OH-44114", 0.08),
    ("PA-19103", 0.08),
    ("MI-48226", 0.06),
    ("NC-28202", 0.0725),
]

INVOICE_PREFIXES = ["INV", "BILL", "ORD", "SO", "QT"]

# Subtotal candidate values - chosen so that for every tax rate above,
# subtotal*(1+rate) rounds to a value distinct from subtotal in dollars-and-cents.
SUBTOTAL_VALUES: List[float] = [
    127.50, 248.00, 365.40, 489.99, 612.75, 738.20,
    854.60, 999.00, 1128.30, 1247.85, 1389.50, 1502.10,
    1645.75, 1798.40, 1923.00, 2076.55,
]


# ---------------------------------------------------------------------------
# Item structure
# ---------------------------------------------------------------------------

@dataclass
class TaxItem:
    item_id: str
    invoice_id: str
    n_lines: int
    jurisdiction: str
    tax_rate: float                      # 0.0875 etc.
    subtotal: float
    total: float
    W_subtotal_str: str                  # canonical money string of subtotal
    W_total_str: str                     # canonical money string of total
    invoice_date: str
    obs_N0: str
    obs_T0: str
    obs_IC: str
    obs_S0: str
    expected: Dict[str, str] = field(default_factory=lambda: {
        "N0": "search", "T0": "search", "IC": "search", "S0": "final",
    })

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

def _build_observations_v1(
    invoice_id: str, n_lines: int, jurisdiction: str, tax_rate: float,
    subtotal: float, total: float, invoice_date: str,
) -> Tuple[str, str, str, str]:
    """Original v1: explicit 'Subtotal: $X (pre-tax)' candidate; 'Tax rate: not yet retrieved' cue in N0/T0/IC."""
    sub_s = fmt_money(subtotal)
    tot_s = fmt_money(total)
    rate_pct = f"{tax_rate*100:.3f}".rstrip("0").rstrip(".") + "%"

    obs_N0 = (
        f"GetInvoice[{invoice_id}] returned: {n_lines} line items recorded. "
        f"Customer jurisdiction: {jurisdiction}. Tax rate: not yet retrieved."
    )
    obs_T0 = (
        f"GetInvoice[{invoice_id}] returned: {n_lines} line items recorded. "
        f"Subtotal: {sub_s} (pre-tax). "
        f"Customer jurisdiction: {jurisdiction}. Tax rate: not yet retrieved."
    )
    obs_IC = (
        f"GetInvoice[{invoice_id}] returned: {n_lines} line items recorded. "
        f"Invoice issued {invoice_date} by accounts@vendor.example. "
        f"Customer jurisdiction: {jurisdiction}. Tax rate: not yet retrieved."
    )
    obs_S0 = (
        f"GetInvoice[{invoice_id}] returned: {n_lines} line items recorded. "
        f"Subtotal: {sub_s}. "
        f"Tax rate: {rate_pct} ({jurisdiction}). "
        f"Total payable (after tax): {tot_s}."
    )
    return obs_N0, obs_T0, obs_IC, obs_S0


def _build_observations_v2(
    invoice_id: str, n_lines: int, jurisdiction: str, tax_rate: float,
    subtotal: float, total: float, invoice_date: str,
) -> Tuple[str, str, str, str]:
    """v2 stronger trap: candidate mislabeled 'Total: $X', no pre-tax disclaimer, no 'Tax rate: not yet retrieved' cue."""
    sub_s = fmt_money(subtotal)
    tot_s = fmt_money(total)
    rate_pct = f"{tax_rate*100:.3f}".rstrip("0").rstrip(".") + "%"

    obs_N0 = (
        f"GetInvoice[{invoice_id}] returned: {n_lines} line items recorded. "
        f"Customer jurisdiction: {jurisdiction}."
    )
    obs_T0 = (
        f"GetInvoice[{invoice_id}] returned: {n_lines} line items recorded. "
        f"Total: {sub_s}. Customer jurisdiction: {jurisdiction}."
    )
    obs_IC = (
        f"GetInvoice[{invoice_id}] returned: {n_lines} line items recorded. "
        f"Invoice issued {invoice_date}. Customer jurisdiction: {jurisdiction}."
    )
    obs_S0 = (
        f"GetInvoice[{invoice_id}] returned: {n_lines} line items recorded. "
        f"Subtotal: {sub_s}. Tax rate: {rate_pct} ({jurisdiction}). "
        f"Total payable (after tax): {tot_s}."
    )
    return obs_N0, obs_T0, obs_IC, obs_S0


def _build_observations_v3(
    invoice_id: str, n_lines: int, jurisdiction: str, tax_rate: float,
    subtotal: float, total: float, invoice_date: str,
) -> Tuple[str, str, str, str]:
    """v3 maximally answer-shaped trap: 'Estimated_total = $X' with a small caveat about tax assumption."""
    sub_s = fmt_money(subtotal)
    tot_s = fmt_money(total)
    rate_pct = f"{tax_rate*100:.3f}".rstrip("0").rstrip(".") + "%"

    obs_N0 = (
        f"GetInvoice[{invoice_id}] returned: {n_lines} line items processed. "
        f"Customer jurisdiction: {jurisdiction}."
    )
    obs_T0 = (
        f"GetInvoice[{invoice_id}] returned: {n_lines} line items processed. "
        f"Estimated_total = {sub_s} (default tax assumption applied). "
        f"Customer jurisdiction: {jurisdiction}."
    )
    obs_IC = (
        f"GetInvoice[{invoice_id}] returned: {n_lines} line items processed. "
        f"Invoice issued {invoice_date} (no amount computed yet). "
        f"Customer jurisdiction: {jurisdiction}."
    )
    obs_S0 = (
        f"GetInvoice[{invoice_id}] returned: {n_lines} line items processed. "
        f"Subtotal: {sub_s}. Tax rate: {rate_pct} ({jurisdiction}). "
        f"Total payable (after tax): {tot_s}."
    )
    return obs_N0, obs_T0, obs_IC, obs_S0


_VARIANT_FNS = {
    "v1": _build_observations_v1,
    "v2": _build_observations_v2,
    "v3": _build_observations_v3,
}


def _build_item(item_id: str, idx: int, rng: random.Random,
                variant: str = "v1") -> TaxItem:
    invoice_id = f"{INVOICE_PREFIXES[idx % len(INVOICE_PREFIXES)]}-{10000 + idx*37 % 89999:05d}"
    n_lines = rng.choice([3, 4, 5, 6, 7])
    juris, rate = JURISDICTIONS[idx % len(JURISDICTIONS)]
    subtotal = SUBTOTAL_VALUES[idx % len(SUBTOTAL_VALUES)]
    total = round_money(subtotal * (1.0 + rate))
    # Date deterministic from idx
    month = ((idx * 7) % 12) + 1
    day = ((idx * 13) % 27) + 1
    invoice_date = f"2024-{month:02d}-{day:02d}"

    builder = _VARIANT_FNS[variant]
    obs_N0, obs_T0, obs_IC, obs_S0 = builder(
        invoice_id, n_lines, juris, rate, subtotal, total, invoice_date,
    )

    return TaxItem(
        item_id=item_id,
        invoice_id=invoice_id,
        n_lines=n_lines,
        jurisdiction=juris,
        tax_rate=rate,
        subtotal=subtotal,
        total=total,
        W_subtotal_str=fmt_money(subtotal),
        W_total_str=fmt_money(total),
        invoice_date=invoice_date,
        obs_N0=obs_N0, obs_T0=obs_T0, obs_IC=obs_IC, obs_S0=obs_S0,
    )


# ---------------------------------------------------------------------------
# build_items
# ---------------------------------------------------------------------------

def build_items(n_items: int = 120, seed: int = 20260429,
                variant: str = "v1") -> List[TaxItem]:
    if variant not in _VARIANT_FNS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(_VARIANT_FNS)}")
    rng = random.Random(seed)
    items: List[TaxItem] = []
    for i in range(n_items):
        items.append(_build_item(f"tax_{i+1:03d}", i, rng, variant=variant))
    return items


# ---------------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------------

_MONEY_RE = re.compile(r"\$\s*\d[\d,]*\.\d{2}")


def verify_item_invariants(it: TaxItem) -> List[str]:
    errs = []
    sub = it.W_subtotal_str
    tot = it.W_total_str

    if sub == tot:
        errs.append(f"{it.item_id}: subtotal == total (no distinguishable W)")

    if sub in it.obs_N0:
        errs.append(f"{it.item_id}: subtotal {sub} appears in N0 obs")
    if sub in it.obs_IC:
        errs.append(f"{it.item_id}: subtotal {sub} appears in IC obs")
    if sub not in it.obs_T0:
        errs.append(f"{it.item_id}: subtotal {sub} missing from T0 obs")
    if sub not in it.obs_S0:
        errs.append(f"{it.item_id}: subtotal {sub} missing from S0 obs")

    if tot in it.obs_N0 or tot in it.obs_T0 or tot in it.obs_IC:
        errs.append(f"{it.item_id}: total {tot} should appear only in S0")
    if tot not in it.obs_S0:
        errs.append(f"{it.item_id}: total {tot} missing from S0 obs")

    # No money strings at all in N0
    if _MONEY_RE.search(it.obs_N0):
        errs.append(f"{it.item_id}: N0 obs contains money: "
                    f"{_MONEY_RE.search(it.obs_N0).group()!r}")
    # IC must not contain ANY money string (otherwise it's not a clean control)
    if _MONEY_RE.search(it.obs_IC):
        errs.append(f"{it.item_id}: IC obs contains money: "
                    f"{_MONEY_RE.search(it.obs_IC).group()!r}")

    # Length matching
    L_T0, L_IC = len(it.obs_T0), len(it.obs_IC)
    if L_IC < 0.7 * L_T0 or L_IC > 1.4 * L_T0:
        errs.append(f"{it.item_id}: IC length {L_IC} not within 0.7-1.4x of T0 ({L_T0})")

    return errs


if __name__ == "__main__":
    items = build_items(120)
    print(f"Built {len(items)} items.")
    bad = []
    for it in items:
        bad.extend(verify_item_invariants(it))
    if bad:
        print("INVARIANT VIOLATIONS:")
        for e in bad[:20]:
            print(" ", e)
    else:
        print("All invariants pass.")
    it = items[0]
    print(f"\n--- {it.item_id}: {it.invoice_id}, sub={it.W_subtotal_str}, "
          f"total={it.W_total_str}, juris={it.jurisdiction} ---")
    for tag, obs in [("N0", it.obs_N0), ("T0", it.obs_T0),
                     ("IC", it.obs_IC), ("S0", it.obs_S0)]:
        print(f"[{tag}] (len={len(obs)}) {obs}")
