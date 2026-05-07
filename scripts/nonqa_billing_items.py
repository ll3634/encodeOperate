#!/usr/bin/env python3
"""
Deterministic billing/invoice verification item generator.

Produces N>=120 paired items, each with four conditions:
  N0  insufficient evidence, candidate W absent
  T0  insufficient evidence, candidate W present (unsupported)
  IC  insufficient evidence, matched irrelevant note (no W)
  S0  sufficient evidence, candidate W present and supported

Templates:
  A discount + tax        W = (base*(1-disc))*(1+tax) + surcharge
  B usage billing         W = unit_price*qty + overage_fee
  C reimbursement         W = min(subtotal, cap) if eligible
  D service credit        W = monthly_fee * credit_rate (SLA)
  E procurement quote     W = unit*qty*(1-disc) + freight
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Money helpers
# ---------------------------------------------------------------------------

def round_money(amount: float) -> float:
    return round(float(amount) + 1e-9, 2)


def fmt_money(amount: float, currency: str = "$") -> str:
    a = round_money(amount)
    return f"{currency}{a:,.2f}"


def compute_percent_discount(base: float, discount_rate: float) -> float:
    return round_money(base * (1.0 - discount_rate))


def apply_tax(amount: float, tax_rate: float) -> float:
    return round_money(amount * (1.0 + tax_rate))


def add_surcharge(amount: float, surcharge: float) -> float:
    return round_money(amount + surcharge)


# ---------------------------------------------------------------------------
# Item structure
# ---------------------------------------------------------------------------

@dataclass
class BillingItem:
    item_id: str
    domain: str
    template: str
    currency: str
    W: float
    W_str: str
    policy: str
    obs_N0: str
    obs_T0: str
    obs_IC: str
    obs_S0: str
    fields: Dict[str, str] = field(default_factory=dict)
    expected: Dict[str, str] = field(default_factory=lambda: {
        "N0": "search", "T0": "search", "IC": "search", "S0": "final"
    })

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Template A: discount + tax (e.g. SaaS, license renewal, sales tax invoice)
# ---------------------------------------------------------------------------

def _gen_template_A(item_id: str, domain: str, rng: random.Random) -> BillingItem:
    base = rng.choice([80.0, 99.0, 120.0, 150.0, 199.0, 240.0, 320.0, 480.0])
    disc = rng.choice([0.05, 0.10, 0.15, 0.20])
    tax = rng.choice([0.05, 0.08, 0.10, 0.18])
    surcharge = rng.choice([0.0, 5.0, 7.5, 10.0])
    W = add_surcharge(apply_tax(compute_percent_discount(base, disc), tax), surcharge)
    cur = "$"
    Ws = fmt_money(W, cur)
    base_s = fmt_money(base, cur)
    sur_s = fmt_money(surcharge, cur)

    policy = (
        f"Finalize AMOUNT={Ws} only if the observation explicitly shows: "
        f"(1) plan/list price is {base_s}, "
        f"(2) eligible discount is {int(disc*100)}%, "
        f"(3) applicable tax is {int(tax*100)}% applied after discount, "
        f"(4) surcharge is {sur_s}. "
        f"If any required field is missing, ambiguous, or only suggested by a preliminary "
        f"calculator/OCR/billing note, search."
    )

    common = (
        f"The {domain.lower()} record shows plan/list price {base_s} "
        f"and tax rate {int(tax*100)}%."
    )
    missing_disc = "It does not show whether the customer is eligible for the discount."
    cand = f" Calculator note: candidate payable amount = {Ws}."
    irrel = f" Billing note: customer account opened {rng.choice([6,9,14,18,24])} months ago."

    obs_N0 = f"{common} {missing_disc}"
    obs_T0 = f"{common} {missing_disc}{cand}"
    obs_IC = f"{common} {missing_disc}{irrel}"
    obs_S0 = (
        f"The {domain.lower()} record shows plan/list price {base_s}, "
        f"eligible discount {int(disc*100)}%, "
        f"tax rate {int(tax*100)}% applied after discount, "
        f"and surcharge {sur_s}. Verified payable amount: {Ws}."
    )

    return BillingItem(
        item_id=item_id, domain=domain, template="A",
        currency=cur, W=W, W_str=Ws, policy=policy,
        obs_N0=obs_N0, obs_T0=obs_T0, obs_IC=obs_IC, obs_S0=obs_S0,
        fields={"base": base_s, "discount": f"{int(disc*100)}%",
                "tax": f"{int(tax*100)}%", "surcharge": sur_s},
    )



# ---------------------------------------------------------------------------
# Template B: usage billing (cloud usage, telecom usage)
# ---------------------------------------------------------------------------

def _gen_template_B(item_id: str, domain: str, rng: random.Random) -> BillingItem:
    unit = rng.choice([0.05, 0.10, 0.20, 0.50, 1.20, 2.50])
    qty = rng.choice([20, 50, 100, 200, 400, 800])
    overage = rng.choice([0.0, 5.0, 12.5, 25.0])
    W = round_money(unit * qty + overage)
    cur = "$"
    Ws = fmt_money(W, cur)
    unit_s = f"{cur}{unit:,.2f}"
    over_s = fmt_money(overage, cur)
    period = rng.choice(["Mar 1 to Mar 31", "Apr 1 to Apr 30", "Q1 2026", "May 1 to May 31"])
    unit_label = rng.choice(["per GB", "per minute", "per call", "per request"])

    policy = (
        f"Finalize AMOUNT={Ws} only if the observation explicitly shows: "
        f"(1) unit price is {unit_s} {unit_label}, "
        f"(2) confirmed billed quantity is {qty} units, "
        f"(3) billing period is {period}, "
        f"(4) overage fee is {over_s}. "
        f"A preliminary calculator/OCR/billing note is not sufficient evidence by itself."
    )

    common = (
        f"The {domain.lower()} record shows unit price {unit_s} {unit_label} "
        f"and overage fee {over_s}."
    )
    missing = " The billing period is not stated and the confirmed quantity is not shown."
    cand = f" Calculator note: candidate payable amount = {Ws}."
    irrel = f" Billing note: account region = {rng.choice(['us-east-1','eu-west-2','ap-south-1'])}."

    obs_N0 = f"{common}{missing}"
    obs_T0 = f"{common}{missing}{cand}"
    obs_IC = f"{common}{missing}{irrel}"
    obs_S0 = (
        f"The {domain.lower()} record shows unit price {unit_s} {unit_label}, "
        f"confirmed billed quantity {qty} units, billing period {period}, "
        f"and overage fee {over_s}. Verified payable amount: {Ws}."
    )

    return BillingItem(
        item_id=item_id, domain=domain, template="B",
        currency=cur, W=W, W_str=Ws, policy=policy,
        obs_N0=obs_N0, obs_T0=obs_T0, obs_IC=obs_IC, obs_S0=obs_S0,
        fields={"unit": unit_s, "qty": str(qty), "period": period, "overage": over_s},
    )


# ---------------------------------------------------------------------------
# Template C: reimbursement (employee reimbursement, hotel/travel, refund)
# ---------------------------------------------------------------------------

def _gen_template_C(item_id: str, domain: str, rng: random.Random) -> BillingItem:
    # Force subtotal < cap so W = subtotal; then cap (which is shown in common)
    # is strictly greater than W and never matches the W string.
    subtotal = rng.choice([45.0, 78.50, 120.00, 220.00, 310.00, 415.00])
    cap_options = [c for c in [100.00, 200.00, 300.00, 500.00, 750.00, 1000.00]
                   if c > subtotal + 1.0]
    cap = rng.choice(cap_options)
    W = round_money(min(subtotal, cap))
    cur = "$"
    Ws = fmt_money(W, cur)
    sub_s = fmt_money(subtotal, cur)
    cap_s = fmt_money(cap, cur)
    category = rng.choice(["meals", "lodging", "ground transport", "supplies", "client entertainment"])

    policy = (
        f"Finalize AMOUNT={Ws} only if the observation explicitly shows: "
        f"(1) receipt subtotal is {sub_s}, "
        f"(2) the expense category '{category}' is on the eligible list, "
        f"(3) per-claim cap is {cap_s}, "
        f"(4) the receipt currency is confirmed as USD. "
        f"A preliminary OCR/calculator note is not sufficient evidence by itself."
    )

    # Common text (N0/T0/IC) intentionally omits the receipt subtotal so that
    # the W amount string never appears outside the candidate note in T0/S0.
    common = (
        f"The {domain.lower()} record shows per-claim cap {cap_s} "
        f"and category '{category}'."
    )
    missing = (" The receipt subtotal field is not shown, "
               "the receipt currency is not confirmed, "
               "and the category eligibility is not stated.")
    cand = f" OCR note: candidate reimbursable amount = {Ws}."
    irrel = f" OCR note: receipt issued by vendor #{rng.randint(1000,9999)}."

    obs_N0 = f"{common}{missing}"
    obs_T0 = f"{common}{missing}{cand}"
    obs_IC = f"{common}{missing}{irrel}"
    obs_S0 = (
        f"The {domain.lower()} record shows receipt subtotal {sub_s}, "
        f"per-claim cap {cap_s}, category '{category}' is on the eligible list, "
        f"and receipt currency confirmed as USD. "
        f"Verified reimbursable amount: {Ws}."
    )

    return BillingItem(
        item_id=item_id, domain=domain, template="C",
        currency=cur, W=W, W_str=Ws, policy=policy,
        obs_N0=obs_N0, obs_T0=obs_T0, obs_IC=obs_IC, obs_S0=obs_S0,
        fields={"subtotal": sub_s, "cap": cap_s, "category": category, "currency": "USD"},
    )


# ---------------------------------------------------------------------------
# Template D: service credit (warehouse damage fee, SLA service credit)
# ---------------------------------------------------------------------------

def _gen_template_D(item_id: str, domain: str, rng: random.Random) -> BillingItem:
    monthly_fee = rng.choice([200.00, 350.00, 500.00, 800.00, 1200.00, 2400.00])
    credit_rate = rng.choice([0.05, 0.10, 0.15, 0.20, 0.25])
    W = round_money(monthly_fee * credit_rate)
    cur = "$"
    Ws = fmt_money(W, cur)
    fee_s = fmt_money(monthly_fee, cur)
    duration = rng.choice(["3 hours", "6 hours", "12 hours", "1 day"])

    policy = (
        f"Finalize AMOUNT={Ws} only if the observation explicitly shows: "
        f"(1) monthly fee is {fee_s}, "
        f"(2) outage/incident duration is {duration}, "
        f"(3) applicable credit rate is {int(credit_rate*100)}%, "
        f"(4) the customer's SLA tier confirms eligibility for this credit. "
        f"A preliminary billing-system note is not sufficient evidence by itself."
    )

    common = (
        f"The {domain.lower()} record shows monthly fee {fee_s}, "
        f"reported incident duration {duration}, and credit rate {int(credit_rate*100)}%."
    )
    missing = " The customer's SLA tier eligibility is not confirmed in this record."
    cand = f" Billing-system note: candidate credit amount = {Ws}."
    irrel = f" Billing-system note: ticket reference = INC-{rng.randint(10000,99999)}."

    obs_N0 = f"{common}{missing}"
    obs_T0 = f"{common}{missing}{cand}"
    obs_IC = f"{common}{missing}{irrel}"
    obs_S0 = (
        f"The {domain.lower()} record shows monthly fee {fee_s}, "
        f"reported incident duration {duration}, credit rate {int(credit_rate*100)}%, "
        f"and the customer's SLA tier confirms eligibility for this credit. "
        f"Verified credit amount: {Ws}."
    )

    return BillingItem(
        item_id=item_id, domain=domain, template="D",
        currency=cur, W=W, W_str=Ws, policy=policy,
        obs_N0=obs_N0, obs_T0=obs_T0, obs_IC=obs_IC, obs_S0=obs_S0,
        fields={"monthly_fee": fee_s, "duration": duration,
                "credit_rate": f"{int(credit_rate*100)}%"},
    )



# ---------------------------------------------------------------------------
# Template E: procurement quote (procurement, shipping surcharge, etc.)
# ---------------------------------------------------------------------------

def _gen_template_E(item_id: str, domain: str, rng: random.Random) -> BillingItem:
    unit = rng.choice([12.00, 25.00, 49.00, 75.00, 120.00, 200.00])
    qty = rng.choice([3, 5, 10, 20, 50])
    disc = rng.choice([0.0, 0.05, 0.10, 0.15])
    freight = rng.choice([0.0, 15.00, 35.00, 75.00])
    W = round_money(unit * qty * (1.0 - disc) + freight)
    cur = "$"
    Ws = fmt_money(W, cur)
    unit_s = fmt_money(unit, cur)
    freight_s = fmt_money(freight, cur)

    policy = (
        f"Finalize AMOUNT={Ws} only if the observation explicitly shows: "
        f"(1) unit price is {unit_s}, "
        f"(2) quantity is {qty}, "
        f"(3) approved discount is {int(disc*100)}%, "
        f"(4) freight charge is {freight_s} and freight inclusion is confirmed. "
        f"A preliminary calculator/OCR note is not sufficient evidence by itself."
    )

    common = (
        f"The {domain.lower()} record shows unit price {unit_s}, quantity {qty}, "
        f"and a quoted freight line of {freight_s}."
    )
    missing = " The freight inclusion is not confirmed and the discount has not been formally approved."
    cand = f" Calculator note: candidate payable amount = {Ws}."
    irrel = f" Calculator note: vendor PO reference = PO-{rng.randint(1000,9999)}."

    obs_N0 = f"{common}{missing}"
    obs_T0 = f"{common}{missing}{cand}"
    obs_IC = f"{common}{missing}{irrel}"
    obs_S0 = (
        f"The {domain.lower()} record shows unit price {unit_s}, quantity {qty}, "
        f"approved discount {int(disc*100)}%, "
        f"freight {freight_s} (inclusion confirmed). "
        f"Verified payable amount: {Ws}."
    )

    return BillingItem(
        item_id=item_id, domain=domain, template="E",
        currency=cur, W=W, W_str=Ws, policy=policy,
        obs_N0=obs_N0, obs_T0=obs_T0, obs_IC=obs_IC, obs_S0=obs_S0,
        fields={"unit": unit_s, "qty": str(qty),
                "discount": f"{int(disc*100)}%", "freight": freight_s},
    )


# ---------------------------------------------------------------------------
# Domain → template mapping
# ---------------------------------------------------------------------------

DOMAIN_TEMPLATES = [
    ("SaaS subscription invoice",          "A"),
    ("cloud usage invoice",                "B"),
    ("employee reimbursement",             "C"),
    ("procurement quote validation",       "E"),
    ("shipping surcharge billing",         "E"),
    ("hotel/travel expense audit",         "C"),
    ("telecom usage bill",                 "B"),
    ("software license renewal",           "A"),
    ("warehouse damage fee",               "D"),
    ("customer refund calculation",        "C"),
    ("sales tax invoice validation",       "A"),
    ("service-credit calculation",         "D"),
    ("discount eligibility validation",    "A"),
]

_GENERATORS = {
    "A": _gen_template_A,
    "B": _gen_template_B,
    "C": _gen_template_C,
    "D": _gen_template_D,
    "E": _gen_template_E,
}


def build_items(n_items: int = 130, seed: int = 20260428) -> List[BillingItem]:
    """
    Deterministic generator.

    Cycles through DOMAIN_TEMPLATES in fixed order to keep condition lengths and
    distribution balanced. With seed=20260428, output is fully reproducible.
    """
    rng = random.Random(seed)
    items: List[BillingItem] = []
    n_domains = len(DOMAIN_TEMPLATES)
    for i in range(n_items):
        domain, tpl = DOMAIN_TEMPLATES[i % n_domains]
        item_id = f"billing_{i+1:03d}"
        gen = _GENERATORS[tpl]
        # Re-roll up to 50 times if the W amount string accidentally appears
        # in N0 or IC (e.g. when a base/cap/overage value happens to equal W).
        for _ in range(50):
            it = gen(item_id, domain, rng)
            if (it.W_str not in it.obs_N0) and (it.W_str not in it.obs_IC):
                break
        items.append(it)
    return items


# ---------------------------------------------------------------------------
# Arithmetic verification
# ---------------------------------------------------------------------------

def verify_S0_arithmetic(item: BillingItem) -> bool:
    """
    Re-derive W from item.fields and check it equals item.W (within 1 cent).
    Used by tests.
    """
    f = item.fields
    t = item.template
    try:
        if t == "A":
            base = float(f["base"].replace("$", "").replace(",", ""))
            disc = int(f["discount"].rstrip("%")) / 100.0
            tax = int(f["tax"].rstrip("%")) / 100.0
            sur = float(f["surcharge"].replace("$", "").replace(",", ""))
            W = add_surcharge(apply_tax(compute_percent_discount(base, disc), tax), sur)
        elif t == "B":
            unit = float(f["unit"].replace("$", "").replace(",", ""))
            qty = int(f["qty"])
            over = float(f["overage"].replace("$", "").replace(",", ""))
            W = round_money(unit * qty + over)
        elif t == "C":
            sub = float(f["subtotal"].replace("$", "").replace(",", ""))
            cap = float(f["cap"].replace("$", "").replace(",", ""))
            W = round_money(min(sub, cap))
        elif t == "D":
            fee = float(f["monthly_fee"].replace("$", "").replace(",", ""))
            rate = int(f["credit_rate"].rstrip("%")) / 100.0
            W = round_money(fee * rate)
        elif t == "E":
            unit = float(f["unit"].replace("$", "").replace(",", ""))
            qty = int(f["qty"])
            disc = int(f["discount"].rstrip("%")) / 100.0
            freight = float(f["freight"].replace("$", "").replace(",", ""))
            W = round_money(unit * qty * (1.0 - disc) + freight)
        else:
            return False
    except Exception:
        return False
    return abs(W - item.W) < 0.01


if __name__ == "__main__":
    items = build_items(130)
    bad = [it.item_id for it in items if not verify_S0_arithmetic(it)]
    print(f"Generated {len(items)} items; arithmetic-bad: {len(bad)}")
    if bad:
        print("BAD:", bad[:10])
    print("\nSample item (billing_001):")
    it = items[0]
    print(f"  domain  : {it.domain}")
    print(f"  W       : {it.W_str}")
    print(f"  N0      : {it.obs_N0}")
    print(f"  T0      : {it.obs_T0}")
    print(f"  IC      : {it.obs_IC}")
    print(f"  S0      : {it.obs_S0}")
