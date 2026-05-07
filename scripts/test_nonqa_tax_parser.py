#!/usr/bin/env python3
"""Tests for parse_nonqa_tax_outputs.py and nonqa_tax_items.py invariants."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest

from parse_nonqa_tax_outputs import normalize_money, parse_output
from nonqa_tax_items import build_items, verify_item_invariants


# ---------------------------------------------------------------------------
# normalize_money
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("$127.50", "$127.50"),
    ("127.50", "$127.50"),
    ("$1,127.50", "$1,127.50"),
    ("1127.50", "$1,127.50"),
    ("$ 1,127.50", "$1,127.50"),
    ("$1,127.5", "$1,127.50"),
    ("$1,127", "$1,127.00"),
    ("  $999.00  ", "$999.00"),
    ("$1,234,567.89", "$1,234,567.89"),
    ("12345", "$12,345.00"),
])
def test_normalize_money_ok(raw, expected):
    assert normalize_money(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", None, "abc", "no money here"])
def test_normalize_money_bad(raw):
    assert normalize_money(raw) is None


# ---------------------------------------------------------------------------
# parse_output search
# ---------------------------------------------------------------------------

def test_parse_search_action_form():
    p = parse_output("Action: GetTaxRate[CA-94103]", "$127.50")
    assert p.first_is_search == 1
    assert p.search_target == "CA-94103"
    assert p.commit_W == 0
    assert p.parse_failure == 0


def test_parse_search_bare():
    p = parse_output("GetTaxRate[NY-10013]", "$127.50")
    assert p.first_is_search == 1
    assert p.search_target == "NY-10013"


def test_parse_search_case_insensitive():
    p = parse_output("action: gettaxrate[TX-78701]", "$127.50")
    assert p.first_is_search == 1


# ---------------------------------------------------------------------------
# parse_output final + commit_W
# ---------------------------------------------------------------------------

def test_parse_final_commit_W():
    p = parse_output("Final: AMOUNT=$127.50", "$127.50")
    assert p.first_is_final == 1
    assert p.commit_W == 1
    assert p.final_amount == "$127.50"


def test_parse_final_no_commit_different_amount():
    p = parse_output("Final: AMOUNT=$138.66", "$127.50")
    assert p.first_is_final == 1
    assert p.commit_W == 0
    assert p.final_amount == "$138.66"


def test_parse_final_bare():
    p = parse_output("AMOUNT=$127.50", "$127.50")
    assert p.first_is_final == 1
    assert p.commit_W == 1


def test_parse_final_with_commas():
    p = parse_output("Final: AMOUNT=$1,247.85", "$1,247.85")
    assert p.first_is_final == 1
    assert p.commit_W == 1


def test_parse_final_no_dollar_sign():
    p = parse_output("Final: AMOUNT=127.50", "$127.50")
    assert p.first_is_final == 1
    assert p.commit_W == 1


def test_parse_final_strips_trailing_period():
    p = parse_output("Final: AMOUNT=$127.50.", "$127.50")
    # The regex includes the trailing dot in cents handling. Confirm normalization.
    # Normalize "127.50." -> "$127.50".
    assert p.first_is_final == 1
    # We accept either equality on normalized form.
    assert p.commit_W == 1


# ---------------------------------------------------------------------------
# parse_output edge cases
# ---------------------------------------------------------------------------

def test_parse_empty():
    p = parse_output("", "$127.50")
    assert p.parse_failure == 1
    assert p.first_is_search == 0 and p.first_is_final == 0


def test_parse_garbage():
    p = parse_output("I think the tax rate is unknown.", "$127.50")
    assert p.parse_failure == 1


def test_parse_first_line_only():
    p = parse_output("Action: GetTaxRate[CA-94103]\nFinal: AMOUNT=$127.50", "$127.50")
    assert p.first_is_search == 1
    assert p.first_is_final == 0


def test_parse_leading_blank_lines():
    p = parse_output("\n\n  Final: AMOUNT=$127.50", "$127.50")
    assert p.first_is_final == 1
    assert p.commit_W == 1


# ---------------------------------------------------------------------------
# Item invariants (regression test on the generator)
# ---------------------------------------------------------------------------

def test_all_120_items_invariants():
    items = build_items(120)
    assert len(items) == 120
    bad = []
    for it in items:
        bad.extend(verify_item_invariants(it))
    assert not bad, f"Invariant violations: {bad[:10]}"


def test_subtotal_total_distinct_per_item():
    items = build_items(120)
    for it in items:
        assert it.W_subtotal_str != it.W_total_str, it.item_id
