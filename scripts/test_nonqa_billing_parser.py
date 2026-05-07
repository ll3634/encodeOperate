#!/usr/bin/env python3
"""
Unit tests for the calculator-assisted billing-verification sanity check.

Covers:
  - Output parser
  - Amount normalization
  - S0 arithmetic consistency for all generated items
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from nonqa_billing_items import build_items, verify_S0_arithmetic
from parse_nonqa_billing_outputs import (
    normalize_amount,
    parse_output,
    first_nonempty_line,
)


# ---------------------------------------------------------------------------
# normalize_amount
# ---------------------------------------------------------------------------

def test_normalize_amount_basic():
    assert normalize_amount("$128.40") == "128.40"
    assert normalize_amount("128.40") == "128.40"
    assert normalize_amount("  $128.40 ") == "128.40"


def test_normalize_amount_thousands():
    assert normalize_amount("$1,234.50") == "1234.50"
    assert normalize_amount("1,234.50") == "1234.50"


def test_normalize_amount_currency_kept():
    assert normalize_amount("$128.40", normalize_currency=False) == "$128.40"
    assert normalize_amount("128.40", normalize_currency=False) == "128.40"


def test_normalize_amount_int():
    assert normalize_amount("$100") == "100.00"


def test_normalize_amount_garbage():
    assert normalize_amount("") is None
    assert normalize_amount("abc") is None
    assert normalize_amount(None) is None


def test_normalize_amount_negative():
    assert normalize_amount("-$45.00") == "-45.00"
    assert normalize_amount("-45.00") == "-45.00"


# ---------------------------------------------------------------------------
# first_nonempty_line
# ---------------------------------------------------------------------------

def test_first_nonempty_line():
    assert first_nonempty_line("\n\n  Action: Search[x]  \nFinal: AMOUNT=$1") == "Action: Search[x]"
    assert first_nonempty_line("Final: AMOUNT=$10") == "Final: AMOUNT=$10"
    assert first_nonempty_line("") == ""


# ---------------------------------------------------------------------------
# parse_output - search
# ---------------------------------------------------------------------------

def test_parse_search_clean():
    p = parse_output("Action: Search[discount eligibility]", "$128.40")
    assert p.first_is_search == 1
    assert p.first_is_final == 0
    assert p.parse_failure == 0
    assert p.commit_W == 0
    assert p.search_query_text == "discount eligibility"


def test_parse_search_extra_leading_blank():
    p = parse_output("\n\nAction: Search[verify]\n", "$50.00")
    assert p.first_is_search == 1
    assert p.parse_failure == 0


def test_parse_search_only_action_no_brackets_is_failure():
    # "Action:" with no Search[...] pattern is not the allowed format
    p = parse_output("Action: search for invoice", "$10.00")
    assert p.first_is_search == 0
    assert p.parse_failure == 1


def test_parse_search_bare_no_action_prefix():
    # Model often drops the "Action:" prefix; bare Search[...] is accepted.
    p = parse_output("Search[discount eligibility]", "$128.40")
    assert p.first_is_search == 1
    assert p.parse_failure == 0
    assert p.search_query_text == "discount eligibility"


def test_parse_final_bare_no_final_prefix():
    p = parse_output("AMOUNT=$128.40", "$128.40")
    assert p.first_is_final == 1
    assert p.commit_W == 1
    assert p.parse_failure == 0


# ---------------------------------------------------------------------------
# parse_output - final
# ---------------------------------------------------------------------------

def test_parse_final_match_dollar():
    p = parse_output("Final: AMOUNT=$128.40", "$128.40")
    assert p.first_is_final == 1
    assert p.commit_W == 1
    assert p.parse_failure == 0


def test_parse_final_match_no_dollar():
    p = parse_output("Final: AMOUNT=128.40", "$128.40")
    assert p.first_is_final == 1
    assert p.commit_W == 1


def test_parse_final_wrong_amount():
    p = parse_output("Final: AMOUNT=$200.00", "$128.40")
    assert p.first_is_final == 1
    assert p.commit_W == 0
    assert p.final_amount == "200.00"


def test_parse_final_amount_with_trailing_punct():
    p = parse_output("Final: AMOUNT=$128.40.", "$128.40")
    assert p.commit_W == 1


def test_parse_final_amount_with_thousands():
    p = parse_output("Final: AMOUNT=$1,234.50", "$1,234.50")
    assert p.commit_W == 1


def test_parse_garbage():
    p = parse_output("I think the amount is around $128.40", "$128.40")
    assert p.parse_failure == 1
    assert p.first_is_search == 0
    assert p.first_is_final == 0
    assert p.commit_W == 0


def test_parse_empty():
    p = parse_output("", "$128.40")
    assert p.parse_failure == 1


# ---------------------------------------------------------------------------
# Item arithmetic consistency
# ---------------------------------------------------------------------------

def test_all_items_arithmetic():
    items = build_items(130)
    bad = [it.item_id for it in items if not verify_S0_arithmetic(it)]
    assert not bad, f"Arithmetic failures: {bad}"


def test_T0_contains_W_N0_does_not():
    items = build_items(130)
    for it in items:
        assert it.W_str in it.obs_T0, f"T0 missing W for {it.item_id}"
        assert it.W_str in it.obs_S0, f"S0 missing W for {it.item_id}"
        assert it.W_str not in it.obs_N0, f"N0 has W for {it.item_id}"
        assert it.W_str not in it.obs_IC, f"IC has W for {it.item_id}"


def test_item_count_and_uniqueness():
    items = build_items(130)
    assert len(items) == 130
    assert len({it.item_id for it in items}) == 130


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
