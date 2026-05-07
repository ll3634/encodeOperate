#!/usr/bin/env python3
"""
Unit tests for the meeting-scheduling sanity check parser and item generator.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from nonqa_meeting_items import build_items, verify_item_invariants
from parse_nonqa_meeting_outputs import (
    normalize_slot,
    parse_output,
    first_nonempty_line,
)


# ---------------------------------------------------------------------------
# normalize_slot
# ---------------------------------------------------------------------------

def test_normalize_slot_canonical():
    assert normalize_slot("Tue 14:00") == "Tue 14:00"
    assert normalize_slot("Mon 09:00") == "Mon 09:00"


def test_normalize_slot_full_day_name():
    assert normalize_slot("Tuesday 14:00") == "Tue 14:00"
    assert normalize_slot("Wednesday 09:00") == "Wed 09:00"


def test_normalize_slot_12h():
    assert normalize_slot("Tue 2 PM") == "Tue 14:00"
    assert normalize_slot("Tue 2:00 PM") == "Tue 14:00"
    assert normalize_slot("Mon 9 AM") == "Mon 09:00"
    assert normalize_slot("Mon 12 PM") == "Mon 12:00"
    assert normalize_slot("Mon 12 AM") == "Mon 00:00"


def test_normalize_slot_with_at():
    assert normalize_slot("Tue at 14:00") == "Tue 14:00"
    assert normalize_slot("Tuesday at 2 PM") == "Tue 14:00"


def test_normalize_slot_case_insensitive():
    assert normalize_slot("tue 14:00") == "Tue 14:00"
    assert normalize_slot("TUE 14:00") == "Tue 14:00"


def test_normalize_slot_garbage():
    assert normalize_slot("") is None
    assert normalize_slot(None) is None
    assert normalize_slot("Sunday lunch") is None
    assert normalize_slot("Tue 14:30") is None  # half-hour not allowed
    assert normalize_slot("Tue 25:00") is None


# ---------------------------------------------------------------------------
# first_nonempty_line
# ---------------------------------------------------------------------------

def test_first_nonempty_line():
    assert first_nonempty_line("\n\n  Action: GetCalendar[Bob]  \n") == "Action: GetCalendar[Bob]"
    assert first_nonempty_line("Final: SLOT=Tue 14:00") == "Final: SLOT=Tue 14:00"
    assert first_nonempty_line("") == ""


# ---------------------------------------------------------------------------
# parse_output - search (GetCalendar)
# ---------------------------------------------------------------------------

def test_parse_search_clean():
    p = parse_output("Action: GetCalendar[Bob]", "Tue 14:00")
    assert p.first_is_search == 1
    assert p.first_is_final == 0
    assert p.parse_failure == 0
    assert p.commit_W == 0
    assert p.search_target == "Bob"


def test_parse_search_bare_no_action_prefix():
    p = parse_output("GetCalendar[Bob]", "Tue 14:00")
    assert p.first_is_search == 1
    assert p.parse_failure == 0
    assert p.search_target == "Bob"


def test_parse_search_extra_blank_lines():
    p = parse_output("\n\nAction: GetCalendar[Carol]\n", "Mon 10:00")
    assert p.first_is_search == 1
    assert p.parse_failure == 0


def test_parse_action_without_brackets_is_failure():
    p = parse_output("Action: check Bob's calendar", "Tue 14:00")
    assert p.first_is_search == 0
    assert p.parse_failure == 1


# ---------------------------------------------------------------------------
# parse_output - final
# ---------------------------------------------------------------------------

def test_parse_final_match_canonical():
    p = parse_output("Final: SLOT=Tue 14:00", "Tue 14:00")
    assert p.first_is_final == 1
    assert p.commit_W == 1
    assert p.parse_failure == 0
    assert p.final_slot == "Tue 14:00"


def test_parse_final_bare_no_final_prefix():
    p = parse_output("SLOT=Tue 14:00", "Tue 14:00")
    assert p.first_is_final == 1
    assert p.commit_W == 1


def test_parse_final_match_12h():
    p = parse_output("Final: SLOT=Tue 2 PM", "Tue 14:00")
    assert p.commit_W == 1


def test_parse_final_match_full_day_name():
    p = parse_output("Final: SLOT=Tuesday at 2 PM", "Tue 14:00")
    assert p.commit_W == 1


def test_parse_final_wrong_slot():
    p = parse_output("Final: SLOT=Wed 11:00", "Tue 14:00")
    assert p.first_is_final == 1
    assert p.commit_W == 0
    assert p.final_slot == "Wed 11:00"


def test_parse_final_trailing_period():
    p = parse_output("Final: SLOT=Tue 14:00.", "Tue 14:00")
    assert p.commit_W == 1


def test_parse_garbage():
    p = parse_output("I think Tue 14:00 works.", "Tue 14:00")
    assert p.parse_failure == 1
    assert p.commit_W == 0


def test_parse_empty():
    p = parse_output("", "Tue 14:00")
    assert p.parse_failure == 1


# ---------------------------------------------------------------------------
# Item invariants
# ---------------------------------------------------------------------------

def test_all_items_invariants():
    items = build_items(120)
    bad = []
    for it in items:
        bad.extend(verify_item_invariants(it))
    assert not bad, f"Invariant violations: {bad[:5]}"


def test_item_count_and_uniqueness():
    items = build_items(120)
    assert len(items) == 120
    assert len({it.item_id for it in items}) == 120


def test_w_appears_in_all_A_free_lists():
    items = build_items(120)
    for it in items:
        assert it.W_str in it.A_free, it.item_id
        assert it.W_str in it.obs_N0, it.item_id  # via A_free
        assert it.W_str in it.obs_IC, it.item_id  # via A_free


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
