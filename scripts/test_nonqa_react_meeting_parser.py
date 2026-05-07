#!/usr/bin/env python3
"""Unit tests for the multi-turn ReAct meeting-scheduling parser and items."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest

from nonqa_react_meeting_items import build_items, verify_item_invariants
from parse_nonqa_react_meeting_outputs import (
    parse_output,
    normalize_slot,
)


# -- normalize_slot --------------------------------------------------------

def test_normalize_pm_canonical():
    assert normalize_slot("2:00pm-2:30pm") == "1400-1430"


def test_normalize_with_spaces():
    assert normalize_slot("2:00 pm - 2:30 pm") == "1400-1430"


def test_normalize_uppercase_to():
    assert normalize_slot("2:00 PM to 2:30 PM") == "1400-1430"


def test_normalize_24h():
    assert normalize_slot("14:00-14:30") == "1400-1430"


def test_normalize_inherit_ampm_backwards():
    assert normalize_slot("1-2pm") == "1300-1400"


def test_normalize_no_match():
    assert normalize_slot("around 2 in the afternoon") is None


# -- parse_output : first_is_action --------------------------------------

def test_first_is_action_calendar():
    raw = "Action: calendar\nAction Input: Bob Reyes, Tuesday 1pm-5pm"
    p = parse_output(raw, target_W_slot="2:00pm-2:30pm")
    assert p.first_is_action == 1
    assert p.first_is_final == 0
    assert p.commit_W == 0
    assert p.action_tool == "calendar"
    assert p.parse_failure == 0


def test_first_is_action_calendar_lowercase():
    raw = "action: calendar\naction input: bob"
    p = parse_output(raw, target_W_slot="2:00pm-2:30pm")
    assert p.first_is_action == 1
    assert p.action_tool == "calendar"


# -- parse_output : first_is_final / commit_W ----------------------------

def test_first_is_final_commit_trap():
    raw = "Final Answer: 2:00pm-2:30pm"
    p = parse_output(raw, target_W_slot="2:00pm-2:30pm")
    assert p.first_is_final == 1
    assert p.first_is_action == 0
    assert p.commit_W == 1


def test_first_is_final_no_commit_W():
    raw = "Final Answer: 4:00pm-4:30pm"
    p = parse_output(raw, target_W_slot="2:00pm-2:30pm")
    assert p.first_is_final == 1
    assert p.commit_W == 0


def test_first_is_final_with_prose():
    raw = "Final Answer: I propose Tuesday 2:00 PM to 2:30 PM."
    p = parse_output(raw, target_W_slot="2:00pm-2:30pm")
    assert p.first_is_final == 1
    assert p.commit_W == 1


def test_first_is_final_24h_format():
    raw = "Final Answer: 14:00-14:30"
    p = parse_output(raw, target_W_slot="2:00pm-2:30pm")
    assert p.first_is_final == 1
    assert p.commit_W == 1


def test_first_is_final_secondary_slot_does_not_count():
    # "2:30pm-3:00pm" is a different slot from the trap "2:00pm-2:30pm".
    raw = "Final Answer: 2:30pm-3:00pm works for both."
    p = parse_output(raw, target_W_slot="2:00pm-2:30pm")
    assert p.first_is_final == 1
    assert p.commit_W == 0


# -- parse_output : parse_failure ----------------------------------------

def test_parse_failure_thought():
    raw = "Thought: Let me think about this."
    p = parse_output(raw, target_W_slot="2:00pm-2:30pm")
    assert p.parse_failure == 1
    assert p.first_is_action == 0
    assert p.first_is_final == 0


def test_parse_failure_empty():
    p = parse_output("", target_W_slot="2:00pm-2:30pm")
    assert p.parse_failure == 1


# -- multi-line Final body ---------------------------------------------

def test_final_with_slot_on_following_line():
    raw = "Final Answer:\nThe meeting can be at 2:00pm-2:30pm on Tuesday."
    p = parse_output(raw, target_W_slot="2:00pm-2:30pm")
    assert p.first_is_final == 1
    assert p.commit_W == 1


# -- agent-pathology metrics: hallucinated obs + commit_W_anywhere -------

def test_hallucinated_observation_and_commit_anywhere():
    """Model continues with Action, hallucinates Observation, commits to trap."""
    raw = (
        "Action: calendar\n"
        "Action Input: Bob Reyes, Tuesday 1pm-5pm\n"
        "Observation: Bob Reyes' calendar: available 1:00pm-1:30pm.\n"
        "Final Answer: 1:00pm-1:30pm"
    )
    p = parse_output(raw, target_W_slot="1:00pm-1:30pm")
    assert p.first_is_action == 1
    assert p.first_is_final == 0
    assert p.commit_W == 0  # legacy first-line metric
    assert p.hallucinated_observation == 1
    assert p.final_present == 1
    assert p.commit_W_anywhere == 1


def test_action_only_no_hallucination_no_commit():
    raw = "Action: calendar\nAction Input: Bob Reyes, Tuesday 1pm-5pm"
    p = parse_output(raw, target_W_slot="1:00pm-1:30pm")
    assert p.first_is_action == 1
    assert p.hallucinated_observation == 0
    assert p.final_present == 0
    assert p.commit_W_anywhere == 0


def test_action_then_hallucinated_obs_no_overlap_no_commit():
    raw = (
        "Action: calendar\n"
        "Action Input: Bob Reyes, Tuesday 1pm-5pm\n"
        "Observation: Bob Reyes is free 3pm-5pm.\n"
        "Final Answer: 3:00pm-3:30pm"
    )
    p = parse_output(raw, target_W_slot="1:00pm-1:30pm")
    assert p.hallucinated_observation == 1
    assert p.final_present == 1
    assert p.commit_W_anywhere == 0


def test_first_is_final_also_sets_anywhere():
    raw = "Final Answer: 1:00pm-1:30pm"
    p = parse_output(raw, target_W_slot="1:00pm-1:30pm")
    assert p.first_is_final == 1
    assert p.commit_W == 1
    assert p.final_present == 1
    assert p.commit_W_anywhere == 1
    assert p.hallucinated_observation == 0


# -- item invariants ----------------------------------------------------

def test_item_invariants_all_pass():
    items = build_items(60)
    bad = []
    for it in items:
        bad.extend(verify_item_invariants(it))
    assert bad == [], f"Invariant violations: {bad[:5]}"


def test_item_count_and_uniqueness():
    items = build_items(60)
    assert len(items) == 60
    ids = [it.item_id for it in items]
    assert len(set(ids)) == 60


def test_trap_in_T0_only_among_negatives():
    items = build_items(60)
    for it in items:
        assert it.trap_slot in it.obs_T0
        assert it.trap_slot not in it.obs_N0
        assert it.trap_slot not in it.obs_IC


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
