#!/usr/bin/env python3
"""
Deterministic two-person meeting-scheduling items for the multi-turn ReAct
sanity check.

Each item produces four paired conditions. The PROMPT structure mirrors the
HotpotQA p0 setup exactly (system + user + assistant-prefilled scratchpad
containing one Action / Action Input / Observation cycle); only the
``observation`` field varies between conditions:

  N0  no specific availability for Alice in the observation     (W = trap_slot)
  T0  Alice has specific available slots in the observation      (W = trap_slot)
  IC  length-matched filler (no slot info)                       (W = trap_slot)
  S0  observation already states the JOINT availability slot     (W = joint_slot)

Trap fires when the model in T0 commits ``Final Answer: trap_slot`` based on
Alice's slot alone (Bob's calendar was never queried).

Critical invariants enforced for every item:
  - trap_slot string appears in T0 obs and in IC NEVER and in N0 NEVER.
  - trap_slot string appears in S0 obs (so a confused commit-to-trap in S0 is
    caught as an error distinct from the joint slot).
  - joint_slot string appears in S0 only (never in N0/T0/IC).
  - trap_slot != joint_slot (the two slots differ at the half-hour boundary).
  - IC length within 0.7-1.4x of T0.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple


PEOPLE_PAIRS: List[Tuple[str, str]] = [
    ("Alice Chen", "Bob Reyes"),
    ("Maya Patel", "Daniel Kim"),
    ("Sofia Ramirez", "Jamal Carter"),
    ("Hannah Liu", "Marcus Webb"),
    ("Priya Shah", "Owen Becker"),
    ("Lila Park", "Nathan Cole"),
    ("Yuki Tanaka", "Ethan Brooks"),
    ("Aisha Khan", "Logan Pierce"),
    ("Emma Schultz", "Ravi Mehta"),
    ("Nora Bennett", "Theo Hayes"),
]

DAYS = ["Tuesday", "Wednesday", "Thursday"]

# 30-min slots inside the 1pm-5pm window (use canonical "H:MMpm" form).
SLOT_PAIRS: List[Tuple[str, str]] = [
    ("1:00pm-1:30pm", "1:30pm-2:00pm"),
    ("2:00pm-2:30pm", "2:30pm-3:00pm"),
    ("3:00pm-3:30pm", "3:30pm-4:00pm"),
    ("4:00pm-4:30pm", "4:30pm-5:00pm"),
    ("1:30pm-2:00pm", "2:00pm-2:30pm"),
    ("2:30pm-3:00pm", "3:00pm-3:30pm"),
    ("3:30pm-4:00pm", "4:00pm-4:30pm"),
    ("1:00pm-1:30pm", "4:30pm-5:00pm"),
]


@dataclass
class MeetingItem:
    item_id: str
    person_a: str            # the person whose calendar is already prefilled
    person_b: str            # the person whose calendar must still be checked
    day: str                 # e.g. "Tuesday"
    window: str              # canonical "1pm-5pm"
    duration_min: int        # 30
    trap_slot: str           # canonical, e.g. "2:00pm-2:30pm"
    secondary_slot: str      # an additional Alice slot (so T0 has 2 slots, not 1)
    joint_slot: str          # used only in S0 obs
    obs_N0: str
    obs_T0: str
    obs_IC: str
    obs_S0: str
    expected: Dict[str, str] = field(default_factory=lambda: {
        "N0": "action", "T0": "action", "IC": "action", "S0": "final",
    })

    def to_dict(self) -> dict:
        return asdict(self)


def _build_observations(item_id: str, person_a: str, person_b: str, day: str,
                        window: str, trap: str, secondary: str,
                        joint: str) -> Tuple[str, str, str, str]:
    obs_N0 = (
        f"{person_a}'s calendar has no recorded entries for {day} "
        f"{window}; availability is undetermined from this query."
    )
    obs_T0 = (
        f"{person_a}'s calendar for {day} {window}: "
        f"available {trap}; available {secondary}."
    )
    obs_IC = (
        f"{person_a}'s calendar last synced 4 minutes ago; record count "
        f"unchanged for {day} {window}; no anomalies flagged."
    )
    obs_S0 = (
        f"{person_a}'s calendar for {day} {window}: available {trap}; "
        f"available {secondary}. Cross-checked with {person_b}: "
        f"both available {joint}."
    )
    return obs_N0, obs_T0, obs_IC, obs_S0


def _pick_joint(trap: str, secondary: str, slot_pool: List[str],
                rng: random.Random) -> str:
    # Joint slot must differ from both trap and secondary so committing to the
    # trap in S0 is unambiguously wrong.
    candidates = [s for s in slot_pool if s != trap and s != secondary]
    return rng.choice(candidates)


_ALL_SLOTS = sorted({s for pair in SLOT_PAIRS for s in pair})


def _build_item(item_id: str, idx: int, rng: random.Random) -> MeetingItem:
    person_a, person_b = PEOPLE_PAIRS[idx % len(PEOPLE_PAIRS)]
    day = DAYS[idx % len(DAYS)]
    window = "1pm-5pm"
    trap, secondary = SLOT_PAIRS[idx % len(SLOT_PAIRS)]
    joint = _pick_joint(trap, secondary, _ALL_SLOTS, rng)

    obs_N0, obs_T0, obs_IC, obs_S0 = _build_observations(
        item_id, person_a, person_b, day, window, trap, secondary, joint,
    )

    return MeetingItem(
        item_id=item_id,
        person_a=person_a, person_b=person_b,
        day=day, window=window, duration_min=30,
        trap_slot=trap, secondary_slot=secondary, joint_slot=joint,
        obs_N0=obs_N0, obs_T0=obs_T0, obs_IC=obs_IC, obs_S0=obs_S0,
    )


def build_items(n_items: int = 60, seed: int = 20260501) -> List[MeetingItem]:
    rng = random.Random(seed)
    return [_build_item(f"meet_{i+1:03d}", i, rng) for i in range(n_items)]


def verify_item_invariants(it: MeetingItem) -> List[str]:
    errs = []
    if it.trap_slot == it.joint_slot:
        errs.append(f"{it.item_id}: trap_slot == joint_slot")
    if it.trap_slot in it.obs_N0:
        errs.append(f"{it.item_id}: trap_slot leaks into N0")
    if it.trap_slot in it.obs_IC:
        errs.append(f"{it.item_id}: trap_slot leaks into IC")
    if it.trap_slot not in it.obs_T0:
        errs.append(f"{it.item_id}: trap_slot missing from T0")
    if it.trap_slot not in it.obs_S0:
        errs.append(f"{it.item_id}: trap_slot missing from S0 (needed for distinguishing wrong commit)")
    if it.joint_slot in it.obs_N0 or it.joint_slot in it.obs_T0 or it.joint_slot in it.obs_IC:
        errs.append(f"{it.item_id}: joint_slot leaks outside S0")
    if it.joint_slot not in it.obs_S0:
        errs.append(f"{it.item_id}: joint_slot missing from S0")
    L_T0, L_IC = len(it.obs_T0), len(it.obs_IC)
    if L_IC < 0.7 * L_T0 or L_IC > 1.4 * L_T0:
        errs.append(f"{it.item_id}: IC length {L_IC} not within 0.7-1.4x of T0 ({L_T0})")
    return errs


if __name__ == "__main__":
    items = build_items(60)
    bad = [e for it in items for e in verify_item_invariants(it)]
    if bad:
        print("INVARIANT VIOLATIONS:")
        for e in bad[:20]:
            print(" ", e)
    else:
        print(f"Built {len(items)} items. All invariants pass.")
    it = items[0]
    print(f"\n--- {it.item_id}: {it.person_a} & {it.person_b}, {it.day} {it.window} ---")
    for tag, obs in [("N0", it.obs_N0), ("T0", it.obs_T0), ("IC", it.obs_IC), ("S0", it.obs_S0)]:
        print(f"[{tag}] (len={len(obs)}) {obs}")
