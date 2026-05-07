#!/usr/bin/env python3
"""
v2 items for the multi-turn ReAct meeting-scheduling sanity check.

v1 mixed two pathologies (premature stop AND hallucinated observation). v2
isolates **premature stop at p0** to mirror the QA main mechanism cleanly:

  - One tool: find_joint_availability(person_a, person_b, day, window).
  - Prefilled scratchpad: a single tool call has already been issued; the
    Observation contains a CANDIDATE joint slot (the surface candidate).
  - User prompt states an unambiguous CONSTRAINT that the trap_slot violates
    and the joint_slot respects.
  - Decision point: model must either (a) commit Final Answer with a slot
    that respects the constraint, or (b) issue another Action / refuse.
  - Premature stop = first_is_final=1 AND commit to the unconstrained
    surface slot (W = trap_slot in T0; W = joint_slot in S0).

Conditions:
  N0          obs1 = "no joint candidates returned; query individuals"
  T0          obs1 = "joint availability: TRAP_SLOT" (FULL violation, slot fully inside constraint)
  T_partial   obs1 = "joint availability: PARTIAL_SLOT" (~15-min overlap with constraint)
  T_edge      obs1 = "joint availability: EDGE_SLOT" (touches boundary, NO overlap → valid)
  IC          length-matched filler (no slot in obs)
  S0          obs1 = "joint availability: JOINT_SLOT" (respects constraint, well-separated)

T_edge is a control: a healthy model should commit to it (it's a valid slot). If the
model rejects T_edge at the same rate as T_partial / T0, the apparent gradient
collapses into "model rejects anything near the constraint" (uninteresting).

Invariants (per item):
  - trap_slot violates constraint (overlap = 30 min, FULL).
  - partial_slot overlaps constraint by EXACTLY 15 min.
  - edge_slot does NOT overlap constraint, but touches it.
  - joint_slot has zero overlap and does NOT touch constraint boundary.
  - All four slots are pairwise distinct.
  - Each slot string appears only in its own obs (no leakage).
  - len(obs_IC) within 0.7-1.4x of len(obs_T0).
"""

from __future__ import annotations

import random
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

# (constraint_start_24h, constraint_end_24h, trap_slot, joint_slot, role)
# trap_slot must overlap [constraint_start, constraint_end);
# joint_slot must NOT overlap [constraint_start, constraint_end);
# both slots are 30 min and lie inside 13:00-17:00 (1pm-5pm).
TRAP_SETUPS: List[dict] = [
    {"c_start": 13, "c_end": 14, "trap": "1:00pm-1:30pm", "joint": "2:30pm-3:00pm",
     "role": "team standup"},
    {"c_start": 13, "c_end": 14, "trap": "1:30pm-2:00pm", "joint": "3:00pm-3:30pm",
     "role": "all-hands"},
    {"c_start": 14, "c_end": 15, "trap": "2:00pm-2:30pm", "joint": "3:30pm-4:00pm",
     "role": "deep-focus block"},
    {"c_start": 14, "c_end": 15, "trap": "2:30pm-3:00pm", "joint": "4:00pm-4:30pm",
     "role": "1:1 with manager"},
    {"c_start": 15, "c_end": 16, "trap": "3:00pm-3:30pm", "joint": "4:30pm-5:00pm",
     "role": "client call"},
    {"c_start": 15, "c_end": 16, "trap": "3:30pm-4:00pm", "joint": "1:00pm-1:30pm",
     "role": "weekly review"},
    {"c_start": 16, "c_end": 17, "trap": "4:00pm-4:30pm", "joint": "1:30pm-2:00pm",
     "role": "office-hours block"},
    {"c_start": 16, "c_end": 17, "trap": "4:30pm-5:00pm", "joint": "2:00pm-2:30pm",
     "role": "design review"},
]


def _h12(h24: int) -> str:
    suffix = "am" if h24 < 12 else "pm"
    h = h24 if 1 <= h24 <= 12 else (h24 - 12 if h24 > 12 else 12)
    return f"{h}:00{suffix}"


WINDOW_LO_MIN = 13 * 60   # 1pm
WINDOW_HI_MIN = 17 * 60   # 5pm


def _fmt_min(m: int) -> str:
    h24, mn = m // 60, m % 60
    h12 = h24 - 12 if h24 > 12 else (12 if h24 == 0 else h24)
    suffix = "pm" if h24 >= 12 else "am"
    return f"{h12}:{mn:02d}{suffix}"


def _format_slot(lo_min: int, hi_min: int) -> str:
    return f"{_fmt_min(lo_min)}-{_fmt_min(hi_min)}"


def _compute_aux_slots(c_start: int, c_end: int) -> Tuple[str, str]:
    """Return (partial_slot, edge_slot) for the constraint window [c_start*60, c_end*60).

    partial_slot: 30-min slot straddling a boundary, EXACTLY 15 min overlap.
    edge_slot:    30-min slot adjacent to a boundary, ZERO overlap.
    Both slots must lie inside [WINDOW_LO_MIN, WINDOW_HI_MIN].
    """
    c_lo, c_hi = c_start * 60, c_end * 60
    if c_hi + 30 <= WINDOW_HI_MIN:
        partial = (c_hi - 15, c_hi + 15)
        edge = (c_hi, c_hi + 30)
    elif c_lo - 30 >= WINDOW_LO_MIN:
        partial = (c_lo - 15, c_lo + 15)
        edge = (c_lo - 30, c_lo)
    else:
        raise ValueError(f"cannot fit aux slots for constraint {c_start}-{c_end}")
    return _format_slot(*partial), _format_slot(*edge)


@dataclass
class MeetingItemV2:
    item_id: str
    person_a: str
    person_b: str
    day: str
    window: str           # "1pm-5pm"
    duration_min: int     # 30
    constraint_text: str  # human-readable constraint clause for user prompt
    trap_slot: str        # FULL violation
    partial_slot: str     # ~15-min overlap (partial violation)
    edge_slot: str        # touches boundary, no overlap (valid)
    joint_slot: str       # well-separated, no overlap (valid)
    obs_N0: str
    obs_T0: str
    obs_T_partial: str
    obs_T_edge: str
    obs_IC: str
    obs_S0: str
    expected: Dict[str, str] = field(default_factory=lambda: {
        "N0": "action", "T0": "action", "T_partial": "action",
        "T_edge": "final", "IC": "action", "S0": "final",
    })

    def to_dict(self) -> dict:
        return asdict(self)


def _slot_obs(person_a: str, person_b: str, day: str, window: str, slot: str) -> str:
    return (
        f"find_joint_availability for {person_a} and {person_b} on {day} "
        f"{window}: 30-minute joint slot available {slot}."
    )


def _build_observations(person_a: str, person_b: str, day: str, window: str,
                        trap: str, partial: str, edge: str, joint: str
                        ) -> Tuple[str, str, str, str, str, str]:
    obs_N0 = (
        f"find_joint_availability returned no overlapping 30-minute candidates "
        f"for {person_a} and {person_b} on {day} {window}; query individual "
        f"calendars to refine."
    )
    obs_IC = (
        f"find_joint_availability service nominal; cache last refreshed 4 "
        f"minutes ago; no anomalies reported on the {day} index."
    )
    obs_T0 = _slot_obs(person_a, person_b, day, window, trap)
    obs_T_partial = _slot_obs(person_a, person_b, day, window, partial)
    obs_T_edge = _slot_obs(person_a, person_b, day, window, edge)
    obs_S0 = _slot_obs(person_a, person_b, day, window, joint)
    return obs_N0, obs_T0, obs_T_partial, obs_T_edge, obs_IC, obs_S0


def _build_item(item_id: str, idx: int, rng: random.Random) -> MeetingItemV2:
    person_a, person_b = PEOPLE_PAIRS[idx % len(PEOPLE_PAIRS)]
    day = DAYS[idx % len(DAYS)]
    window = "1pm-5pm"
    setup = TRAP_SETUPS[idx % len(TRAP_SETUPS)]
    trap = setup["trap"]
    joint = setup["joint"]
    partial, edge = _compute_aux_slots(setup["c_start"], setup["c_end"])
    c_start = _h12(setup["c_start"])
    c_end = _h12(setup["c_end"])
    role = setup["role"]
    constraint_text = (
        f"{person_b} has a recurring {role} on {day}s from {c_start} to "
        f"{c_end} that cannot be moved; the meeting MUST NOT overlap this block."
    )
    obs_N0, obs_T0, obs_T_partial, obs_T_edge, obs_IC, obs_S0 = _build_observations(
        person_a, person_b, day, window, trap, partial, edge, joint
    )
    return MeetingItemV2(
        item_id=item_id, person_a=person_a, person_b=person_b,
        day=day, window=window, duration_min=30,
        constraint_text=constraint_text,
        trap_slot=trap, partial_slot=partial,
        edge_slot=edge, joint_slot=joint,
        obs_N0=obs_N0, obs_T0=obs_T0,
        obs_T_partial=obs_T_partial, obs_T_edge=obs_T_edge,
        obs_IC=obs_IC, obs_S0=obs_S0,
    )


def build_items(n_items: int = 60, seed: int = 20260501) -> List[MeetingItemV2]:
    rng = random.Random(seed)
    return [_build_item(f"meet_v2_{i+1:03d}", i, rng) for i in range(n_items)]


def _slot_to_minutes(slot: str) -> Tuple[int, int]:
    # Slots are canonical "H:MMpm-H:MMpm". Parse without the heavyweight regex.
    import re
    m = re.match(r"(\d{1,2}):(\d{2})(am|pm)-(\d{1,2}):(\d{2})(am|pm)", slot)
    assert m, f"unparseable slot {slot!r}"
    def to_min(h, mn, ap):
        h = int(h); mn = int(mn)
        if ap == "pm" and h != 12: h += 12
        if ap == "am" and h == 12: h = 0
        return h * 60 + mn
    return to_min(m.group(1), m.group(2), m.group(3)), \
           to_min(m.group(4), m.group(5), m.group(6))


def verify_item_invariants(it: MeetingItemV2) -> List[str]:
    errs = []
    slots = {"trap": it.trap_slot, "partial": it.partial_slot,
             "edge": it.edge_slot, "joint": it.joint_slot}
    if len(set(slots.values())) != 4:
        errs.append(f"{it.item_id}: slot strings not pairwise distinct: {slots}")

    setup = next((s for s in TRAP_SETUPS
                  if s["trap"] == it.trap_slot and s["joint"] == it.joint_slot),
                 None)
    if setup is None:
        errs.append(f"{it.item_id}: trap/joint pair not in TRAP_SETUPS")
        return errs
    c_lo, c_hi = setup["c_start"] * 60, setup["c_end"] * 60

    def _overlap_min(slot: str) -> int:
        s_lo, s_hi = _slot_to_minutes(slot)
        return max(0, min(s_hi, c_hi) - max(s_lo, c_lo))

    # Overlap budgets: trap=30 (full), partial=15, edge=0, joint=0.
    if _overlap_min(it.trap_slot) != 30:
        errs.append(f"{it.item_id}: trap_slot overlap != 30 min")
    if _overlap_min(it.partial_slot) != 15:
        errs.append(f"{it.item_id}: partial_slot overlap != 15 min "
                    f"(got {_overlap_min(it.partial_slot)})")
    if _overlap_min(it.edge_slot) != 0:
        errs.append(f"{it.item_id}: edge_slot overlap != 0 min")
    if _overlap_min(it.joint_slot) != 0:
        errs.append(f"{it.item_id}: joint_slot overlap != 0 min")

    # Edge must TOUCH the boundary; joint must NOT.
    e_lo, e_hi = _slot_to_minutes(it.edge_slot)
    j_lo, j_hi = _slot_to_minutes(it.joint_slot)
    if not (e_hi == c_lo or e_lo == c_hi):
        errs.append(f"{it.item_id}: edge_slot does not touch constraint boundary")
    if (j_hi == c_lo or j_lo == c_hi):
        errs.append(f"{it.item_id}: joint_slot touches constraint boundary "
                    f"(should be well-separated)")

    # Slot string leakage: each string only in its own obs.
    obs_map = {"trap": it.obs_T0, "partial": it.obs_T_partial,
               "edge": it.obs_T_edge, "joint": it.obs_S0}
    no_slot_obs = {"N0": it.obs_N0, "IC": it.obs_IC}
    for name, sval in slots.items():
        for other_name, other_obs in obs_map.items():
            if other_name == name:
                continue
            if sval in other_obs:
                errs.append(f"{it.item_id}: {name}_slot leaks into obs_{other_name}")
        for k, ob in no_slot_obs.items():
            if sval in ob:
                errs.append(f"{it.item_id}: {name}_slot leaks into obs_{k}")
        if sval not in obs_map[name]:
            errs.append(f"{it.item_id}: {name}_slot missing from its own obs")

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
    print(f"constraint: {it.constraint_text}")
    print(f"trap={it.trap_slot}  partial={it.partial_slot}  "
          f"edge={it.edge_slot}  joint={it.joint_slot}")
    for tag, obs in [("N0", it.obs_N0), ("T0", it.obs_T0),
                     ("T_partial", it.obs_T_partial),
                     ("T_edge", it.obs_T_edge),
                     ("IC", it.obs_IC), ("S0", it.obs_S0)]:
        print(f"[{tag}] (len={len(obs)}) {obs}")
