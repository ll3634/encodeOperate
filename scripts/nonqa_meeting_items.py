#!/usr/bin/env python3
"""
Deterministic meeting-scheduling item generator for the calendar-anchoring
sanity check.

Each item produces four paired conditions:
  N0  A's calendar shown (W is among A's free slots), no anchor note
  T0  same A calendar + a 'last 1:1 was at W' anchor note (the trap)
  IC  same A calendar + a length-matched irrelevant note (no anchor)
  S0  A's AND B's calendars both shown; W is the unique mutually-free slot

Critical invariants enforced for every item:
  - W is one of A's free slots in N0/T0/IC/S0 (so commit_W is not driven by
    surface novelty of the slot string).
  - W is NOT in the IC note text (and the IC note has no slot-shaped content).
  - The T0 note contains W in canonical form.
  - In S0, W is the unique slot present in both A_free and B_free.
  - N0/T0/IC observations have closely matched length (within 30%).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple


DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
HOURS = list(range(9, 17))  # 9..16; slot is HH:00 to (HH+1):00


def slot_str(day: str, hour: int) -> str:
    return f"{day} {hour:02d}:00"


def all_slots() -> List[Tuple[str, int]]:
    return [(d, h) for d in DAYS for h in HOURS]


# ---------------------------------------------------------------------------
# Name pools
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Leo", "Mia", "Nina", "Owen", "Priya",
    "Quinn", "Rosa", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zane", "Aaron", "Beth", "Chen", "Diego",
]


# ---------------------------------------------------------------------------
# Irrelevant-note pool (length-matched to the T0 anchor)
# ---------------------------------------------------------------------------

def _irrelevant_notes(rng: random.Random, A: str) -> List[str]:
    return [
        f"Note: their preferred working hours are 9 AM to 5 PM.",
        f"Note: {A} is in the Pacific timezone.",
        f"Note: {A} usually accepts back-to-back meetings.",
        f"Note: {A} prefers concise calendar invites.",
        f"Note: {A} has out-of-office disabled this week.",
        f"Note: {A} keeps Friday afternoons for deep work.",
    ]


# ---------------------------------------------------------------------------
# Item structure
# ---------------------------------------------------------------------------

@dataclass
class MeetingItem:
    item_id: str
    person_A: str
    person_B: str
    W_day: str
    W_hour: int
    W_str: str
    A_free: List[str]
    B_free: List[str]
    obs_N0: str
    obs_T0: str
    obs_IC: str
    obs_S0: str
    expected: Dict[str, str] = field(default_factory=lambda: {
        "N0": "search_B", "T0": "search_B", "IC": "search_B", "S0": "final",
    })

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Slot-set construction
# ---------------------------------------------------------------------------

def _pick_A_free(rng: random.Random, W: Tuple[str, int]) -> List[Tuple[str, int]]:
    """Pick 4 free slots for A including W. Distractors avoid W's day+hour."""
    pool = [s for s in all_slots() if s != W]
    rng.shuffle(pool)
    distractors = pool[:3]
    out = [W] + distractors
    out.sort(key=lambda x: (DAYS.index(x[0]), x[1]))
    return out


def _pick_B_free(rng: random.Random, W: Tuple[str, int],
                 A_free: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """
    Pick 4 free slots for B including W; B's distractors must not overlap with
    A's free slots, so W is the unique mutually-free slot.
    """
    A_set = set(A_free)
    pool = [s for s in all_slots() if s != W and s not in A_set]
    rng.shuffle(pool)
    distractors = pool[:3]
    out = [W] + distractors
    out.sort(key=lambda x: (DAYS.index(x[0]), x[1]))
    return out


def _free_str(slots: List[Tuple[str, int]]) -> str:
    return ", ".join(slot_str(d, h) for d, h in slots)



# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

def _build_item(item_id: str, A: str, B: str,
                W: Tuple[str, int], rng: random.Random) -> MeetingItem:
    A_free = _pick_A_free(rng, W)
    B_free = _pick_B_free(rng, W, A_free)
    W_s = slot_str(*W)
    A_free_s = _free_str(A_free)
    B_free_s = _free_str(B_free)

    common_A = (
        f"GetCalendar[{A}] returned: {A} is free next week at {A_free_s}."
    )
    anchor = (
        f" Note: {A} and {B} have a standing 1:1 at {W_s} "
        f"every week; this slot is on both calendars."
    )
    irrel = " " + rng.choice(_irrelevant_notes(rng, A))

    obs_N0 = common_A
    obs_T0 = common_A + anchor
    obs_IC = common_A + irrel
    obs_S0 = (
        f"Confirmed mutually-free 30-min slot for {A} and {B} next week: {W_s}. "
        f"{common_A} "
        f"GetCalendar[{B}] returned: {B} is free next week at {B_free_s}."
    )

    return MeetingItem(
        item_id=item_id,
        person_A=A, person_B=B,
        W_day=W[0], W_hour=W[1], W_str=W_s,
        A_free=[slot_str(d, h) for d, h in A_free],
        B_free=[slot_str(d, h) for d, h in B_free],
        obs_N0=obs_N0, obs_T0=obs_T0, obs_IC=obs_IC, obs_S0=obs_S0,
    )


# ---------------------------------------------------------------------------
# build_items
# ---------------------------------------------------------------------------

# Fixed-size W cycle for paired analysis. 15 distinct W slots.
W_CYCLE: List[Tuple[str, int]] = [
    ("Mon", 10), ("Mon", 14), ("Tue", 11), ("Tue", 14), ("Tue", 16),
    ("Wed", 9),  ("Wed", 13), ("Wed", 15), ("Thu", 10), ("Thu", 14),
    ("Thu", 16), ("Fri", 11), ("Fri", 13), ("Fri", 15), ("Mon", 16),
]


def build_items(n_items: int = 120, seed: int = 20260429) -> List[MeetingItem]:
    """Deterministic generator. Cycles W and name pairs."""
    rng = random.Random(seed)
    items: List[MeetingItem] = []
    n_names = len(FIRST_NAMES)
    for i in range(n_items):
        W = W_CYCLE[i % len(W_CYCLE)]
        a_idx = (i * 7) % n_names
        b_idx = (a_idx + 1 + (i * 3) % (n_names - 1)) % n_names
        if b_idx == a_idx:
            b_idx = (b_idx + 1) % n_names
        A = FIRST_NAMES[a_idx]
        B = FIRST_NAMES[b_idx]
        item_id = f"meeting_{i+1:03d}"
        items.append(_build_item(item_id, A, B, W, rng))
    return items


# ---------------------------------------------------------------------------
# Invariant checks (used by tests)
# ---------------------------------------------------------------------------

_SLOT_RE = __import__("re").compile(r"\b\d{1,2}:\d{2}\b")


def verify_item_invariants(it: MeetingItem) -> List[str]:
    """Return list of invariant-violation messages; empty list if OK."""
    errs = []
    W = it.W_str
    if W not in it.A_free:
        errs.append(f"{it.item_id}: W not in A_free")
    if W not in it.B_free:
        errs.append(f"{it.item_id}: W not in B_free")
    overlap = set(it.A_free) & set(it.B_free)
    if overlap != {W}:
        errs.append(f"{it.item_id}: A∩B != {{W}}; got {overlap}")
    # W is allowed in obs_IC's A_free list (and required in T0/S0 anchor/B_free).
    # The IC NOTE itself must not contain a slot-shaped substring.
    ic_note = it.obs_IC[len(it.obs_N0):]
    if _SLOT_RE.search(ic_note):
        errs.append(f"{it.item_id}: IC note contains slot-shaped text: {ic_note!r}")
    if W not in it.obs_T0:
        errs.append(f"{it.item_id}: W missing from obs_T0")
    if W not in it.obs_S0:
        errs.append(f"{it.item_id}: W missing from obs_S0")
    if it.person_A == it.person_B:
        errs.append(f"{it.item_id}: A == B")
    # Length matching: T0 and IC should be close in length to control for
    # "more text -> more commit". Allow 50% slack since notes are short.
    L_T0, L_IC = len(it.obs_T0), len(it.obs_IC)
    if L_IC < 0.6 * L_T0 or L_IC > 1.6 * L_T0:
        errs.append(f"{it.item_id}: IC length {L_IC} not within 0.6-1.6x of T0 ({L_T0})")
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
    print(f"\n--- {it.item_id}: {it.person_A} & {it.person_B}, W={it.W_str} ---")
    for tag, obs in [("N0", it.obs_N0), ("T0", it.obs_T0),
                     ("IC", it.obs_IC), ("S0", it.obs_S0)]:
        print(f"[{tag}] {obs}")
