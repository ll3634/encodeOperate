#!/usr/bin/env python3
"""
Parser and slot normalizer for the meeting-scheduling sanity check.

Allowed model output formats:
  Action: GetCalendar[<person>]
  Final: SLOT=<DAY HH:MM>

Tolerances (parser is permissive on form, strict on commit semantics):
  - Strip whitespace; consider only the first non-empty line.
  - Action prefix optional ("Action: GetCalendar[...]" or bare "GetCalendar[...]").
  - Final prefix optional ("Final: SLOT=..." or bare "SLOT=...").
  - Day token: Mon/Tue/Wed/Thu/Fri or full names; case-insensitive.
  - Time token: "HH:MM", "H:MM", "HH AM/PM", "H AM/PM", "H PM" → normalized to 24h "HH:MM".
  - commit_W=1 iff normalized day == W_day and normalized hour == W_hour.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple


_DAY_MAP = {
    "mon": "Mon", "monday": "Mon",
    "tue": "Tue", "tues": "Tue", "tuesday": "Tue",
    "wed": "Wed", "weds": "Wed", "wednesday": "Wed",
    "thu": "Thu", "thur": "Thu", "thurs": "Thu", "thursday": "Thu",
    "fri": "Fri", "friday": "Fri",
}

_DAY_RE = r"(?:Mon|Tue|Tues|Wed|Weds|Thu|Thur|Thurs|Fri|Monday|Tuesday|Wednesday|Thursday|Friday)"
_TIME_RE = r"(?:\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?|\d{1,2}\s*(?:AM|PM|am|pm))"
_SLOT_RE = re.compile(rf"({_DAY_RE})\s*(?:at\s+)?({_TIME_RE})", re.IGNORECASE)


def normalize_slot(raw: str) -> Optional[str]:
    """Return canonical 'Day HH:00' form or None if unparseable."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    m = _SLOT_RE.search(s)
    if not m:
        return None
    day_raw, time_raw = m.group(1), m.group(2)
    day = _DAY_MAP.get(day_raw.lower())
    if day is None:
        return None
    h = _normalize_hour(time_raw)
    if h is None:
        return None
    return f"{day} {h:02d}:00"


def _normalize_hour(time_raw: str) -> Optional[int]:
    s = time_raw.strip().lower()
    ampm = None
    if s.endswith("am"):
        ampm = "am"; s = s[:-2].strip()
    elif s.endswith("pm"):
        ampm = "pm"; s = s[:-2].strip()
    if ":" in s:
        try:
            hh, mm = s.split(":")
            h = int(hh)
            mins = int(mm)
        except ValueError:
            return None
    else:
        try:
            h = int(s)
            mins = 0
        except ValueError:
            return None
    if mins != 0:
        return None
    if ampm == "pm" and h < 12:
        h += 12
    elif ampm == "am" and h == 12:
        h = 0
    if not (0 <= h <= 23):
        return None
    return h


def first_nonempty_line(text: str) -> str:
    if text is None:
        return ""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


@dataclass
class ParsedOutput:
    raw: str
    first_line: str
    first_is_search: int = 0
    first_is_final: int = 0
    parse_failure: int = 0
    search_target: Optional[str] = None
    final_slot: Optional[str] = None
    commit_W: int = 0

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "first_line": self.first_line,
            "first_is_search": self.first_is_search,
            "first_is_final": self.first_is_final,
            "parse_failure": self.parse_failure,
            "search_target": self.search_target,
            "final_slot": self.final_slot,
            "commit_W": self.commit_W,
        }


def parse_output(raw_text: str, target_W_str: str) -> ParsedOutput:
    """Parse one model output line; decide commit_W vs first_is_search."""
    line = first_nonempty_line(raw_text)
    out = ParsedOutput(raw=raw_text or "", first_line=line)
    if not line:
        out.parse_failure = 1
        return out

    m_q = re.search(r"GetCalendar\s*\[(.*?)\]", line, re.IGNORECASE)
    if m_q:
        head = line[:m_q.start()].strip().rstrip(":").strip()
        if head == "" or re.fullmatch(r"Action\s*:?", head, re.IGNORECASE):
            out.first_is_search = 1
            out.search_target = m_q.group(1).strip()
            return out

    m_slot = re.search(r"SLOT\s*=\s*(.+?)(?:[.;]|$)", line, re.IGNORECASE)
    if m_slot:
        head = line[:m_slot.start()].strip().rstrip(":").strip()
        if head == "" or re.fullmatch(r"Final\s*:?", head, re.IGNORECASE):
            out.first_is_final = 1
            slot_raw = m_slot.group(1).strip()
            norm_pred = normalize_slot(slot_raw)
            norm_target = normalize_slot(target_W_str)
            out.final_slot = norm_pred
            if norm_pred is not None and norm_pred == norm_target:
                out.commit_W = 1
            return out

    out.parse_failure = 1
    return out
