#!/usr/bin/env python3
"""
Parser for the multi-turn ReAct meeting-scheduling sanity check.

For each generation we emit:
  first_is_action          : 1 if the first non-empty line is "Action: calendar..."
                             (model continued the ReAct loop at p0)
  first_is_final           : 1 if the first non-empty line is "Final Answer: ..."
                             (model committed at p0)
  parse_failure            : 1 if neither first-line rule matched
  commit_W                 : (legacy) 1 if first_is_final AND the final body
                             contains W. Kept for back-compat.
  final_present            : 1 if "Final Answer:" appears ANYWHERE in raw
  commit_W_anywhere        : 1 if final_present AND the final body contains W
                             (premature in N0/T0/IC, correct in S0). Captures
                             the agent-pathology trajectory: model issues a
                             second Action, hallucinates its Observation in
                             the same generation, and commits to W.
  hallucinated_observation : 1 if an "Observation:" line appears AFTER the
                             prefilled Action (i.e., model faked a tool call).
  first_line               : the first non-empty line (truncated, for diagnostics)
  final_text               : Final Answer body if any (truncated)
  action_tool              : tool name parsed from first Action line if any
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ParsedOutput:
    raw: str = ""
    first_line: str = ""
    first_is_action: int = 0
    first_is_final: int = 0
    parse_failure: int = 0
    commit_W: int = 0
    final_present: int = 0
    commit_W_anywhere: int = 0
    hallucinated_observation: int = 0
    final_text: Optional[str] = None
    action_tool: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def first_nonempty_line(text: str) -> str:
    if not text:
        return ""
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s
    return ""


# Match "1:00pm-1:30pm", "1:00 pm - 1:30 pm", "01:00 PM to 01:30 PM",
# "13:00-13:30", "1pm-1:30pm", etc.
_SLOT_RE = re.compile(
    r"(\d{1,2})(?::(\d{2}))?\s*(am|pm|AM|PM)?\s*(?:-|to|\u2013|\u2014|until)\s*"
    r"(\d{1,2})(?::(\d{2}))?\s*(am|pm|AM|PM)?",
    re.IGNORECASE,
)


def _to_24h(h: int, m: int, ampm: Optional[str]) -> Optional[int]:
    if ampm:
        ap = ampm.lower()
        if ap == "pm" and h != 12:
            h += 12
        elif ap == "am" and h == 12:
            h = 0
    if h < 0 or h > 23 or m < 0 or m > 59:
        return None
    return h * 60 + m


def normalize_slot(s: str) -> Optional[str]:
    """Return canonical 'HHMM-HHMM' (24h) for any recognized slot string."""
    m = _SLOT_RE.search(s)
    if not m:
        return None
    h1 = int(m.group(1))
    mn1 = int(m.group(2) or 0)
    ap1 = m.group(3)
    h2 = int(m.group(4))
    mn2 = int(m.group(5) or 0)
    ap2 = m.group(6) or ap1  # if second has no ampm, inherit from first
    # If first lacks ampm but second has one (e.g. "1-2pm"), inherit backwards.
    if ap1 is None and ap2 is not None:
        ap1 = ap2
    t1 = _to_24h(h1, mn1, ap1)
    t2 = _to_24h(h2, mn2, ap2)
    if t1 is None or t2 is None:
        return None
    return f"{t1//60:02d}{t1%60:02d}-{t2//60:02d}{t2%60:02d}"


def _slot_matches(text: str, target_slot: str) -> bool:
    target_norm = normalize_slot(target_slot)
    if target_norm is None:
        return False
    for m in _SLOT_RE.finditer(text):
        cand = m.group(0)
        if normalize_slot(cand) == target_norm:
            return True
    return False


_FINAL_RE = re.compile(r"^\s*(?:final\s*answer|final)\s*[:\-]\s*(.*)$", re.IGNORECASE)
_ACTION_RE = re.compile(r"^\s*action\s*[:\-]\s*(\S+).*$", re.IGNORECASE)
_OBS_RE = re.compile(r"^\s*observation\s*[:\-]", re.IGNORECASE)


def _extract_final_body(raw_text: str) -> Optional[str]:
    """Return the first 'Final Answer:' body found anywhere in raw_text."""
    lines = (raw_text or "").splitlines()
    for i, line in enumerate(lines):
        m = _FINAL_RE.match(line.rstrip())
        if not m:
            continue
        body_lines = [m.group(1)]
        for j in range(i + 1, min(len(lines), i + 8)):
            s = lines[j].rstrip()
            if _ACTION_RE.match(s) or _FINAL_RE.match(s) or _OBS_RE.match(s):
                break
            body_lines.append(s)
        return " ".join(body_lines).strip()
    return None


def parse_output(raw_text: str, target_W_slot: str) -> ParsedOutput:
    """Parse one generation; decide first_is_action / first_is_final / commit_W
    plus the agent-pathology metrics commit_W_anywhere and
    hallucinated_observation."""
    out = ParsedOutput(raw=raw_text or "", first_line=first_nonempty_line(raw_text))

    # Whole-trajectory metrics first (independent of first_line).
    final_body = _extract_final_body(raw_text or "")
    if final_body is not None:
        out.final_present = 1
        out.final_text = final_body[:200]
        if _slot_matches(final_body, target_W_slot):
            out.commit_W_anywhere = 1
    # Hallucinated tool observation: an Observation: line in the model output
    # (the prefilled Observation is in the prompt, not in raw_text).
    for line in (raw_text or "").splitlines():
        if _OBS_RE.match(line):
            out.hallucinated_observation = 1
            break

    if not out.first_line:
        out.parse_failure = 1
        return out

    m_fin = _FINAL_RE.match(out.first_line)
    if m_fin:
        out.first_is_final = 1
        # Legacy commit_W: only counts when first line is Final Answer.
        if final_body is not None and _slot_matches(final_body, target_W_slot):
            out.commit_W = 1
        return out

    m_act = _ACTION_RE.match(out.first_line)
    if m_act:
        out.first_is_action = 1
        out.action_tool = m_act.group(1).strip().rstrip(":").lower()
        return out

    out.parse_failure = 1
    return out
