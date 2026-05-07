#!/usr/bin/env python3
"""
Parser for the multi-turn ReAct SQL data-analysis surface.

Mirrors parse_nonqa_react_codesearch_outputs.py but matches entity name spans
in result tables instead of file paths. Per-generation fields:

  first_is_action          : 1 if first non-empty line is "Action: <tool>" or
                             a bare "execute_query[...]" call
  first_is_final           : 1 if model commits at p0 (Final Answer or any
                             non-action commit text)
  parse_failure            : 1 if no first non-empty line at all
  commit_W                 : 1 if first_is_final AND final body contains W
                             (the trap entity in T0/N0/IC; the correct entity
                             in S0)
  commit_correct           : 1 if first_is_final AND final body contains the
                             correct (true-answer) entity
  final_present            : 1 if "Final Answer:" appears anywhere in raw
  commit_W_anywhere        : 1 if final_present AND final body contains W
  commit_correct_anywhere  : 1 if final_present AND final body contains the
                             correct entity
  hallucinated_observation : 1 if model emitted its own "Observation:" line
  first_line               : first non-empty line (truncated)
  final_text               : Final Answer body (truncated)
  action_tool              : tool name parsed from first Action line (if any)
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
    commit_correct: int = 0
    final_present: int = 0
    commit_W_anywhere: int = 0
    commit_correct_anywhere: int = 0
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


_FINAL_RE = re.compile(r"^\s*(?:final\s*answer|final)\s*[:\-]\s*(.*)$", re.IGNORECASE)
_ACTION_RE = re.compile(r"^\s*action\s*[:\-]\s*(\S+).*$", re.IGNORECASE)
_OBS_RE = re.compile(r"^\s*observation\s*[:\-]", re.IGNORECASE)
# Bare tool call without "Action:" prefix, e.g., execute_query[SELECT ...]
_BARE_TOOL_RE = re.compile(r"^\s*([a-z_][a-z0-9_]*)\s*\[", re.IGNORECASE)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _entity_in_text(text: str, entity: str) -> bool:
    """Case-insensitive, whitespace-collapsed substring match."""
    if not text or not entity:
        return False
    return _normalize(entity) in _normalize(text)


def _extract_final_body(raw_text: str) -> Optional[str]:
    """Return the first 'Final Answer:' body found anywhere in raw_text.

    Body extends until a new Action/Final/Observation line or end of text,
    capturing up to 12 follow-on lines.
    """
    lines = (raw_text or "").splitlines()
    for i, line in enumerate(lines):
        m = _FINAL_RE.match(line.rstrip())
        if not m:
            continue
        body_lines = [m.group(1)]
        for j in range(i + 1, min(len(lines), i + 12)):
            s = lines[j].rstrip()
            if _ACTION_RE.match(s) or _FINAL_RE.match(s) or _OBS_RE.match(s):
                break
            body_lines.append(s)
        return " ".join(body_lines).strip()
    return None


def parse_output(raw_text: str, W_entity: str,
                 correct_entity: Optional[str] = None) -> ParsedOutput:
    """Decide commit vs continue at p0 and check entity-name commits.

    The user prompt asks for an entity name, so the model often emits a bare
    sentence rather than a 'Final Answer:' prefix; we treat any first line
    that is not an Action call (or bare tool call) as a commit, and check
    whether the body contains W and/or the correct entity.
    """
    out = ParsedOutput(raw=raw_text or "", first_line=first_nonempty_line(raw_text))

    final_body = _extract_final_body(raw_text or "")
    if final_body is not None:
        out.final_present = 1
        out.final_text = final_body[:300]
        if _entity_in_text(final_body, W_entity):
            out.commit_W_anywhere = 1
        if correct_entity and _entity_in_text(final_body, correct_entity):
            out.commit_correct_anywhere = 1

    for line in (raw_text or "").splitlines():
        if _OBS_RE.match(line):
            out.hallucinated_observation = 1
            break

    if not out.first_line:
        out.parse_failure = 1
        return out

    m_act = _ACTION_RE.match(out.first_line)
    if m_act:
        out.first_is_action = 1
        out.action_tool = m_act.group(1).strip().rstrip(":").lower()
        return out
    m_bare = _BARE_TOOL_RE.match(out.first_line)
    if m_bare and not _FINAL_RE.match(out.first_line):
        out.first_is_action = 1
        out.action_tool = m_bare.group(1).strip().lower()
        return out

    out.first_is_final = 1
    body_for_match = final_body if final_body is not None else (raw_text or "")
    if final_body is None:
        out.final_text = (raw_text or "").strip()[:300]
    if _entity_in_text(body_for_match, W_entity):
        out.commit_W = 1
        out.commit_W_anywhere = 1
    if correct_entity and _entity_in_text(body_for_match, correct_entity):
        out.commit_correct = 1
        out.commit_correct_anywhere = 1
    return out
