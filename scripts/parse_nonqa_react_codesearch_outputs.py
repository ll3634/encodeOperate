#!/usr/bin/env python3
"""
Parser for the multi-turn ReAct code-search canonical-implementation surface.

Mirrors parse_nonqa_react_meeting_outputs.py but matches `path:line` spans
instead of time slots. Per-generation fields:

  first_is_action          : 1 if first non-empty line is "Action: <tool>"
  first_is_final           : 1 if first non-empty line is "Final Answer: ..."
  parse_failure            : 1 if neither rule matched
  commit_W                 : 1 if first_is_final AND final body contains W path
                             (W = legacy_path in T0; canonical_path in S0)
  final_present            : 1 if "Final Answer:" appears anywhere in raw
  commit_W_anywhere        : 1 if final_present AND final body contains W path
  hallucinated_observation : 1 if model emitted its own "Observation:" line
                             (faked tool output)
  first_line               : first non-empty line (truncated, for diagnostics)
  final_text               : Final Answer body (truncated)
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


_FINAL_RE = re.compile(r"^\s*(?:final\s*answer|final)\s*[:\-]\s*(.*)$", re.IGNORECASE)
_ACTION_RE = re.compile(r"^\s*action\s*[:\-]\s*(\S+).*$", re.IGNORECASE)
_OBS_RE = re.compile(r"^\s*observation\s*[:\-]", re.IGNORECASE)


def _normalize_path(s: str) -> str:
    """Collapse whitespace and lowercase for substring comparison."""
    return re.sub(r"\s+", " ", s).strip().lower()


def _path_matches(text: str, target_path: str) -> bool:
    """Match against target. Accepts either the full 'path:line' or path-only.

    Strict: requires the target's full 'path:line' substring (case-insensitive,
    whitespace-collapsed).
    """
    if not text or not target_path:
        return False
    return _normalize_path(target_path) in _normalize_path(text)


def _extract_final_body(raw_text: str) -> Optional[str]:
    """Return the first 'Final Answer:' body found anywhere in raw_text.

    Body extends until a new Action/Final/Observation line, or end of text,
    whichever comes first. We capture up to 12 follow-on lines.
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


_PATHISH_RE = re.compile(r"[A-Za-z0-9_./-]+\.py(?::\d+)?")


def parse_output(raw_text: str, target_W_path: str) -> ParsedOutput:
    """Decide commit vs continue at p0. The user prompt asks for a path:line,
    so the model often emits a bare path without a 'Final Answer:' prefix; we
    treat any first-line that does not start with 'Action:' as a commit, and
    check whether the body contains W (target path)."""
    out = ParsedOutput(raw=raw_text or "", first_line=first_nonempty_line(raw_text))

    final_body = _extract_final_body(raw_text or "")
    if final_body is not None:
        out.final_present = 1
        out.final_text = final_body[:300]

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

    # Anything else at p0 counts as a commit. Use the explicit Final Answer
    # body if present; otherwise treat the whole raw text as the commit body.
    out.first_is_final = 1
    body_for_match = final_body if final_body is not None else (raw_text or "")
    if final_body is None and _PATHISH_RE.search(raw_text or ""):
        out.final_present = 1
        out.final_text = (raw_text or "").strip()[:300]
    if _path_matches(body_for_match, target_W_path):
        out.commit_W = 1
        out.commit_W_anywhere = 1
    return out
