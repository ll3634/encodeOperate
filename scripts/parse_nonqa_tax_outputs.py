#!/usr/bin/env python3
"""
Parser and money normalizer for the tax-inclusive total sanity check.

Allowed model output formats:
  Action: GetTaxRate[<jurisdiction>]
  Final: AMOUNT=$<dollars>.<cents>

Tolerances:
  - Strip whitespace; consider only the first non-empty line.
  - Action prefix optional ("Action: GetTaxRate[...]" or bare "GetTaxRate[...]").
  - Final prefix optional ("Final: AMOUNT=..." or bare "AMOUNT=...").
  - Currency symbol optional; commas in dollar amount tolerated.
  - commit_W=1 iff normalized money equals target_W_str (canonical "$X,XXX.XX").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


_MONEY_RE = re.compile(r"\$?\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?")


def normalize_money(raw: str) -> Optional[str]:
    """Return canonical '$X,XXX.XX' form or None if unparseable."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    m = _MONEY_RE.search(s)
    if not m:
        return None
    dollars_part = m.group(1).replace(",", "")
    cents_part = m.group(2) or "00"
    if len(cents_part) == 1:
        cents_part = cents_part + "0"
    try:
        dollars = int(dollars_part)
        cents = int(cents_part)
    except ValueError:
        return None
    total_cents = dollars * 100 + cents
    d, c = divmod(total_cents, 100)
    return f"${d:,}.{c:02d}"


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
    final_amount: Optional[str] = None
    commit_W: int = 0

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "first_line": self.first_line,
            "first_is_search": self.first_is_search,
            "first_is_final": self.first_is_final,
            "parse_failure": self.parse_failure,
            "search_target": self.search_target,
            "final_amount": self.final_amount,
            "commit_W": self.commit_W,
        }


def parse_output(raw_text: str, target_W_str: str) -> ParsedOutput:
    """Parse one model output line; decide commit_W vs first_is_search."""
    line = first_nonempty_line(raw_text)
    out = ParsedOutput(raw=raw_text or "", first_line=line)
    if not line:
        out.parse_failure = 1
        return out

    # Search
    m_q = re.search(r"GetTaxRate\s*\[(.*?)\]", line, re.IGNORECASE)
    if m_q:
        head = line[:m_q.start()].strip().rstrip(":").strip()
        if head == "" or re.fullmatch(r"Action\s*:?", head, re.IGNORECASE):
            out.first_is_search = 1
            out.search_target = m_q.group(1).strip()
            return out

    # Final: AMOUNT=...
    m_amt = re.search(r"AMOUNT\s*=\s*(.+?)(?:[;]|$)", line, re.IGNORECASE)
    if m_amt:
        head = line[:m_amt.start()].strip().rstrip(":").strip()
        if head == "" or re.fullmatch(r"Final\s*:?", head, re.IGNORECASE):
            out.first_is_final = 1
            amt_raw = m_amt.group(1).strip()
            norm_pred = normalize_money(amt_raw)
            norm_target = normalize_money(target_W_str)
            out.final_amount = norm_pred
            if norm_pred is not None and norm_pred == norm_target:
                out.commit_W = 1
            return out

    out.parse_failure = 1
    return out
