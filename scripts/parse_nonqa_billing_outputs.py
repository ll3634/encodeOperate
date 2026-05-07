#!/usr/bin/env python3
"""
Parser and amount normalizer for the calculator-assisted billing-verification
sanity check.

Allowed model output formats:
  Action: Search[short query]
  Final: AMOUNT=<amount>

Parsing rules (per spec):
  - Strip whitespace.
  - Use the first non-empty line.
  - If line starts with "Action:" and contains "Search[", first_is_search=1.
  - If line starts with "Final:" and contains "AMOUNT=", first_is_final=1
    and extract the amount after AMOUNT=.
  - Save raw output always.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


_AMOUNT_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize_amount(raw: str, normalize_currency: bool = True) -> Optional[str]:
    """
    Normalize an amount string for comparison.

    Steps:
      - strip whitespace
      - remove thousands-separator commas
      - if normalize_currency=True, strip leading currency sign for comparison
        (so "$128.40" and "128.40" both → "128.40")
      - keep the numeric value with two decimal places when possible
      - if no numeric content is found, return None
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    s = s.replace(",", "")
    sign = ""
    if s.startswith("-"):
        sign = "-"
        s = s[1:].strip()
    leading_currency = ""
    if s and s[0] in "$£€¥":
        leading_currency = s[0]
        s = s[1:].strip()
    m = _AMOUNT_NUM_RE.search(s)
    if not m:
        return None
    num_str = m.group(0).replace(",", "")
    try:
        val = float(num_str)
    except ValueError:
        return None
    norm = f"{sign}{val:.2f}"
    if not normalize_currency and leading_currency:
        norm = f"{leading_currency}{norm}"
    return norm


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
    final_amount: Optional[str] = None
    search_query_text: Optional[str] = None
    commit_W: int = 0

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "first_line": self.first_line,
            "first_is_search": self.first_is_search,
            "first_is_final": self.first_is_final,
            "parse_failure": self.parse_failure,
            "final_amount": self.final_amount,
            "search_query_text": self.search_query_text,
            "commit_W": self.commit_W,
        }


def parse_output(
    raw_text: str,
    target_W_str: str,
    normalize_currency: bool = True,
) -> ParsedOutput:
    """
    Parse one model output line and decide commit_W vs first_is_search vs
    parse_failure.

    target_W_str is the exact expected W string (e.g. "$128.40"); we compare
    using normalize_amount() on both sides.
    """
    line = first_nonempty_line(raw_text)
    out = ParsedOutput(raw=raw_text or "", first_line=line)

    if not line:
        out.parse_failure = 1
        return out

    # Search format. Accept either "Action: Search[...]" or bare "Search[...]".
    m_q = re.search(r"Search\s*\[(.*?)\]", line, re.IGNORECASE)
    if m_q:
        head = line[:m_q.start()].strip().rstrip(":").strip()
        if head == "" or re.fullmatch(r"Action\s*:?", head, re.IGNORECASE):
            out.first_is_search = 1
            out.search_query_text = m_q.group(1).strip()
            return out

    # Final format. Accept either "Final: AMOUNT=..." or bare "AMOUNT=...".
    m_amt = re.search(r"AMOUNT\s*=\s*(\S+)", line, re.IGNORECASE)
    if m_amt:
        head = line[:m_amt.start()].strip().rstrip(":").strip()
        if head == "" or re.fullmatch(r"Final\s*:?", head, re.IGNORECASE):
            out.first_is_final = 1
            amt_raw = m_amt.group(1).rstrip(".,;:")
            norm_pred = normalize_amount(amt_raw, normalize_currency=normalize_currency)
            norm_target = normalize_amount(target_W_str, normalize_currency=normalize_currency)
            out.final_amount = norm_pred
            if norm_pred is not None and norm_pred == norm_target:
                out.commit_W = 1
            return out

    out.parse_failure = 1
    return out
