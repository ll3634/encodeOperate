#!/usr/bin/env python3
"""
Answer scoring functions for evaluation.
"""

import re
import string
from fractions import Fraction
from typing import Optional, List, Union


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    
    # Convert to lowercase
    answer = answer.lower()
    
    # Remove articles
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
    
    # Remove punctuation
    answer = answer.translate(str.maketrans('', '', string.punctuation))
    
    # Normalize whitespace
    answer = ' '.join(answer.split())
    
    return answer.strip()


def exact_match(prediction: str, gold: str) -> bool:
    """Check if prediction exactly matches gold after normalization."""
    return normalize_answer(prediction) == normalize_answer(gold)


def fuzzy_match(prediction: str, gold: str, threshold: float = 0.8) -> bool:
    """
    Check if prediction fuzzy-matches gold.
    Uses simple token overlap ratio.
    """
    pred_tokens = set(normalize_answer(prediction).split())
    gold_tokens = set(normalize_answer(gold).split())
    
    if not gold_tokens:
        return not pred_tokens
    
    if not pred_tokens:
        return False
    
    # Compute Jaccard similarity
    intersection = len(pred_tokens & gold_tokens)
    union = len(pred_tokens | gold_tokens)
    
    if union == 0:
        return True
    
    similarity = intersection / union
    return similarity >= threshold


def contains_answer(prediction: str, gold: str) -> bool:
    """Check if prediction contains the gold answer."""
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)

    return gold_norm in pred_norm


# ---------------------------------------------------------------------------
# Robust numeric extraction & comparison (for GSM8K / MATH numeric answers)
# ---------------------------------------------------------------------------

# Matches numbers with optional comma-separators or plain numbers (incl. decimals)
_RE_NUMBER = re.compile(r'-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?')


def extract_last_number(text: str) -> Optional[float]:
    """Extract the last numeric value from *text*.

    Handles plain integers (``15``), decimals (``3.14``), negatives (``-3``),
    and thousands-separated values (``1,520``).  Returns ``None`` when no
    number is found.
    """
    if not text:
        return None
    matches = _RE_NUMBER.findall(text)
    if not matches:
        return None
    last = matches[-1].replace(",", "")
    try:
        return float(last)
    except ValueError:
        return None


def numeric_match(prediction: str, gold: str, rel_tol: float = 1e-5) -> bool:
    """Compare the last number in *prediction* against the gold answer.

    Both sides are parsed via :func:`extract_last_number`.  Two numbers are
    considered equal when they are exactly equal **or** within *rel_tol*
    relative tolerance (useful for floating-point answers).
    """
    pred_num = extract_last_number(prediction)
    gold_num = extract_last_number(gold)
    if pred_num is None or gold_num is None:
        return False
    # Exact check first (avoids float edge-cases)
    if pred_num == gold_num:
        return True
    # Relative tolerance
    if gold_num == 0:
        return abs(pred_num) < rel_tol
    return abs(pred_num - gold_num) / abs(gold_num) < rel_tol


# ---------------------------------------------------------------------------
# Math-friendly equivalence (for MATH / GSM8K style numeric + LaTeX answers)
# ---------------------------------------------------------------------------


def _extract_last_boxed(text: str) -> Optional[str]:
    """Extract the content of the last \boxed{...} or \fbox{...} block.

    Best-effort brace parsing (handles nested braces).
    """
    if not text:
        return None
    for marker in ("\\boxed", "\\fbox"):
        idx = text.rfind(marker)
        if idx != -1:
            # Find the first '{' after marker
            j = text.find("{", idx)
            if j == -1:
                continue
            depth = 0
            for k in range(j, len(text)):
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[j + 1 : k]
    return None


_RE_LATEX_FRAC = re.compile(r"\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}")
_RE_LATEX_TEXT = re.compile(r"\\text\{([^{}]*)\}")
_RE_LATEX_SQRT = re.compile(r"\\sqrt\{([^{}]+)\}")


def normalize_math_answer(ans: str) -> str:
    """Normalize math-ish answers without deleting semantically important symbols."""
    if ans is None:
        return ""
    s = str(ans).strip()

    # Prefer boxed content if present
    boxed = _extract_last_boxed(s)
    if boxed is not None:
        s = boxed

    # Remove common wrappers / latex spacing
    s = s.replace("$", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "").replace("\\:", "")

    # Keep \text{...} content
    s = _RE_LATEX_TEXT.sub(r"\1", s)

    # Convert a few latex constructs into plain-ish forms
    # Repeat until stable to handle multiple fractions.
    prev = None
    while prev != s:
        prev = s
        s = _RE_LATEX_FRAC.sub(r"(\1)/(\2)", s)
    s = _RE_LATEX_SQRT.sub(r"sqrt(\1)", s)
    s = s.replace("\\cdot", "*")
    s = s.replace("\\pi", "pi")

    # Drop braces but keep content
    s = s.replace("{", "").replace("}", "")

    # Collapse whitespace
    s = " ".join(s.split())
    return s.strip().lower()


def _try_parse_rational(s: str) -> Optional[Fraction]:
    """Parse simple integers/decimals/fractions into a Fraction if possible."""
    if not s:
        return None
    t = normalize_math_answer(s)

    # If there's an '=', keep the RHS (common 'x = ...')
    if "=" in t:
        t = t.split("=")[-1].strip()

    # Remove surrounding parentheses
    t = t.strip()
    if t.startswith("(") and t.endswith(")"):
        t = t[1:-1].strip()

    # Remove thousands separators
    t = t.replace(",", "")

    # Plain fraction a/b
    if re.fullmatch(r"-?[0-9]+\s*/\s*-?[0-9]+", t):
        a, b = [x.strip() for x in t.split("/")]
        try:
            return Fraction(int(a), int(b))
        except Exception:
            return None

    # Integer
    if re.fullmatch(r"-?[0-9]+", t):
        try:
            return Fraction(int(t), 1)
        except Exception:
            return None

    # Decimal
    if re.fullmatch(r"-?[0-9]*\.[0-9]+", t):
        try:
            return Fraction(t)
        except Exception:
            return None

    return None


def math_equiv(prediction: str, gold: str) -> bool:
    """Best-effort equivalence for math answers.

    - Handles \boxed{...}
    - Handles simple LaTeX fractions
    - Falls back to normalized string equality
    """
    p_num = _try_parse_rational(prediction)
    g_num = _try_parse_rational(gold)
    if p_num is not None and g_num is not None:
        return p_num == g_num

    return normalize_math_answer(prediction) == normalize_math_answer(gold)


def answer_scorer(
    prediction: str,
    gold: Union[str, List[str]],
    mode: str = "exact"
) -> dict:
    """
    Score a prediction against gold answer(s).
    
    Args:
        prediction: Model's prediction
        gold: Gold answer or list of acceptable answers
        mode: Scoring mode - "exact", "fuzzy", "contains", "any"
        
    Returns:
        Dict with score and match details.
        Supported modes: "exact", "fuzzy", "contains", "numeric", "any".
    """
    if isinstance(gold, str):
        gold_list = [gold]
    else:
        gold_list = gold
    
    if not gold_list:
        return {"score": 0.0, "matched": False, "matched_answer": None}
    
    for g in gold_list:
        if mode == "exact":
            if exact_match(prediction, g):
                return {"score": 1.0, "matched": True, "matched_answer": g}
        elif mode == "fuzzy":
            if fuzzy_match(prediction, g):
                return {"score": 1.0, "matched": True, "matched_answer": g}
        elif mode == "contains":
            if contains_answer(prediction, g):
                return {"score": 1.0, "matched": True, "matched_answer": g}
        elif mode == "numeric":
            # Strict numeric comparison – preferred for GSM8K / MATH
            if numeric_match(prediction, g):
                return {"score": 1.0, "matched": True, "matched_answer": g, "match_type": "numeric"}
            # Fallback to symbolic math_equiv (handles LaTeX fractions etc.)
            if math_equiv(prediction, g):
                return {"score": 0.9, "matched": True, "matched_answer": g, "match_type": "math_equiv"}
        elif mode == "any":
            # Try all modes
            if exact_match(prediction, g):
                return {"score": 1.0, "matched": True, "matched_answer": g, "match_type": "exact"}
            if contains_answer(prediction, g):
                return {"score": 0.8, "matched": True, "matched_answer": g, "match_type": "contains"}
            if fuzzy_match(prediction, g, threshold=0.6):
                return {"score": 0.6, "matched": True, "matched_answer": g, "match_type": "fuzzy"}
            if math_equiv(prediction, g):
                return {"score": 0.9, "matched": True, "matched_answer": g, "match_type": "math_equiv"}
    
    return {"score": 0.0, "matched": False, "matched_answer": None}


def compute_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    
    common = set(pred_tokens) & set(gold_tokens)
    
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


if __name__ == "__main__":
    # Self-test
    print("Scorer self-test:")
    
    tests = [
        ("Paris", "Paris", "exact", True),
        ("The capital is Paris", "Paris", "contains", True),
        ("Paris France", "Paris", "fuzzy", True),
        ("London", "Paris", "exact", False),
    ]
    
    for pred, gold, mode, expected in tests:
        result = answer_scorer(pred, gold, mode)
        status = "✓" if result["matched"] == expected else "✗"
        print(f"  {status} {mode}: '{pred}' vs '{gold}' -> {result['matched']}")

