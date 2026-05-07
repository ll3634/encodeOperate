#!/usr/bin/env python3
"""Wording propagation across paper drafts. Reduced scope: paper/*.md only.
Auto-applies Rules A, C, D; flags Rules B, E, F. See task spec.
"""
from __future__ import annotations
import re, json, shutil, datetime
from pathlib import Path

WORK = Path("/home/featurize/work")
PAPER = WORK / "paper"
OUT = WORK / "results" / "wording_propagation"
OUT.mkdir(parents=True, exist_ok=True)
TS = datetime.date.today().strftime("%Y%m%d")
BACKUP = PAPER / f".backup_{TS}"
BACKUP.mkdir(exist_ok=True)

FILES = sorted([p for p in PAPER.glob("*.md")])

# ----- Pattern definitions -----
QUALIFIERS_A = re.compile(
    r"behavioral|Δ2sr|slope|low[- ]dose|pre[- ]saturation|activation",
    re.IGNORECASE)
PAT_A = re.compile(r"43\s*[×xX](?![0-9A-Za-z])")  # 43× or 43x as a token
REPL_A_FULL = "43× behavioral pre-saturation slope (Δ2sr per ρ; activation-level ratio at ρ=0.20 is 4.7×)"
REPL_A_COMPACT = "43× behavioral slope (4.7× activation)"
REPL_A_LATER = "43× slope"

PAT_B1 = re.compile(
    r"evidence[^\n]{0,80}(orthogonal|near-orthogonal)[^\n]{0,80}(inert|non-operative)",
    re.IGNORECASE)
PAT_B2 = re.compile(
    r"orthogonal[^\n]{0,80}evidence[^\n]{0,80}(inert|non-operative)",
    re.IGNORECASE)

PAT_C = re.compile(r"\bencoded but not operative\b", re.IGNORECASE)
APPEND_C = " — outside the operative subspace at L20"
# Logical-start prefix: whitespace, list markers, numbers, section/blockquote symbols
_C_PREFIX_RE = re.compile(r"^[\s\-\*\u2022>\u00a7\u00b71-9.\)\(]*$")

PAT_D = re.compile(
    r"signed[_ ]mean[_ ]?Δm|signed[_ ]mean[^\n]{0,20}Δm|\|Δm_flip\||\|Δm\|[^\n]{0,20}flip")
REPL_D = "mean per-prompt |Δm|"

PAT_E = re.compile(r"\bD2\b")
PAT_F = re.compile(r"0\.625|0\.643|D3[^\n]{0,30}(0\.6|0\.5)")

APPENDIX_FILES = {"07_t7_appendix_factory.md"}  # treat 07 as appendix; still scan but don't flag E/F


def split_sections(text):
    """Return list of (start_line_1based, end_line_1based_excl, body) by markdown headings."""
    lines = text.splitlines(keepends=True)
    bounds = [i for i, l in enumerate(lines) if re.match(r"^#{1,6}\s", l)]
    bounds.append(len(lines))
    if not bounds or bounds[0] != 0:
        bounds.insert(0, 0)
    secs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        secs.append((a + 1, b + 1, "".join(lines[a:b])))
    return secs


def split_paragraphs(text):
    """Yield (start_line_1b, end_line_1b_excl, body) per paragraph (blank-line separated)."""
    lines = text.splitlines(keepends=True)
    para = []; start = 1
    for i, l in enumerate(lines, start=1):
        if l.strip() == "":
            if para:
                yield (start, i, "".join(para)); para = []
            start = i + 1
        else:
            if not para:
                start = i
            para.append(l)
    if para:
        yield (start, len(lines) + 1, "".join(para))


def context_window(text, lineno, radius_paragraphs=2):
    """Return text covering ±radius_paragraphs around lineno."""
    paras = list(split_paragraphs(text))
    idx = next((k for k, (s, e, _) in enumerate(paras) if s <= lineno < e), None)
    if idx is None:
        return text
    lo = max(0, idx - radius_paragraphs)
    hi = min(len(paras), idx + radius_paragraphs + 1)
    return "".join(p[2] for p in paras[lo:hi])


def find_matches(pattern, text):
    out = []
    for i, line in enumerate(text.splitlines(), start=1):
        for m in pattern.finditer(line):
            out.append({"line": i, "col": m.start() + 1, "text": line.rstrip("\n"),
                        "match": m.group(0)})
    return out


def grep_all_files(pattern):
    rows = []
    for fp in FILES:
        text = fp.read_text()
        for m in find_matches(pattern, text):
            m["file"] = fp.name
            rows.append(m)
    return rows


def apply_rule_A(text):
    """First occurrence per section gets full form (unless qualifier already nearby);
    subsequent in same section get short '43× slope'."""
    secs = split_sections(text)
    if not secs:
        return text, []
    new_text = text
    diffs = []
    # We iterate sections, find matches, build replacements section by section
    out_chunks = []
    cursor = 0
    lines = text.splitlines(keepends=True)
    for s, e, body in secs:
        sec_text = "".join(lines[s - 1:e - 1])
        rebuilt, count, dlist = _apply_A_section(sec_text)
        if count > 0:
            for d in dlist:
                d["section_start_line"] = s
                diffs.append(d)
        out_chunks.append(rebuilt)
    new_text = "".join(out_chunks)
    return new_text, diffs


def _apply_A_section(sec_text):
    """Return (new_text, n_replacements, diff_list)."""
    diffs = []
    matches = list(PAT_A.finditer(sec_text))
    if not matches:
        return sec_text, 0, diffs
    out = []
    last = 0
    n_done = 0
    for m in matches:
        a, b = m.start(), m.end()
        # Window ±60 chars to detect qualifier
        ctx_l = sec_text[max(0, a - 60):a]
        ctx_r = sec_text[b:b + 60]
        if QUALIFIERS_A.search(ctx_l) or QUALIFIERS_A.search(ctx_r):
            out.append(sec_text[last:b])
            last = b
            continue
        if n_done == 0:
            # decide compact vs full by line length
            line_start = sec_text.rfind("\n", 0, a) + 1
            line_end = sec_text.find("\n", b)
            if line_end == -1:
                line_end = len(sec_text)
            line_len = line_end - line_start
            repl = REPL_A_COMPACT if line_len < 80 else REPL_A_FULL
        else:
            repl = REPL_A_LATER
        out.append(sec_text[last:a] + repl)
        diffs.append({"orig_match": m.group(0),
                      "replacement": repl,
                      "char_offset_in_section": a,
                      "line_excerpt": sec_text[max(0, a - 30):min(len(sec_text), b + 30)]})
        last = b
        n_done += 1
    out.append(sec_text[last:])
    return "".join(out), n_done, diffs


def _is_protected_C_context(body, match_start, match_end):
    """Skip Rule C if match is in a heading, bold-emphasized run, quoted identity tag,
    table cell, or 'label-style' position (logical start of line)."""
    # Find the line containing the match
    line_start = body.rfind("\n", 0, match_start) + 1
    line_end = body.find("\n", match_end)
    if line_end == -1:
        line_end = len(body)
    line = body[line_start:line_end]
    if re.match(r"^\s*#{1,6}\s", line):
        return True  # markdown heading
    if line.lstrip().startswith("|"):
        return True  # table row
    pre = body[line_start:match_start]
    # Bold: odd count of ** before match means inside bold run
    if pre.count("**") % 2 == 1:
        return True
    # Quoted identity tags: typographic or ASCII quotes wrapping the match on this line
    for ql, qr in [("\u201c", "\u201d"), ("\"", "\""), ("'", "'")]:
        ql_idx = line.rfind(ql, 0, match_start - line_start)
        qr_idx = line.find(qr, match_end - line_start)
        if ql_idx != -1 and qr_idx != -1 and ql_idx < (match_start - line_start) <= qr_idx:
            return True
    # Logical-start-of-line: only whitespace / list markers / section refs / blockquote
    # before the match → it's being used as a label, not a substantive claim.
    if _C_PREFIX_RE.match(pre):
        return True
    return False


def apply_rule_C(text):
    """Append marker on first occurrence per paragraph if 'operative subspace' not within ±2 paragraphs.
    Skip headings, bold runs, and quoted identity-tag contexts."""
    paras = list(split_paragraphs(text))
    diffs = []
    new_chunks = []
    for idx, (s, e, body) in enumerate(paras):
        ctx = context_window(text, s, radius_paragraphs=2)
        if "operative subspace" in ctx.lower():
            new_chunks.append(body)
            continue
        m = PAT_C.search(body)
        if not m:
            new_chunks.append(body)
            continue
        if _is_protected_C_context(body, m.start(), m.end()):
            new_chunks.append(body)
            continue
        a, b = m.start(), m.end()
        new_body = body[:b] + APPEND_C + body[b:]
        diffs.append({"paragraph_start_line": s,
                      "match": m.group(0),
                      "appended": APPEND_C,
                      "before": body[max(0, a - 30):b],
                      "after": new_body[max(0, a - 30):b + len(APPEND_C)]})
        new_chunks.append(new_body)
    # Reassemble text preserving blank lines: paragraphs were extracted with body joined; reconstruct using paragraphs and gaps
    # simpler: rebuild from original lines, replacing paragraphs by body
    # use a different approach: rebuild from paragraph slicing
    return _reassemble_paragraphs(text, paras, new_chunks), diffs


def _reassemble_paragraphs(orig, paras, new_bodies):
    """Reassemble preserving inter-paragraph blank lines."""
    lines = orig.splitlines(keepends=True)
    out = []
    cursor = 0  # 0-based
    for (s, e, _body), new_body in zip(paras, new_bodies):
        # emit any blanks between cursor and s-1
        out.extend(lines[cursor:s - 1])
        out.append(new_body)
        cursor = e - 1
    out.extend(lines[cursor:])
    return "".join(out)


def apply_rule_D(text):
    """Replace pattern D matches with REPL_D anywhere in main-text files."""
    diffs = []
    new_text, n = PAT_D.subn(REPL_D, text)
    if n > 0:
        for m in PAT_D.finditer(text):
            diffs.append({"match": m.group(0), "replacement": REPL_D,
                          "char_offset": m.start()})
    return new_text, diffs


def collect_flags(text, fname):
    flags = []
    for pat, rule, hint in [
        (PAT_B1, "B", "evidence is encoded but outside the operative subspace; near-orthogonality with A is necessary but not sufficient (D1 and D3' are also near-orthogonal to A but operative — see §3.1)"),
        (PAT_B2, "B", "see Rule B suggestion above"),
        (PAT_E, "E", "(D2 excluded from main text due to label imbalance; see appendix)"),
        (PAT_F, "F", "D3'_no_S0: 0.879 mean per-prompt |Δm|"),
    ]:
        # Special handling for B: only flag if 'operative subspace'/D1/D3 NOT within ±2 paragraphs
        for m in pat.finditer(text):
            line_no = text[:m.start()].count("\n") + 1
            if rule == "B":
                ctx = context_window(text, line_no, radius_paragraphs=2).lower()
                if "operative subspace" in ctx or re.search(r"\bd1\b|\bd3\b", ctx):
                    continue
            if rule == "E" and fname in APPENDIX_FILES:
                continue
            if rule == "F":
                line_text = text.splitlines()[line_no - 1] if line_no <= len(text.splitlines()) else ""
                if "appendix" in line_text.lower():
                    continue
            line_text = text.splitlines()[line_no - 1] if line_no <= len(text.splitlines()) else ""
            flags.append({"file": fname, "line": line_no, "rule": rule,
                          "match": m.group(0), "line_text": line_text,
                          "suggestion": hint})
    return flags


def main():
    inv_lines = ["# File inventory\n", f"\nGenerated: {datetime.datetime.now().isoformat()}\n",
                 "\n## Files processed\n"]
    for fp in FILES:
        inv_lines.append(f"- `paper/{fp.name}` ({sum(1 for _ in fp.read_text().splitlines())} lines)\n")
    inv_lines.append(f"\nBackup directory: `paper/.backup_{TS}/`\n")
    (OUT / "file_inventory.md").write_text("".join(inv_lines))

    diffs_all = []
    flags_all = []
    pre_grep = {}; post_grep = {}

    # Pre-patch grep
    pre_grep["A"] = grep_all_files(PAT_A)
    pre_grep["B1"] = grep_all_files(PAT_B1); pre_grep["B2"] = grep_all_files(PAT_B2)
    pre_grep["C"] = grep_all_files(PAT_C)
    pre_grep["D"] = grep_all_files(PAT_D)
    pre_grep["E"] = grep_all_files(PAT_E)
    pre_grep["F"] = grep_all_files(PAT_F)

    for fp in FILES:
        original = fp.read_text()
        # backup
        shutil.copy2(fp, BACKUP / fp.name)
        text = original
        file_diffs = {"A": [], "C": [], "D": []}
        text, dA = apply_rule_A(text);  file_diffs["A"] = dA
        text, dC = apply_rule_C(text);  file_diffs["C"] = dC
        text, dD = apply_rule_D(text);  file_diffs["D"] = dD
        # collect flags from ORIGINAL text (semantic; user reviews source-of-truth)
        flags = collect_flags(original, fp.name)
        flags_all.extend(flags)

        if any(file_diffs.values()):
            fp.write_text(text)
            diffs_all.append({"file": fp.name, "diffs": file_diffs,
                              "n_A": len(dA), "n_C": len(dC), "n_D": len(dD)})

    # Post-patch grep
    post_grep["A"] = grep_all_files(PAT_A)
    post_grep["B1"] = grep_all_files(PAT_B1); post_grep["B2"] = grep_all_files(PAT_B2)
    post_grep["C"] = grep_all_files(PAT_C)
    post_grep["D"] = grep_all_files(PAT_D)
    post_grep["E"] = grep_all_files(PAT_E)
    post_grep["F"] = grep_all_files(PAT_F)

    # ----- Reports -----
    # patches_applied.diff (markdown summary, since no real diff tool here)
    plines = ["# Auto-applied patches (Rules A, C, D)\n",
              f"\nGenerated: {datetime.datetime.now().isoformat()}",
              f"\nBackup: `paper/.backup_{TS}/`\n"]
    if not diffs_all:
        plines.append("\nNo auto-patches applied.\n")
    for fd in diffs_all:
        plines.append(f"\n## paper/{fd['file']}  (A:{fd['n_A']} C:{fd['n_C']} D:{fd['n_D']})\n")
        for rule_key in ("A", "C", "D"):
            for d in fd["diffs"][rule_key]:
                plines.append(f"\n- **Rule {rule_key}** match=`{d.get('match', d.get('orig_match'))}`  →  `{d.get('replacement', d.get('appended'))}`")
                if "line_excerpt" in d:
                    plines.append(f"\n  - excerpt (±30 chars): `{d['line_excerpt'].replace(chr(10),' ⏎ ')}`")
                if "before" in d:
                    plines.append(f"\n  - before: `{d['before']}`\n  - after : `{d['after']}`")
    (OUT / "patches_applied.diff").write_text("\n".join(plines))

    # flags_for_review.md
    flines = ["# Flags for human review (Rules B, E, F)\n",
              f"\nGenerated: {datetime.datetime.now().isoformat()}\n"]
    for rule_letter in ("B", "E", "F"):
        rule_flags = [f for f in flags_all if f["rule"] == rule_letter]
        flines.append(f"\n## Rule {rule_letter} — {len(rule_flags)} flag(s)\n")
        if rule_letter == "B":
            flines.append("\nPattern: 'evidence ↔ orthogonal ↔ inert/non-operative' without operative-subspace context within ±2 paragraphs.\n")
        elif rule_letter == "E":
            flines.append("\nPattern: standalone 'D2' in non-appendix files. User decision: keep, replace, or move to appendix.\n")
        elif rule_letter == "F":
            flines.append("\nPattern: original D3 numerical values (0.625 / 0.643 / D3 ... 0.6/0.5) in main-text contexts.\n")
        if not rule_flags:
            flines.append("\n_(none)_\n")
        for f in rule_flags:
            flines.append(f"\n- `paper/{f['file']}:{f['line']}`  match=`{f['match']}`")
            flines.append(f"\n  - line: `{f['line_text']}`")
            flines.append(f"\n  - suggested: {f['suggestion']}")
    (OUT / "flags_for_review.md").write_text("\n".join(flines))

    # integrity_check.md
    # Rule semantics:
    #   A, D: replace-style (post should be 0 if all matches were patched)
    #   C   : append-style (substring preserved; pre==post is expected)
    #   B*, E, F: flag-only (no modifications; pre==post by design)
    n_C_applied = sum(len(fd["diffs"]["C"]) for fd in diffs_all)
    n_A_applied = sum(len(fd["diffs"]["A"]) for fd in diffs_all)
    n_D_applied = sum(len(fd["diffs"]["D"]) for fd in diffs_all)
    n_B_flags = sum(1 for f in flags_all if f["rule"] == "B")
    n_E_flags = sum(1 for f in flags_all if f["rule"] == "E")
    n_F_flags = sum(1 for f in flags_all if f["rule"] == "F")
    semantics = {
        "A": ("replace", n_A_applied),
        "B1": ("flag-only", n_B_flags), "B2": ("flag-only", n_B_flags),
        "C": ("append", n_C_applied),
        "D": ("replace", n_D_applied),
        "E": ("flag-only", n_E_flags), "F": ("flag-only", n_F_flags),
    }
    ilines = ["# Post-patch grep integrity check\n",
              f"\nGenerated: {datetime.datetime.now().isoformat()}\n",
              "\nLegend:\n",
              "- replace = zero post-count required (Rules A, D)\n",
              "- append  = post==pre expected, substring preserved (Rule C)\n",
              "- flag-only = raised-flags column reflects items routed to flags_for_review.md\n",
              "  (raw match count may exceed flag count because appendix / protected contexts are skipped)\n"]
    for k in ("A", "B1", "B2", "C", "D", "E", "F"):
        pre = len(pre_grep[k]); post = len(post_grep[k])
        sem, n_app = semantics[k]
        if sem == "replace":
            status = "✅ zero" if post == 0 else "⚠️ residual"
            cnt_label = f"applied={n_app}"
        elif sem == "append":
            status = "✅ append-style (substring preserved)" if post == pre else "⚠️ pre/post mismatch"
            cnt_label = f"applied={n_app}"
        else:
            status = "✅ flag-only"
            cnt_label = f"flags_raised={n_app}"
        ilines.append(f"\n## Pattern {k}  ({sem}, {cnt_label})  pre={pre} → post={post}  [{status}]\n")
        if sem == "replace" and post > 0:
            for h in post_grep[k]:
                ilines.append(f"  - residual: `{h['file']}:{h['line']}`  `{h['match']}`\n")
    (OUT / "integrity_check.md").write_text("".join(ilines))

    # Print compact summary
    print(f"[backup] {BACKUP}")
    print(f"[files] processed {len(FILES)}; modified {len(diffs_all)}")
    for k in ("A", "B1", "B2", "C", "D", "E", "F"):
        print(f"  {k}: pre={len(pre_grep[k])} post={len(post_grep[k])}")
    print(f"[flags] B={sum(1 for f in flags_all if f['rule']=='B')} "
          f"E={sum(1 for f in flags_all if f['rule']=='E')} "
          f"F={sum(1 for f in flags_all if f['rule']=='F')}")
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
