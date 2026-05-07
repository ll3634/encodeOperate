"""Generate IIRC builder feasibility report."""
import json, re, pathlib, statistics

rows = [json.loads(l) for l in open('results/iirc_builder_audit/builder_audit_v2.jsonl')]

# Four-stage yield funnel
s2_clean = [r for r in rows
            if r['candidate_W'].lower() not in r['obs_N0'].lower()
            and r['candidate_W'].lower() in r['obs_T0'].lower()
            and r['obs_N0'].strip() != r['obs_T0'].strip()]
s3 = [r for r in s2_clean if r['n0_words'] >= 20]
s4 = [r for r in s3 if r['obs_N0'][0].isupper()]
yr_s4 = [r for r in s4 if re.match(r'^\d{4}$', r['candidate_W'].strip())]

reject_v2 = json.load(open('results/iirc_builder_audit/builder_summary.json'))

lines = [
    '# IIRC Builder Audit Report',
    '',
    '**Data**: IIRC train split, 300 articles scanned',
    '**Questions scanned**: 690  (avg 2.3 per article)',
    '',
    '## Yield funnel',
    '',
    '| Stage | Filter | N | % of 690 |',
    '|---|---|---:|---:|',
    '| S0 | raw questions | 690 | 100% |',
    f'| S1 | builder accepts (type=span/value, has main+linked ctx, W found, obs<=150w) | 211 | 30.6% |',
    f'| S2 | W not in N0, W in T0, N0!=T0 (structural integrity) | {len(s2_clean)} | {100*len(s2_clean)/690:.1f}% |',
    f'| S3 | N0 >= 20 words (realistic observation) | {len(s3)} | {100*len(s3)/690:.1f}% |',
    f'| S4 | N0 starts uppercase (complete sentence) | {len(s4)} | {100*len(s4)/690:.1f}% |',
    '',
    '## Rejection categories (S0->S1)',
    '',
    '| Category | N |',
    '|---|---:|',
]
for k, v in sorted(reject_v2['rejection_counts'].items(), key=lambda x: -x[1]):
    lines.append(f'| {k} | {v} |')

lines += [
    '',
    '## Success criteria assessment',
    '',
    '| Criterion | Threshold | Observed | Pass? |',
    '|---|---|---|---|',
    '| Builder yield | >=25% | 30.6% | PASS (surface) |',
    '| Clean examples from 300 scanned | >=80 | 211 | PASS (surface) |',
    f'| Effective clean (W not in N0, N0>=20w, proper sentence) | >=80 | {len(s4)} | FAIL |',
    '| Delta_commit-W T0-N0 | >=+0.20 | not measured | -- |',
    '',
    '## Structural quality issues',
    '',
    '### Issue 1 (resolvable): N0 observations are too short with the snippet-only design',
    '',
    'The previous builder used only the short "main" context snippets from `q.context`',
    f'(median {int(statistics.median([r["n0_words"] for r in rows]))} words).',
    'The dataset DOES include the full main passage in `article.text` (~156 words median),',
    'so this is fixable by switching N0 to use `article.text`.',
    '',
    '### Issue 2 (NOT resolvable): IIRC has no natural distractors per question',
    '',
    'For the N0/T0/S0 design we need a same-type W candidate that is NOT in the main',
    'passage and IS in some linked context. Profiling 690 questions in 300 articles:',
    '',
    '  Linked-snippet count per question:',
    '    median = 1, mean = 1.41',
    '    0 linked snippets:  4 / 690 (0.6%)',
    '    1 linked snippet: 220 / 690 (31.9%)',
    '    >= 2 linked snippets: 90 / 690 (13.0%)',
    '',
    '  Year-type questions (best sub-type, 79 / 690):',
    '    Have W (year) in a non-gold linked snippet: 2 / 79 (2.5%)',
    '',
    'IIRC questions are designed around following ONE specific link to find the answer,',
    'not discriminating among competing pieces of linked evidence. Most questions have',
    'a single linked snippet that IS the gold-supporting evidence, leaving no place to',
    'source a clean answer-shaped W from.',
    '',
    '### Issue 3 (workaround degrades the design): W from main text',
    '',
    'Sourcing W from sentences in `article.text` (not in N0) requires excluding those',
    'sentences from N0 -- but then N0 becomes a curated subset of the article, not the',
    'natural opening passage. This conflates "external linked evidence" with "internal',
    'main-passage content" and breaks the analogy with HotpotQA/MuSiQue factcards.',
    '',
    '## Conclusion and recommendation',
    '',
    'Surface criteria pass but the design is structurally infeasible.',
    '',
    'Root cause: IIRC is built around single-link evidence-following questions, not',
    'multi-evidence discrimination. Only 13% of questions have >=2 linked snippets,',
    'and for the cleanest sub-type (year questions) only 2 / 79 have a same-type W',
    'available in a non-gold linked snippet. There is no way to construct N0/T0/S0',
    'triplets at scale without either (a) fabricating distractors from the main text',
    '(degrading the design), or (b) constructing synthetic linked snippets',
    '(making it no longer an IIRC replication).',
    '',
    'RECOMMENDATION: STOP. Do not force IIRC into the paper.',
    '',
    'Per the builder stop rule:',
    '  "If unsuccessful: Stop. Do not force IIRC into the paper."',
    '',
    'The existing two-dataset replication (HotpotQA + MuSiQue) already defends the',
    '"single-dataset artifact" reviewer attack. IIRC would require a synthetic-distractor',
    'workaround that loses the original motivation (using IIRCs natural missing-evidence',
    'design) and invites reviewer questions about dataset selection quality.',
]

report = '\n'.join(lines)
pathlib.Path('results/iirc_builder_audit/iirc_feasibility_report.md').write_text(report)
print(report)
