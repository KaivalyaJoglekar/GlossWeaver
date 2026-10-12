"""Advisory lexical preservation checks; no learned entailment or confidence claim."""
from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

from glossweaver.data.grammar_labels import ARTICLES, AUXILIARIES, PREPOSITIONS, PRONOUNS, simple_lemma, tokenize

NEGATION = re.compile(r"\b(?:not|no|never|without|neither|nor)\b|\b\w+n['’]t\b", re.I)
NUMBERS = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)")
DATES = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?\b")
TIME = {'yesterday': 'past', 'ago': 'past', 'tomorrow': 'future', 'tonight': 'tonight', 'today': 'today', 'now': 'now', 'next': 'next', 'last': 'last'}
STOP = ARTICLES | AUXILIARIES | PREPOSITIONS | PRONOUNS | {'not', 'no', 'never', 'please', 'yes'} | set(TIME)
# Small, explicitly documented equivalences, rather than treating all lexical changes as errors.
EQUIVALENTS = {'purchase': 'buy', 'purchased': 'buy', 'purchases': 'buy', 'shop': 'store', 'physician': 'doctor', 'medication': 'medicine', 'automobile': 'car'}
FORM_LEMMAS = {'sees': 'see', 'uses': 'use', 'takes': 'take', 'makes': 'make', 'likes': 'like', 'loves': 'love', 'arrives': 'arrive', 'writes': 'write', 'reads': 'read'}
NUMBER_WORDS = dict(zip('zero one two three four five six seven eight nine ten eleven twelve'.split(), map(str, range(13))))


def _numbers(text: str) -> Counter:
    return Counter(NUMBERS.findall(text)) + Counter(NUMBER_WORDS[t] for t in tokenize(text) if t in NUMBER_WORDS)


def _content(text: str) -> Counter:
    return Counter(EQUIVALENTS.get(t, FORM_LEMMAS.get(t, simple_lemma(t))) for t in tokenize(text) if t not in STOP and not t.isdigit() and t not in NUMBER_WORDS)


def validate_meaning(source: str, generated: str) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    checks = []

    def check(category: str, mismatch: bool, explanation: str, *, severity: str = 'warning', source_span: str | None = None, generated_span: str | None = None, available: bool = True) -> None:
        checks.append({'category': category, 'status': 'unavailable' if not available else 'potential_issue' if mismatch else 'no_issue_detected'})
        if mismatch and available:
            issues.append({'category': category, 'severity': severity, 'explanation': explanation, 'source_span': source_span, 'generated_span': generated_span})

    neg_source, neg_output = NEGATION.search(source), NEGATION.search(generated)
    check('negation', bool(neg_source) != bool(neg_output), 'Explicit negation differs between the input and reconstruction. Verify polarity.', source_span=neg_source.group() if neg_source else None, generated_span=neg_output.group() if neg_output else None)
    check('numbers', _numbers(source) != _numbers(generated), 'Numeric values or their counts differ. Verify quantities.', source_span=', '.join(_numbers(source)), generated_span=', '.join(_numbers(generated)))
    check('dates', Counter(DATES.findall(source)) != Counter(DATES.findall(generated)), 'Date notation differs. Verify the intended date; equivalent date formats may also trigger this check.')
    s_time, g_time = set(tokenize(source)) & TIME.keys(), set(tokenize(generated)) & TIME.keys()
    check('temporal', s_time != g_time, 'Explicit time words differ. Tense and paraphrases need human verification.', source_span=', '.join(sorted(s_time)), generated_span=', '.join(sorted(g_time)))
    # Capitalization is not entity annotation. All-caps gloss has no usable name signal.
    excluded = PRONOUNS | AUXILIARIES | ARTICLES | set(TIME) | {'english'}
    names = [t for t in re.findall(r"\b[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)*\b", source) if t.lower() not in excluded]
    missing_names = [t for t in names if not re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", generated, re.I)]
    check('named_entities', bool(missing_names), 'A capitalized source token is absent. It may be a name; capitalization alone is uncertain.', severity='notice', source_span=', '.join(missing_names) or None, available=bool(names) and not source.isupper())
    missing = list((_content(source) - _content(generated)).elements())
    added = list((_content(generated) - _content(source)).elements())
    check('content_omission', bool(missing), 'Source content words may be missing or paraphrased: ' + ', '.join(missing), severity='notice', source_span=', '.join(missing) or None)
    check('unsupported_addition', bool(added), 'Additional content words may require context: ' + ', '.join(added), severity='notice', generated_span=', '.join(added) or None)
    return {'status': 'potential_issue' if issues else 'no_issue_detected', 'issues': issues, 'checks': checks, 'advisory': 'Rule-based notices can flag valid paraphrases and miss meaning changes. No issue detected is not proof of equivalence.', 'validator_version': 'lexical-v1'}


def align_changes(source: str, generated: str) -> list[dict[str, str]]:
    """Text alignment is a surface edit description, never a model reasoning trace."""
    a, b = source.split(), generated.split()
    result = []
    for operation, i, j, k, l in SequenceMatcher(None, [t.lower() for t in a], [t.lower() for t in b], autojunk=False).get_opcodes():
        before, after = ' '.join(a[i:j]), ' '.join(b[k:l])
        category = 'unchanged' if operation == 'equal' else 'uncategorized text change'
        words = set(tokenize(after))
        if operation == 'insert' and words and words <= ARTICLES:
            category = 'added article'
        elif operation == 'insert' and words and words <= AUXILIARIES:
            category = 'added auxiliary'
        result.append({'operation': operation, 'source': before, 'generated': after, 'category': category})
    return result
