from __future__ import annotations

import re
from collections import Counter


ERROR_CATEGORIES = (
    "ARTICLE", "PREPOSITION", "TENSE", "PRONOUN", "WORD_ORDER", "LEXICAL",
    "OMISSION", "HALLUCINATION", "REPETITION", "VALID_PARAPHRASE", "OTHER",
)


def possible_hallucination_tokens(gloss: str, prediction: str) -> list[str]:
    """Conservative lexical flag for manual review, not a semantic verdict."""
    gloss_tokens = set(re.findall(r"[A-Za-z]+", gloss.lower()))
    function_words = {
        "a", "an", "the", "to", "of", "in", "on", "at", "by", "for", "from",
        "with", "is", "are", "was", "were", "be", "been", "has", "have", "had",
        "i", "me", "you", "he", "him", "she", "her", "it", "we", "us", "they", "them",
    }
    return sorted({
        token for token in re.findall(r"[A-Za-z]+", prediction.lower())
        if token not in gloss_tokens and token not in function_words
    })


def repeated_tokens(prediction: str, minimum_repetitions: int = 3) -> list[str]:
    counts = Counter(re.findall(r"[A-Za-z]+", prediction.lower()))
    return sorted(token for token, count in counts.items() if count >= minimum_repetitions)

