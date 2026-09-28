from __future__ import annotations

import re


_SPACE_RE = re.compile(r"\s+")
INSTRUCTION = "Convert this gloss-like text into a fluent English sentence:"


def normalize_gloss(gloss: str) -> str:
    """Normalize whitespace without changing or stemming lexical content."""
    return _SPACE_RE.sub(" ", gloss.strip())


def format_gloss_input(gloss: str) -> str:
    """The sole input formatter used by new training, evaluation, API and CLI code."""
    normalized = normalize_gloss(gloss)
    if not normalized:
        raise ValueError("Gloss input is empty")
    return f"{INSTRUCTION} {normalized}"
