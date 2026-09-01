from __future__ import annotations

import re


class CopyHeuristicBaseline:
    """E0: deterministic normalization lower bound; it is not a learned model."""

    def predict(self, gloss: str) -> str:
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", gloss)
        if not words:
            return ""
        sentence = " ".join(word.lower() for word in words)
        return sentence[0].upper() + sentence[1:] + "."

    def predict_batch(self, glosses: list[str]) -> list[str]:
        return [self.predict(gloss) for gloss in glosses]

