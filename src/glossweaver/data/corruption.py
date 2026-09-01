from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass

from .grammar_labels import ARTICLES, AUXILIARIES, PREPOSITIONS, simple_lemma, tokenize

TRANSFORMATIONS = (
    "ARTICLE_DROP",
    "PREPOSITION_DROP",
    "AUXILIARY_DROP",
    "VERB_LEMMA",
    "PUNCTUATION_DROP",
    "AGREEMENT_SIMPLIFICATION",
)


@dataclass(frozen=True)
class CorruptionResult:
    fragment: str
    target: str
    operations: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {"fragment": self.fragment, "target": self.target, "operations": list(self.operations)}


def corrupt_sentence(
    sentence: str,
    *,
    seed: int | None = None,
    probability: float = 0.8,
    max_operations: int = 5,
) -> CorruptionResult:
    if not sentence.strip():
        raise ValueError("Cannot corrupt an empty sentence")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    rng = random.Random(seed)
    original_tokens = tokenize(sentence)
    tokens = list(original_tokens)
    operations: list[str] = []

    def drop_from(lexicon: set[str], operation: str) -> None:
        nonlocal tokens
        if len(operations) >= max_operations or not any(token in lexicon for token in tokens):
            return
        if rng.random() <= probability:
            retained = [token for token in tokens if token not in lexicon]
            if retained:
                tokens = retained
                operations.append(operation)

    drop_from(ARTICLES, "ARTICLE_DROP")
    drop_from(PREPOSITIONS, "PREPOSITION_DROP")
    drop_from(AUXILIARIES, "AUXILIARY_DROP")

    if len(operations) < max_operations and rng.random() <= probability:
        lemmatized = [simple_lemma(token) for token in tokens]
        if lemmatized != tokens:
            tokens = lemmatized
            operations.append("VERB_LEMMA")

    if len(operations) < max_operations and rng.random() <= probability:
        if re.search(r"[^\w\s']", sentence):
            operations.append("PUNCTUATION_DROP")

    if len(operations) < max_operations and rng.random() <= probability:
        simplified = [simple_lemma(token) if token.endswith("s") else token for token in tokens]
        if simplified != tokens:
            tokens = simplified
            operations.append("AGREEMENT_SIMPLIFICATION")

    if not tokens:
        content = [
            token for token in original_tokens
            if token not in ARTICLES | PREPOSITIONS | AUXILIARIES
        ]
        tokens = content or original_tokens[:1]
    return CorruptionResult(" ".join(tokens).upper(), sentence.strip(), tuple(operations))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reproducible telegraphic fragment")
    parser.add_argument("text")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probability", type=float, default=0.8)
    args = parser.parse_args()
    print(json.dumps(corrupt_sentence(
        args.text, seed=args.seed, probability=args.probability
    ).as_dict(), indent=2))


if __name__ == "__main__":
    main()

