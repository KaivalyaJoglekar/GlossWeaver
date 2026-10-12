from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass

from .grammar_labels import ARTICLES, AUXILIARIES, PREPOSITIONS, tokenize

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
    probabilities: dict[str, float] | None = None,
    word_order: bool = False,
) -> CorruptionResult:
    if not sentence.strip():
        raise ValueError("Cannot corrupt an empty sentence")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if max_operations < 0:
        raise ValueError("max_operations must be nonnegative")
    probabilities = probabilities or {}
    unknown = probabilities.keys() - set(TRANSFORMATIONS) - {"WORD_ORDER"}
    if unknown or any(not 0 <= value <= 1 for value in probabilities.values()):
        raise ValueError("Unknown transformation or probability outside [0, 1]")
    def chance(operation: str) -> bool:
        return rng.random() < probabilities.get(operation, probability)
    rng = random.Random(seed)
    from .build_reconstruction_data import _safe_lemma
    original_tokens = tokenize(sentence)
    tokens = list(original_tokens)
    # Preserve original context even after function words are dropped.
    lemmas = {}
    for index, token in enumerate(original_tokens):
        candidate = _safe_lemma(token, original_tokens[index - 1] if index else None)
        if token in lemmas and lemmas[token] != candidate:
            lemmas[token] = token  # Repeated words with ambiguous roles stay intact.
        else:
            lemmas[token] = candidate
    operations: list[str] = []

    def drop_from(lexicon: set[str], operation: str) -> None:
        nonlocal tokens
        if len(operations) >= max_operations or not any(token in lexicon for token in tokens):
            return
        if chance(operation):
            retained = [token for token in tokens if token not in lexicon]
            if retained:
                tokens = retained
                operations.append(operation)

    drop_from(ARTICLES, "ARTICLE_DROP")
    drop_from(PREPOSITIONS, "PREPOSITION_DROP")
    drop_from(AUXILIARIES - {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}, "AUXILIARY_DROP")

    if len(operations) < max_operations and chance("VERB_LEMMA"):
        lemmatized = [lemmas.get(token, token) for token in tokens]
        if lemmatized != tokens:
            tokens = lemmatized
            operations.append("VERB_LEMMA")

    if len(operations) < max_operations and chance("PUNCTUATION_DROP"):
        if re.search(r"[^\w\s']", sentence):
            operations.append("PUNCTUATION_DROP")

    if len(operations) < max_operations and chance("AGREEMENT_SIMPLIFICATION"):
        simplified = [lemmas.get(token, token) if token.endswith("s") else token for token in tokens]
        if simplified != tokens:
            tokens = simplified
            operations.append("AGREEMENT_SIMPLIFICATION")

    # Move only an explicit final temporal adjunct; never randomly scramble roles.
    if word_order and len(operations) < max_operations and len(tokens) > 2 and tokens[-1] in {"yesterday", "today", "tomorrow"} and chance("WORD_ORDER"):
        tokens = [tokens[-1], *tokens[:-1]]
        operations.append("WORD_ORDER")
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
