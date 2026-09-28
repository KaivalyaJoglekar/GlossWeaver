from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

LABEL_NAMES = (
    "ARTICLE",
    "PREPOSITION",
    "AUXILIARY",
    "PRONOUN",
    "TENSE_ASPECT",
    "AGREEMENT_INFLECTION",
    "WORD_ORDER",
)

ARTICLES = {"a", "an", "the"}
PREPOSITIONS = {
    "about", "above", "across", "after", "against", "along", "among", "around",
    "at", "before", "behind", "below", "beneath", "beside", "between", "beyond",
    "by", "despite", "down", "during", "for", "from", "in", "inside", "into",
    "near", "of", "off", "on", "onto", "out", "outside", "over", "through",
    "to", "toward", "under", "until", "up", "upon", "with", "within", "without",
}
AUXILIARIES = {
    "am", "are", "be", "been", "being", "can", "could", "did", "do", "does",
    "had", "has", "have", "is", "may", "might", "must", "shall", "should",
    "was", "were", "will", "would",
}
PRONOUNS = {
    "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it",
    "its", "itself", "we", "us", "our", "ours", "ourselves", "they", "them",
    "their", "theirs", "themselves", "who", "whom", "whose", "this", "that",
    "these", "those",
}
IRREGULAR_LEMMAS = {
    "am": "be", "is": "be", "are": "be", "was": "be", "were": "be",
    "been": "be", "being": "be", "went": "go", "gone": "go", "goes": "go", "did": "do",
    "done": "do", "had": "have", "has": "have", "bought": "buy", "brought": "bring",
    "came": "come", "got": "get", "gave": "give", "made": "make", "said": "say",
    "saw": "see", "seen": "see", "took": "take", "taken": "take", "thought": "think",
    "wrote": "write", "written": "write", "ran": "run", "ate": "eat", "eaten": "eat",
    "caught": "catch", "chose": "choose", "chosen": "choose", "drove": "drive",
    "driven": "drive", "found": "find", "sold": "sell", "taught": "teach",
    "threw": "throw", "thrown": "throw",
    "children": "child", "men": "man", "women": "woman", "people": "person",
}
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)?")


@dataclass(frozen=True)
class LabelDecision:
    label: str
    value: int
    reasons: tuple[str, ...]


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def simple_lemma(token: str) -> str:
    token = token.lower()
    if token in IRREGULAR_LEMMAS:
        return IRREGULAR_LEMMAS[token]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        stem = token[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(token) > 4 and token.endswith("ed"):
        stem = token[:-2]
        if stem.endswith("i"):
            return stem[:-1] + "y"
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(token) > 3 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


@lru_cache(maxsize=1)
def _spacy_model():
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        return None


def _target_features(target: str) -> dict[str, object]:
    tokens = tokenize(target)
    model = _spacy_model()
    if model is None:
        lemmas = [simple_lemma(token) for token in tokens]
        verb_forms = [
            token for token in tokens
            if token in IRREGULAR_LEMMAS or token.endswith(("ed", "ing"))
        ]
        agreement = [
            token for token in tokens
            if token.endswith("s") and len(token) > 3 and token not in AUXILIARIES
        ]
        return {
            "tokens": tokens,
            "lemmas": lemmas,
            "verbs_with_morphology": verb_forms,
            "agreement_tokens": agreement,
            "parser": "lexical-fallback",
        }

    doc = model(target)
    verbs = [
        token.text.lower() for token in doc
        if token.pos_ in {"VERB", "AUX"}
        and (token.text.lower() != token.lemma_.lower() or token.morph)
    ]
    agreement = [
        token.text.lower() for token in doc
        if token.morph.get("Number") or token.morph.get("Person")
    ]
    return {
        "tokens": [token.text.lower() for token in doc if not token.is_punct],
        "lemmas": [token.lemma_.lower() for token in doc if not token.is_punct],
        "verbs_with_morphology": verbs,
        "agreement_tokens": agreement,
        "parser": "spacy-en_core_web_sm",
    }


def explain_grammar_labels(gloss: str, target: str) -> list[LabelDecision]:
    gloss_tokens = set(tokenize(gloss))
    gloss_lemmas = {simple_lemma(token) for token in gloss_tokens}
    features = _target_features(target)
    target_tokens = set(features["tokens"])

    decisions: list[LabelDecision] = []
    categories = (
        ("ARTICLE", ARTICLES),
        ("PREPOSITION", PREPOSITIONS),
        ("AUXILIARY", AUXILIARIES),
        ("PRONOUN", PRONOUNS),
    )
    for label, lexicon in categories:
        missing = sorted((target_tokens & lexicon) - gloss_tokens)
        decisions.append(
            LabelDecision(
                label,
                int(bool(missing)),
                tuple(f"target token '{token}' is not lexicalized in gloss" for token in missing)
                or ("no unmatched target feature found",),
            )
        )

    tense_reasons = []
    for token in features["verbs_with_morphology"]:
        lemma = simple_lemma(token)
        if token not in gloss_tokens and lemma in gloss_lemmas:
            tense_reasons.append(f"target verb '{token}' realizes gloss lemma '{lemma}'")
        elif token not in gloss_tokens and token in IRREGULAR_LEMMAS:
            tense_reasons.append(f"target irregular verb '{token}' is absent from gloss")
    decisions.append(
        LabelDecision(
            "TENSE_ASPECT", int(bool(tense_reasons)),
            tuple(tense_reasons) or ("no unmatched tense/aspect realization found",),
        )
    )

    agreement_reasons = []
    for token in features["agreement_tokens"]:
        lemma = simple_lemma(token)
        if token not in gloss_tokens and lemma in gloss_lemmas and token not in AUXILIARIES:
            agreement_reasons.append(f"target form '{token}' inflects gloss lemma '{lemma}'")
    decisions.append(
        LabelDecision(
            "AGREEMENT_INFLECTION", int(bool(agreement_reasons)),
            tuple(agreement_reasons) or ("no unmatched agreement/inflection found",),
        )
    )
    ignored = ARTICLES | PREPOSITIONS | AUXILIARIES | PRONOUNS
    gloss_content = [simple_lemma(token) for token in tokenize(gloss) if token not in ignored]
    target_content = [
        simple_lemma(token) for token in features["tokens"] if token not in ignored
    ]
    shared = set(gloss_content) & set(target_content)
    gloss_order = [token for token in gloss_content if token in shared]
    target_order = [token for token in target_content if token in shared]
    changed = len(shared) >= 2 and gloss_order != target_order
    decisions.append(LabelDecision(
        "WORD_ORDER",
        int(changed),
        (f"shared content order differs: gloss={gloss_order}, target={target_order}",)
        if changed else ("shared content order is unchanged or indeterminate",),
    ))
    return decisions


def extract_grammar_labels(gloss: str, target: str) -> list[int]:
    decisions = explain_grammar_labels(gloss, target)
    by_name = {decision.label: decision.value for decision in decisions}
    return [by_name[name] for name in LABEL_NAMES]


def label_dict(labels: Iterable[int]) -> dict[str, int]:
    values = list(labels)
    if len(values) != len(LABEL_NAMES):
        raise ValueError(f"Expected {len(LABEL_NAMES)} labels, got {len(values)}")
    return dict(zip(LABEL_NAMES, values, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain weak grammar labels")
    parser.add_argument("--gloss", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    decisions = explain_grammar_labels(args.gloss, args.target)
    print(json.dumps({
        "gloss": args.gloss,
        "target": args.target,
        "labels": {item.label: item.value for item in decisions},
        "reasons": {item.label: list(item.reasons) for item in decisions},
        "parser": _target_features(args.target)["parser"],
    }, indent=2))


if __name__ == "__main__":
    main()
