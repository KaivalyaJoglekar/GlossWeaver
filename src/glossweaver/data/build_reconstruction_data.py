from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from glossweaver.data.grammar_labels import ARTICLES, AUXILIARIES, PREPOSITIONS, IRREGULAR_LEMMAS
from glossweaver.settings import ROOT


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
ADJECTIVES = {
    "big", "small", "little", "large", "young", "old", "new", "red", "blue",
    "green", "black", "white", "good", "bad", "happy", "sad", "fast", "slow",
    "strong", "weak", "skilled", "beautiful", "important", "clear", "dark", "bright", "sunny",
}
DROPPED_PREPOSITIONS = PREPOSITIONS - {"outside"}
COMMON_VERBS = {
    "ask", "bake", "buy", "call", "catch", "change", "choose", "come", "cook", "do", "drive", "explain", "fall", "feel",
    "find", "give", "go", "have", "hear", "help", "keep", "know", "leave", "like",
    "live", "look", "make", "move", "need", "open", "play", "prepare", "read", "remember", "review", "run",
    "say", "see", "sell", "serve", "show", "speak", "stand", "study", "take", "teach", "tell", "think", "throw", "try",
    "use", "visit", "walk", "want", "watch", "work", "write",
}

INFLECTIONS = {
    "bake": ("baked", "baking", "bakes"), "buy": ("bought", "buying", "buys"),
    "catch": ("caught", "catching", "catches"), "choose": ("chose", "choosing", "chooses"),
    "cook": ("cooked", "cooking", "cooks"), "drive": ("drove", "driving", "drives"),
    "explain": ("explained", "explaining", "explains"), "find": ("found", "finding", "finds"),
    "give": ("gave", "giving", "gives"), "go": ("went", "going", "goes"),
    "make": ("made", "making", "makes"), "need": ("needed", "needing", "needs"),
    "play": ("played", "playing", "plays"), "prepare": ("prepared", "preparing", "prepares"),
    "read": ("read", "reading", "reads"), "review": ("reviewed", "reviewing", "reviews"),
    "run": ("ran", "running", "runs"), "sell": ("sold", "selling", "sells"),
    "serve": ("served", "serving", "serves"), "show": ("showed", "showing", "shows"),
    "study": ("studied", "studying", "studies"), "teach": ("taught", "teaching", "teaches"),
    "throw": ("threw", "throwing", "throws"), "visit": ("visited", "visiting", "visits"),
    "walk": ("walked", "walking", "walks"), "want": ("wanted", "wanting", "wants"),
}


@dataclass(frozen=True)
class Pair:
    fragment: str
    target: str
    operations: tuple[str, ...]


def _safe_lemma(token: str, previous: str | None = None) -> str:
    lower = token.lower()
    if lower in IRREGULAR_LEMMAS:
        return IRREGULAR_LEMMAS[lower]
    if lower.endswith("ing") and len(lower) > 5:
        stem = lower[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        if stem in COMMON_VERBS:
            return stem
        if stem + "e" in COMMON_VERBS:
            return stem + "e"
    if lower.endswith("ed") and len(lower) > 4:
        stem = lower[:-2]
        candidates = (stem, stem[:-1] + "y" if stem.endswith("i") else stem, stem + "e")
        for candidate in candidates:
            if candidate in COMMON_VERBS:
                return candidate
    if lower.endswith("s") and lower[:-1] in COMMON_VERBS and previous not in ARTICLES:
        return lower[:-1]
    return lower


def reconstruct_fragment(sentence: str, *, seed: int) -> Pair | None:
    tokens = [match.group(0).lower() for match in WORD_RE.finditer(sentence)]
    if not 5 <= len(tokens) <= 18:
        return None
    if sum(token.isalpha() for token in tokens) != len(tokens):
        return None
    operations: list[str] = []
    rng = random.Random(seed)
    conditional_drop: set[int] = set()
    for index in range(len(tokens) - 2):
        if tokens[index:index + 3] == ["when", "it", "is"]:
            conditional_drop.update((index, index + 1, index + 2))

    # Preserve a controlled adjective-order signal before function words disappear.
    swap_at: set[int] = set()
    for index in range(1, len(tokens) - 1):
        if tokens[index - 1] in ARTICLES and tokens[index] in ADJECTIVES:
            swap_at.add(index)

    kept: list[str] = []
    previous: str | None = None
    for index, token in enumerate(tokens):
        if index in conditional_drop:
            operations.append("OPTIONAL_PRONOUN_REDUCTION" if token == "it" else "AUXILIARY_DROP")
            previous = token
            continue
        if token in ARTICLES:
            operations.append("ARTICLE_DROP")
            previous = token
            continue
        if token in DROPPED_PREPOSITIONS:
            operations.append("PREPOSITION_DROP")
            previous = token
            continue
        if token in AUXILIARIES:
            operations.append("AUXILIARY_DROP")
            previous = token
            continue
        lemma = _safe_lemma(token, previous)
        if lemma != token:
            operations.extend(("VERB_LEMMA", "TENSE_NEUTRALIZATION"))
        kept.append(lemma)
        previous = token

    # `a red car` becomes the explicitly diagnostic `CAR RED` order.
    lowered = [token for token in tokens if token not in ARTICLES | DROPPED_PREPOSITIONS | AUXILIARIES]

    # `explain math to the pupil` becomes `EXPLAIN PUPIL MATH`; the inverse
    # operation tests semantic role recovery instead of simple phrase copying.
    for index, token in enumerate(tokens):
        if token != "to" or index == 0:
            continue
        context_lemmas = {_safe_lemma(item) for item in tokens[max(0, index - 4):index]}
        if not context_lemmas.intersection({"explain", "show", "give", "teach"}):
            continue
        before = _safe_lemma(tokens[index - 1])
        after_index = index + 1
        while after_index < len(tokens) and tokens[after_index] in ARTICLES:
            after_index += 1
        if after_index >= len(tokens):
            continue
        after = _safe_lemma(tokens[after_index])
        try:
            position = next(i for i in range(len(kept) - 1) if kept[i] == before and kept[i + 1] == after)
        except StopIteration:
            continue
        kept[position:position + 2] = [after, before]
        operations.append("WORD_ORDER")
    for adjective_index in sorted(swap_at):
        adjective = tokens[adjective_index]
        noun = tokens[adjective_index + 1]
        try:
            position = next(i for i in range(len(kept) - 1) if kept[i] == adjective and kept[i + 1] == noun)
        except StopIteration:
            continue
        kept[position:position + 2] = [noun, adjective]
        if adjective_index == 1 and tokens[0] in ARTICLES:
            # Subject modifiers are deliberately displaced to the tail so the
            # model learns attachment rather than merely reversing a bigram.
            kept.pop(position + 1)
            kept.append(adjective)
        operations.extend(("CONTROLLED_ADJECTIVE_ORDER", "WORD_ORDER"))

    if kept and kept[0] == "i" and rng.random() < 0.5:
        kept[0] = "me"
        operations.append("OPTIONAL_PRONOUN_REDUCTION")
    operations.append("PUNCTUATION_DROP")
    if any(a != b for a, b in zip(lowered, kept)):
        operations.append("AGREEMENT_SIMPLIFICATION")
    fragment = " ".join(kept).upper()
    if len(kept) < 3 or not fragment:
        return None
    return Pair(fragment, sentence.strip(), tuple(dict.fromkeys(operations)))


def _sentences(paths: list[Path]):
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                text = str(json.loads(line).get("text", ""))
                for sentence in SENTENCE_RE.split(text):
                    sentence = sentence.strip()
                    if sentence and sentence[-1:] not in ".!?":
                        sentence += "."
                    yield sentence


def _controlled_sentences(*, seed: int, count: int):
    """Yield varied, grammatical source sentences without encoding challenge answers."""
    rng = random.Random(seed)
    seen: set[str] = set()
    people = ["mother", "father", "parent", "woman", "man", "sister", "brother", "aunt", "uncle"]
    learners = ["student", "pupil", "child", "class", "learner"]
    educators = ["teacher", "instructor", "tutor", "professor"]
    children = ["boy", "girl", "child", "student", "player"]
    household_verbs = ["cook", "prepare", "serve", "make", "bake"]
    meals = ["food", "dinner", "lunch", "breakfast", "meal", "soup", "bread", "rice"]
    lesson_topics = ["math", "algebra", "science", "history", "grammar", "lesson", "problem", "rule"]
    commerce_objects = ["car", "vehicle", "book", "coat", "table", "bicycle", "ticket", "gift"]
    colours = ["red", "blue", "green", "black", "white", "new", "small", "large"]
    places = ["park", "school", "store", "market", "library", "garden", "playground", "station"]
    times_by_tense = {
        "past": ["yesterday", "this morning", "this afternoon"],
        "future": ["tonight", "tomorrow", "this evening"],
        "progressive": ["tonight", "today", "this morning", "this afternoon"],
        "present": ["tonight", "today", "this morning", "this evening"],
    }
    animals = ["dog", "cat", "puppy", "horse"]

    def form(subject: str, verb: str, rest: str, tense: str) -> str:
        past, ing, third = INFLECTIONS[verb]
        if tense == "past":
            body = f"The {subject} {past} {rest}"
        elif tense == "future":
            body = f"The {subject} will {verb} {rest}"
        elif tense == "progressive":
            body = f"The {subject} is {ing} {rest}"
        else:
            body = f"The {subject} {third} {rest}"
        return body[0].upper() + body[1:] + "."

    attempts = 0
    while len(seen) < count and attempts < count * 80:
        attempts += 1
        domain = rng.randrange(7)
        tense = rng.choice(["past", "future", "progressive", "present"])
        if domain == 0:
            sentence = form(
                rng.choice(people), rng.choice(household_verbs),
                f"{rng.choice(meals)} {rng.choice(times_by_tense[tense])}", tense,
            )
        elif domain == 1:
            educator, learner, topic = rng.choice(educators), rng.choice(learners), rng.choice(lesson_topics)
            verb = rng.choice(["explain", "teach", "show"])
            topic_phrase = f"the {topic}" if topic in {"lesson", "problem", "rule"} else topic
            rest = f"{topic_phrase} to the {learner}" if verb in {"explain", "show"} else f"the {learner} {topic_phrase}"
            sentence = form(educator, verb, rest, tense)
        elif domain == 2:
            subject, obj, colour = rng.choice(people), rng.choice(commerce_objects), rng.choice(colours)
            sentence = form(subject, rng.choice(["buy", "choose", "sell"]), f"a {colour} {obj}", tense)
        elif domain == 3:
            subject, place = rng.choice(children + animals), rng.choice(places)
            verb = rng.choice(["run", "walk", "play", "visit", "go"])
            if verb in {"run", "walk"}:
                rest = f"{rng.choice(['fast', 'quickly'])} in the {place}"
            elif verb == "play":
                rest = f"in the {place}"
            else:
                rest = f"the {place}" if verb == "visit" else f"to the {place}"
            sentence = form(subject, verb, rest, tense)
        elif domain == 4:
            subject = rng.choice(children)
            verb = rng.choice(["throw", "catch"])
            if rng.random() < 0.65:
                past, ing, third = INFLECTIONS[verb]
                inflected = past if tense == "past" else third
                adjective = rng.choice(["strong", "young", "skilled", "happy"])
                obj = rng.choice(["ball", "stone", "rope"])
                sentence = f"The {adjective} {subject} {inflected} the {obj}."
            else:
                rest = rng.choice(["the ball with force", "a small ball outside", "the red ball in the park"])
                sentence = form(subject, verb, rest, tense)
        elif domain == 5:
            subject, topic = rng.choice(learners), rng.choice(lesson_topics)
            sentence = form(subject, rng.choice(["study", "read", "review"]), f"the {topic} at school", tense)
        else:
            subject = rng.choice(children + ["dog", "puppy"])
            if rng.random() < 0.5:
                sentence = form(subject, "play", "outside on a sunny day", tense)
            else:
                _, _, third = INFLECTIONS["play"]
                sentence = f"The sunny {subject} {third} outside."
        if sentence.casefold() not in seen:
            seen.add(sentence.casefold())
            yield sentence
    if len(seen) < count:
        raise RuntimeError(f"Could only construct {len(seen)} of {count} requested controlled sentences")


def _conditional_sentences(*, seed: int, count: int):
    subjects = ["boy", "girl", "child", "student", "player", "pupil", "dog", "puppy", "cat", "family"]
    verbs = ["play", "run", "walk", "read", "study"]
    conditions = ["sunny", "warm", "cold", "bright", "rainy"]
    tenses = ["past", "future", "progressive", "present"]
    candidates: list[str] = []
    for subject in subjects:
        for verb in verbs:
            for condition in conditions:
                for tense in tenses:
                    past, ing, third = INFLECTIONS[verb]
                    if tense == "past":
                        predicate = past
                    elif tense == "future":
                        predicate = f"will {verb}"
                    elif tense == "progressive":
                        predicate = f"is {ing}"
                    else:
                        predicate = third
                    candidates.append(
                        f"The {subject} {predicate} outside when it is {condition}."
                    )
    random.Random(seed).shuffle(candidates)
    if count > len(candidates):
        raise RuntimeError(f"Requested {count} conditional sentences, maximum is {len(candidates)}")
    yield from candidates[:count]


def _schedule_sentences(*, seed: int, count: int):
    subjects = ["boy", "girl", "child", "student", "teacher", "mother", "father", "woman", "man", "family"]
    places = ["school", "store", "park", "library", "market"]
    periods = ["morning", "afternoon", "evening", "night"]
    tenses = ["past", "future", "progressive", "present"]
    candidates: list[str] = []
    for subject in subjects:
        for place in places:
            for period in periods:
                for tense in tenses:
                    past, ing, third = INFLECTIONS["go"]
                    if tense == "past":
                        predicate = past
                    elif tense == "future":
                        predicate = "will go"
                    elif tense == "progressive":
                        predicate = f"is {ing}"
                    else:
                        predicate = third
                    candidates.append(f"The {subject} {predicate} to the {place} in the {period}.")
    random.Random(seed).shuffle(candidates)
    if count > len(candidates):
        raise RuntimeError(f"Requested {count} schedule sentences, maximum is {len(candidates)}")
    yield from candidates[:count]


def _challenge_text() -> tuple[set[str], set[str]]:
    challenge = pd.read_csv(ROOT / "data/challenge/glossweaver_challenge.csv").fillna("")
    glosses = set(challenge["gloss"].astype(str).str.upper())
    references = {
        str(row[column]).casefold()
        for _, row in challenge.iterrows()
        for column in ("reference_1", "reference_2", "reference_3")
    }
    return glosses, references


def build_reconstruction_data(
    input_files: list[str | Path],
    output_dir: str | Path = ROOT / "datasets/processed/reconstruction",
    *,
    seed: int = 42,
    maximum: int = 40_000,
    controlled_count: int = 8_000,
    conditional_count: int = 1_000,
    schedule_count: int = 800,
) -> dict[str, int]:
    pairs: list[dict[str, str]] = []
    seen_targets: set[str] = set()
    challenge_glosses, challenge_references = _challenge_text()

    def add_sentence(sentence: str, origin: str) -> None:
        target_key = sentence.casefold()
        if target_key in seen_targets or target_key in challenge_references:
            return
        digest = hashlib.sha256(target_key.encode()).hexdigest()
        pair = reconstruct_fragment(sentence, seed=int(digest[:8], 16) ^ seed)
        if pair is None or pair.fragment in challenge_glosses:
            return
        seen_targets.add(target_key)
        bucket = int(digest[-8:], 16) % 100
        split = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
        pairs.append({
            "id": f"synthetic_telegraphic:{digest[:20]}",
            "gloss": pair.fragment,
            "fragment": pair.fragment,
            "target": pair.target,
            "operations": "|".join(pair.operations),
            "source_type": "synthetic_telegraphic",
            "corpus_origin": origin,
            "split": split,
        })

    for sentence in _sentences([Path(path) for path in input_files]):
        add_sentence(sentence, "public_librispeech_clean")
        if len(pairs) >= maximum:
            break
    for sentence in _controlled_sentences(seed=seed + 103, count=controlled_count):
        add_sentence(sentence, "controlled_compositional")
    for sentence in _conditional_sentences(seed=seed + 211, count=conditional_count):
        add_sentence(sentence, "controlled_conditions")
    for sentence in _schedule_sentences(seed=seed + 307, count=schedule_count):
        add_sentence(sentence, "controlled_schedules")
    if len(pairs) < 500:
        raise RuntimeError(f"Only generated {len(pairs)} valid pairs; need at least 500")
    frame = pd.DataFrame(pairs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        split_frame = frame[frame["split"] == split].copy()
        split_frame.to_csv(output / f"{split}.csv", index=False)
        counts[split] = len(split_frame)
    preview = frame.sample(n=500, random_state=seed)
    audit_dir = ROOT / "results/data_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    preview.to_csv(audit_dir / "synthetic_preview.csv", index=False)
    quality = pd.DataFrame([{
        "examples": len(frame),
        "empty_fragments": int((frame["fragment"].str.len() == 0).sum()),
        "duplicate_targets": int(frame["target"].duplicated().sum()),
        "extremely_short_inputs": int((frame["fragment"].str.split().str.len() < 3).sum()),
        "malformed_punctuation": int(frame["fragment"].str.contains(r"[^A-Z' ]", regex=True).sum()),
        "challenge_gloss_leaks": int(frame["gloss"].isin(challenge_glosses).sum()),
        "challenge_reference_leaks": int(frame["target"].str.casefold().isin(challenge_references).sum()),
    }])
    quality.to_csv(audit_dir / "synthetic_quality_checks.csv", index=False)
    print(preview[["fragment", "target", "operations"]].head(50).to_string(index=False))
    print(f"Generated split counts: {counts}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean telegraphic reconstruction pairs")
    parser.add_argument(
        "--input-files", nargs="+",
        default=[str(ROOT / "datasets/raw/librispeech_pc/train-clean-100.json")],
    )
    parser.add_argument("--output-dir", default=str(ROOT / "datasets/processed/reconstruction"))
    parser.add_argument("--maximum", type=int, default=40_000)
    parser.add_argument("--controlled-count", type=int, default=8_000)
    parser.add_argument("--conditional-count", type=int, default=1_000)
    parser.add_argument("--schedule-count", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_reconstruction_data(
        args.input_files, args.output_dir, seed=args.seed, maximum=args.maximum,
        controlled_count=args.controlled_count, conditional_count=args.conditional_count,
        schedule_count=args.schedule_count,
    )


if __name__ == "__main__":
    main()
