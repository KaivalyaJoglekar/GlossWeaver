from __future__ import annotations

from collections import Counter
from typing import Sequence

from glossweaver.data.grammar_labels import LABEL_NAMES, extract_grammar_labels


def grammar_diagnostics(
    glosses: Sequence[str], references: Sequence[str], predictions: Sequence[str]
) -> dict[str, dict[str, float | int]]:
    if not (len(glosses) == len(references) == len(predictions)):
        raise ValueError("Glosses, references, and predictions must have equal length")
    counts = {name: Counter() for name in LABEL_NAMES}
    for gloss, reference, prediction in zip(glosses, references, predictions, strict=True):
        gold = extract_grammar_labels(gloss, reference)
        predicted = extract_grammar_labels(gloss, prediction)
        for name, expected, actual in zip(LABEL_NAMES, gold, predicted, strict=True):
            if expected and actual:
                counts[name]["tp"] += 1
            elif not expected and actual:
                counts[name]["fp"] += 1
            elif expected and not actual:
                counts[name]["fn"] += 1
            else:
                counts[name]["tn"] += 1
    result = {}
    for name, category in counts.items():
        tp, fp, fn = category["tp"], category["fp"], category["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }
    return result

