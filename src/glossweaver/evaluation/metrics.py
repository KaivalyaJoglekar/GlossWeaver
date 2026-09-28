from __future__ import annotations

import re
from typing import Sequence

from glossweaver.data.grammar_labels import PRONOUNS, simple_lemma


FUNCTION_WORDS = {
    "a", "an", "the", "to", "of", "in", "on", "at", "by", "for", "from",
    "with", "is", "are", "was", "were", "be", "been", "being", "has", "have",
    "had", "do", "does", "did", "will", "would", "can", "could", "should",
    "x", "desc", "poss", "re", "se",
} | PRONOUNS


def _concepts(text: str) -> list[str]:
    return [
        simple_lemma(token.lower())
        for token in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
        if token.lower() not in FUNCTION_WORDS
    ]


def content_word_recall(gloss: str, prediction: str) -> float:
    source = set(_concepts(gloss))
    predicted = set(_concepts(prediction))
    return len(source & predicted) / len(source) if source else 1.0


def prediction_source_copy_ratio(gloss: str, prediction: str) -> float:
    source = set(_concepts(gloss))
    predicted = _concepts(prediction)
    return sum(token in source for token in predicted) / len(predicted) if predicted else 0.0


def exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have equal length")
    if not references:
        return 0.0
    return sum(
        prediction.strip() == reference.strip()
        for prediction, reference in zip(predictions, references, strict=True)
    ) / len(references)


def compute_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    include_bertscore: bool = True,
    bertscore_model: str = "distilbert-base-uncased",
    glosses: Sequence[str] | None = None,
) -> dict[str, float | str]:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have equal length")
    if not predictions:
        raise ValueError("Cannot evaluate an empty prediction set")
    import sacrebleu
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    bleu = sacrebleu.corpus_bleu(list(predictions), [list(references)])
    chrf = sacrebleu.corpus_chrf(list(predictions), [list(references)])
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = sum(
        rouge.score(reference, prediction)["rougeL"].fmeasure
        for prediction, reference in zip(predictions, references, strict=True)
    ) / len(predictions)
    meteor = sum(
        meteor_score([reference.split()], prediction.split())
        for prediction, reference in zip(predictions, references, strict=True)
    ) / len(predictions)
    result: dict[str, float | str] = {
        "sacrebleu": float(bleu.score),
        "sacrebleu_signature": str(bleu),
        "chrf": float(chrf.score),
        "rouge_l": float(rouge_l),
        "meteor": float(meteor),
        "exact_match": float(exact_match(predictions, references)),
    }
    if glosses is not None:
        if len(glosses) != len(predictions):
            raise ValueError("Glosses and predictions must have equal length")
        result["content_word_recall"] = sum(
            content_word_recall(gloss, prediction)
            for gloss, prediction in zip(glosses, predictions, strict=True)
        ) / len(predictions)
        result["copy_ratio"] = sum(
            prediction_source_copy_ratio(gloss, prediction)
            for gloss, prediction in zip(glosses, predictions, strict=True)
        ) / len(predictions)
    if include_bertscore:
        from bert_score import score as bert_score

        _, _, f1 = bert_score(
            list(predictions), list(references),
            model_type=bertscore_model, lang="en", verbose=False,
        )
        result["bertscore_f1"] = float(f1.mean().item())
        result["bertscore_model"] = bertscore_model
    else:
        result["bertscore_status"] = "skipped by explicit command-line option"
    return result


def compute_multi_reference_metrics(
    predictions: Sequence[str], reference_sets: Sequence[Sequence[str]], *,
    include_bertscore: bool = True,
    bertscore_model: str = "distilbert-base-uncased",
    glosses: Sequence[str] | None = None,
) -> dict[str, float | str]:
    if len(predictions) != len(reference_sets) or not predictions:
        raise ValueError("Predictions and non-empty reference sets must have equal length")
    import sacrebleu
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    width = max(len(refs) for refs in reference_sets)
    streams = [
        [refs[min(index, len(refs) - 1)] for refs in reference_sets]
        for index in range(width)
    ]
    bleu = sacrebleu.corpus_bleu(list(predictions), streams)
    chrf = sacrebleu.corpus_chrf(list(predictions), streams)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = sum(
        max(rouge.score(reference, prediction)["rougeL"].fmeasure for reference in refs)
        for prediction, refs in zip(predictions, reference_sets, strict=True)
    ) / len(predictions)
    meteor = sum(
        max(meteor_score([reference.split()], prediction.split()) for reference in refs)
        for prediction, refs in zip(predictions, reference_sets, strict=True)
    ) / len(predictions)
    result: dict[str, float | str] = {
        "sacrebleu": float(bleu.score), "sacrebleu_signature": str(bleu),
        "chrf": float(chrf.score), "rouge_l": float(rouge_l),
        "meteor": float(meteor),
        "exact_match": sum(
            prediction.strip() in {reference.strip() for reference in refs}
            for prediction, refs in zip(predictions, reference_sets, strict=True)
        ) / len(predictions),
    }
    if glosses is not None:
        result["content_word_recall"] = sum(
            content_word_recall(gloss, prediction)
            for gloss, prediction in zip(glosses, predictions, strict=True)
        ) / len(predictions)
        result["copy_ratio"] = sum(
            prediction_source_copy_ratio(gloss, prediction)
            for gloss, prediction in zip(glosses, predictions, strict=True)
        ) / len(predictions)
    if include_bertscore:
        from bert_score import score as bert_score
        flat_predictions = [prediction for prediction, refs in zip(predictions, reference_sets, strict=True) for _ in refs]
        flat_references = [reference for refs in reference_sets for reference in refs]
        _, _, f1 = bert_score(
            flat_predictions, flat_references, model_type=bertscore_model,
            lang="en", verbose=False,
        )
        cursor = 0
        best = []
        for refs in reference_sets:
            best.append(float(f1[cursor:cursor + len(refs)].max().item()))
            cursor += len(refs)
        result["bertscore_f1"] = sum(best) / len(best)
        result["bertscore_model"] = bertscore_model
    else:
        result["bertscore_status"] = "skipped by explicit command-line option"
    return result
