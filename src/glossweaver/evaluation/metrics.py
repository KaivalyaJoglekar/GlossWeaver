from __future__ import annotations

from typing import Sequence


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

