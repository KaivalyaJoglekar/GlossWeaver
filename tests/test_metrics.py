from glossweaver.evaluation.metrics import (
    content_word_recall,
    exact_match,
    prediction_source_copy_ratio,
)


def test_exact_match():
    assert exact_match(["A", " B "], ["A", "B"]) == 1.0
    assert exact_match(["A", "C"], ["A", "B"]) == 0.5


def test_content_preservation_and_copy_ratio_diagnostics():
    gloss = "MOTHER COOK FOOD TONIGHT"
    assert content_word_recall(gloss, "the other looks at food tonight") == 0.5
    assert content_word_recall(gloss, "The mother will cook food tonight.") == 1.0
    assert prediction_source_copy_ratio("WOMAN BUY CAR RED", "women buy car red") == 1.0
