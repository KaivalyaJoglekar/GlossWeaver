from glossweaver.evaluation.metrics import exact_match


def test_exact_match():
    assert exact_match(["A", " B "], ["A", "B"]) == 1.0
    assert exact_match(["A", "C"], ["A", "B"]) == 0.5

