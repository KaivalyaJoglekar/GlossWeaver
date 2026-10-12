from glossweaver.data.corruption import corrupt_sentence


def test_corruption_is_reproducible_and_nonempty():
    first = corrupt_sentence("The children were playing in the garden.", seed=42)
    second = corrupt_sentence("The children were playing in the garden.", seed=42)
    assert first == second
    assert first.fragment
    assert first.fragment == first.fragment.upper()


def test_never_drops_every_token():
    result = corrupt_sentence("The in was.", seed=3, probability=1.0)
    assert result.fragment



def test_corruption_preserves_nonverbs_negation_and_numbers():
    result = corrupt_sentence('This analysis is important tonight.', seed=42, probability=1)
    assert 'THIS' in result.fragment and 'ANALYSIS' in result.fragment
    negative = corrupt_sentence('The boy did not buy 3 books.', seed=42, probability=1)
    assert 'NOT' in negative.fragment and '3' in negative.fragment

def test_configurable_corruptions_preserve_modality_and_negation():
    from glossweaver.data.corruption import corrupt_sentence, TRANSFORMATIONS
    probabilities = {operation: 0 for operation in TRANSFORMATIONS}
    result = corrupt_sentence('She will not go tomorrow.', seed=4, probabilities=probabilities, word_order=True)
    assert 'WILL' in result.fragment and 'NOT' in result.fragment
    result = corrupt_sentence('She will not go tomorrow.', seed=4, probability=1, word_order=True)
    assert result.fragment.startswith('TOMORROW')
    assert 'WILL' in result.fragment and 'NOT' in result.fragment
