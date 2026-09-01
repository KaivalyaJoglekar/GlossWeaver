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

