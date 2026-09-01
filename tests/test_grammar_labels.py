from glossweaver.data.grammar_labels import LABEL_NAMES, extract_grammar_labels, explain_grammar_labels


def test_expected_recovery_categories():
    labels = dict(zip(
        LABEL_NAMES,
        extract_grammar_labels("BOY GO SCHOOL YESTERDAY", "The boy went to school yesterday."),
        strict=True,
    ))
    assert labels["ARTICLE"] == 1
    assert labels["PREPOSITION"] == 1
    assert labels["TENSE_ASPECT"] == 1


def test_all_decisions_have_explanations():
    decisions = explain_grammar_labels("GIRL SIT ROOM", "The girl was sitting in the room.")
    assert [item.label for item in decisions] == list(LABEL_NAMES)
    assert all(item.reasons for item in decisions)

