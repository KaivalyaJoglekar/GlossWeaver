from glossweaver.data.build_reconstruction_data import _controlled_sentences, reconstruct_fragment


def test_reconstruction_never_stems_non_verbs_by_suffix():
    pair = reconstruct_fragment("This analysis is important tonight.", seed=42)
    assert pair is not None
    assert "ANALYSIS" in pair.fragment
    assert "THIS" in pair.fragment


def test_controlled_adjective_order_and_irregular_verb_lemma():
    pair = reconstruct_fragment("A woman bought a red vehicle.", seed=42)
    assert pair is not None
    assert pair.fragment == "WOMAN BUY VEHICLE RED"
    assert "WORD_ORDER" in pair.operations


def test_controlled_compositions_are_varied_and_do_not_copy_challenge():
    sentences = list(_controlled_sentences(seed=42, count=500))
    assert len(set(sentences)) == 500
    assert "The mother will cook food tonight." not in sentences


def test_subject_modifier_can_be_recovered_from_tail_position():
    pair = reconstruct_fragment("The strong player threw the ball.", seed=42)
    assert pair is not None
    assert pair.fragment == "PLAYER THROW BALL STRONG"


def test_argument_order_and_adverbial_outside_are_preserved():
    pair = reconstruct_fragment("The tutor explained algebra to the pupil.", seed=42)
    assert pair is not None
    assert pair.fragment == "TUTOR EXPLAIN PUPIL ALGEBRA"
    outside = reconstruct_fragment("The girl plays outside on a sunny day.", seed=42)
    assert outside is not None
    assert outside.fragment == "GIRL PLAY OUTSIDE DAY SUNNY"
    conditional = reconstruct_fragment("The girl plays outside when it is sunny.", seed=42)
    assert conditional is not None
    assert conditional.fragment == "GIRL PLAY OUTSIDE SUNNY"
    schedule = reconstruct_fragment("The girl goes to the school in the morning.", seed=42)
    assert schedule is not None
    assert schedule.fragment == "GIRL GO SCHOOL MORNING"
