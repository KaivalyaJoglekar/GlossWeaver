import pytest
from glossweaver.semantics import validate_meaning, align_changes

@pytest.mark.parametrize('source,output,category', [
    ('SHE NOT TAKE MEDICINE', 'She took medicine.', 'negation'),
    ('I have 2 books', 'I have three books.', 'numbers'),
    ('ME GO YESTERDAY', 'I will go tomorrow.', 'temporal'),
    ('APPOINTMENT 2026-10-12', 'Appointment 2026-10-13.', 'dates'),
    ('Alice sees Bob', 'Alice sees John.', 'named_entities'),
    ('BOY WANT FOOD', 'The boy wants.', 'content_omission'),
    ('BOY EAT', 'The boy eats cake.', 'unsupported_addition'),
])
def test_detects_potential_changes(source, output, category):
    assert category in {i['category'] for i in validate_meaning(source, output)['issues']}

@pytest.mark.parametrize('source,output,category', [
    ('SHE NOT TAKE MEDICINE', "She didn't take medication.", 'negation'),
    ('I have 2 books', 'I have two books.', 'numbers'),
    ('ME GO YESTERDAY', 'I went yesterday.', 'temporal'),
    ('DATE 2026-10-12', 'The date is 2026-10-12.', 'dates'),
    ('Alice sees Bob', 'Bob is seen by Alice.', 'named_entities'),
    ('ME BUY CAR', 'I purchased an automobile.', 'content_omission'),
    ('BOY WANT FOOD', 'The boy wants food.', 'unsupported_addition'),
])
def test_valid_grammatical_changes_do_not_trigger_rule(source, output, category):
    assert category not in {i['category'] for i in validate_meaning(source, output)['issues']}

def test_all_caps_names_are_unavailable_and_alignment_is_honest():
    result = validate_meaning('ALICE GO', 'Alice goes.')
    assert next(c for c in result['checks'] if c['category'] == 'named_entities')['status'] == 'unavailable'
    changes = align_changes('BOY WANT FOOD', 'The boy wants food')
    assert changes[0]['category'] == 'added article'
    assert any(c['category'] == 'uncategorized text change' for c in changes)

def test_common_inflections_and_entity_boundaries():
    assert 'content_omission' not in {i['category'] for i in validate_meaning('SHE SEE MAN', 'She sees a man.')['issues']}
    assert 'named_entities' in {i['category'] for i in validate_meaning('Alice sees Bob', 'Alice sees Bobby.')['issues']}
