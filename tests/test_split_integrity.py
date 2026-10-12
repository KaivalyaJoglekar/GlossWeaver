import pytest
from glossweaver.data.split_integrity import group_split, assert_source_isolation, stratified_sample
from glossweaver.data.flores_asl import prepare_flores
from glossweaver.utils import write_jsonl, read_jsonl


def test_alternative_targets_share_one_partition():
    records = [{'id': str(i), 'gloss': f'PERSON GO PLACE {i // 2}', 'target': str(i)} for i in range(100)]
    split = group_split(records)
    assert split == group_split(records)
    groups = {s: [r for r in split if r['split'] == s] for s in ('train', 'validation', 'test')}
    assert_source_isolation(groups['train'], groups['validation'], groups['test'])
    with pytest.raises(ValueError, match='Source overlap'):
        assert_source_isolation([{'gloss': 'ME GO'}], [{'gloss': 'me-go'}])


def test_sample_covers_all_origins_instead_of_file_prefix():
    records = [{'id': f'{origin}{i}', 'gloss': str(i), 'corpus_origin': origin} for origin in ['public', 'controlled'] for i in range(20)]
    sample = stratified_sample(records, 10)
    assert {r['corpus_origin'] for r in sample} == {'public', 'controlled'}
    assert sample == stratified_sample(records, 10)


def test_flores_preserves_notation_and_protects_devtest(tmp_path):
    raw = tmp_path / 'raw'; raw.mkdir()
    rows = [{'id': 1, 'gloss': 'IX-3 NOT-YET #BOOK cl:5', 'sentence': 'She has not read the book.', 'signer': 0}]
    write_jsonl(raw / 'devtest.jsonl', rows)
    write_jsonl(raw / 'dev.jsonl', rows)
    counts = prepare_flores(raw, tmp_path / 'processed')
    assert counts == {'devtest': 1, 'dev': 0}
    assert next(read_jsonl(tmp_path / 'processed/devtest.jsonl'))['gloss'] == rows[0]['gloss']
