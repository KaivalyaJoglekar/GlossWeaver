from fastapi.testclient import TestClient
from app.api import app
from glossweaver.data.split_integrity import group_split, assert_source_isolation
import pytest

client = TestClient(app)

def test_compare_partial_failure_and_validation():
    response = client.post('/api/compare', json={'text': 'ME NOT GO', 'models': ['e0_copy', 'does-not-exist']})
    assert response.status_code == 200
    results = response.json()['results']
    assert results[0]['status'] == 'ok'
    assert results[0]['result']['semantic']['status'] == 'no_issue_detected'
    assert results[1]['status'] == 'error'
    assert results[1]['status_code'] == 400
    assert client.post('/api/validate', json={'source': 'NOT GO', 'generated': 'I go.'}).json()['issues'][0]['category'] == 'negation'
    assert client.post('/api/validate', json={'source': ' ', 'generated': 'x'}).status_code == 422

def test_candidate_schema_and_copy_baseline():
    assert client.post('/api/infer', json={'text': 'ME GO', 'model_key': 'e0_copy', 'num_beams': 1, 'alternatives': 2}).status_code == 422
    response = client.post('/api/infer', json={'text': 'ME GO', 'model_key': 'e0_copy', 'num_beams': 4, 'alternatives': 3})
    assert response.status_code == 200
    assert response.json()['alternatives'] == []
    assert response.json()['model_identifier'] == 'e0_copy'

def test_registry_health_and_real_evaluations():
    assert client.get('/api/health').json()['status'] == 'ok'
    assert all('unavailability_reason' in m for m in client.get('/api/models').json()['models'])
    result = client.get('/api/evaluations').json()
    assert len(result['experiments']) == 4
    assert client.get('/api/evaluations/nonexistent/examples').status_code == 404
    assert client.get('/api/evaluations/nonexistent/examples?limit=501').status_code == 422

def test_variants_of_same_target_cannot_leak():
    records = [{'id': str(i), 'gloss': f'VARIANT {i}', 'target': f'Original sentence {i // 3}'} for i in range(60)]
    split = group_split(records)
    assert split == group_split(list(reversed(records))) or {r['id']: r['split'] for r in split} == {r['id']: r['split'] for r in group_split(list(reversed(records)))}
    assert_source_isolation(*[[r for r in split if r['split'] == name] for name in ('train', 'validation', 'test')])
    with pytest.raises(ValueError, match='Source overlap'):
        assert_source_isolation([records[0]], [records[1]])

def test_masked_grammar_loss_remains_finite():
    import torch
    from transformers import T5Config
    from glossweaver.modeling.grammar_t5 import GrammarAwareT5
    model = GrammarAwareT5(T5Config(vocab_size=16, d_model=8, d_ff=16, num_layers=1, num_decoder_layers=1, num_heads=2, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
    output = model(input_ids=torch.tensor([[3,4]]), labels=torch.tensor([[5,1]]), grammar_labels=torch.full((1,6), float('nan')))
    assert output.grammar_loss.item() == 0
    assert torch.isfinite(output.loss)

def test_integrity_audit_reports_lineage_and_near_duplicates():
    from glossweaver.data.integrity_report import audit_records
    report = audit_records([
        {'id':'a', 'gloss':'ME GO', 'target':'I go to school today', 'split':'train'},
        {'id':'b', 'gloss':'I GO', 'target':'I go to school today', 'split':'test'},
        {'id':'c', 'gloss':'I GO NOW', 'target':'I go to school today.', 'split':'train'},
    ])
    assert report['cross_split_lineage_keys']
    assert report['examples'] == 3

def test_request_limit_and_blank_compare_return_explicit_errors():
    assert client.post('/api/validate', content='x'*40000, headers={'Content-Type':'application/json'}).status_code == 413
    assert client.post('/api/compare', json={'text':' ', 'models':['e0_copy']}).status_code == 422

def test_blinded_human_scores_are_kept_separate(tmp_path):
    import pandas as pd
    import json
    from glossweaver.evaluation.human import prepare, summarize, RATINGS
    source = tmp_path/'predictions.csv'
    pd.DataFrame([{'id':'a', 'gloss':'ME GO', 'reference':'I go.', 'prediction':'I go.', 'model':'fixture'}]).to_csv(source, index=False)
    prepare([source], tmp_path/'human')
    ratings = tmp_path/'human/ratings.csv'
    frame = pd.read_csv(ratings)
    assert 'model' not in frame
    with pytest.raises(ValueError):
        summarize(ratings, tmp_path/'summary.json')
    for name in RATINGS:
        frame[name] = 3
    frame['rater'] = 'reviewer-1'
    frame.to_csv(ratings,index=False)
    summarize(ratings, tmp_path/'summary.json')
    assert json.loads((tmp_path/'summary.json').read_text())['kind'] == 'human_evaluation'
