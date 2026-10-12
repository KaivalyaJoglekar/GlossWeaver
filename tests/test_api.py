from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_copy_inference_and_readiness_are_explicit():
    response = client.post('/api/infer', json={'gloss': 'ME GO STORE YESTERDAY', 'checkpoint': 'e0_copy', 'threshold': 0})
    assert response.status_code == 200
    body = response.json()
    assert body['checkpoint'] == 'e0_copy'
    assert not body['grammar_aware']
    assert 'Copy baseline' in body['notices'][0]
    assert body['elapsed_ms'] >= 0
    assert 'checkpoint_path' not in body['debug']


def test_invalid_inference_inputs_fail_validation():
    for override in ({'gloss': ' '}, {'gloss': 'X' * 2001}, {'num_beams': 0}, {'num_beams': 1000}, {'threshold': -1}, {'threshold': 2}):
        response = client.post('/api/infer', json={'gloss': 'ME GO', 'checkpoint': 'e0_copy', **override})
        assert response.status_code == 422


def test_metrics_contract_and_filters():
    response = client.get('/api/metrics?dataset=aslg_pc12&split=test_500')
    assert response.status_code == 200
    body = response.json()
    assert body['rows']
    assert all(r['dataset'] == 'aslg_pc12' and r['examples'] == 500 for r in body['rows'])
    assert all('bleu' in r and 'display_name' in r and 'checkpoint' not in r for r in body['rows'])
    assert 'grammar_diagnostics' in body['rows'][0]
    assert client.get('/api/metrics/e4_glossweaver').status_code == 200


def test_unknown_model_does_not_fallback():
    assert client.post('/api/infer', json={'gloss': 'ME GO', 'checkpoint': 'missing'}).status_code == 400


def test_busy_model_returns_retryable_error():
    from app.api import _model_lock
    with _model_lock:
        response = client.post('/api/infer', json={'gloss': 'ME GO', 'checkpoint': 'e0_copy'})
    assert response.status_code == 503
    assert response.headers['Retry-After'] == '2'


def test_pretrained_model_is_labeled_separately(tmp_path, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from app import api
    info = replace(api.MODELS['pretrained_flan_t5'], checkpoint_path=tmp_path)
    (tmp_path / 'model.safetensors').write_bytes(b'fixture')
    monkeypatch.setitem(api.MODELS, info.key, info)
    monkeypatch.setattr(api, '_get_reconstructor', lambda key: SimpleNamespace(
        reconstruct=lambda *args, **kwargs: ('I went to the store yesterday.', {}),
        grammar_aware=False))
    response = client.post('/api/infer', json={'gloss': 'ME GO STORE YESTERDAY', 'checkpoint': info.key})
    assert response.status_code == 200
    assert response.json()['debug']['model_type'] == 'pretrained'
    assert 'not been fine-tuned' in response.json()['notices'][0]
