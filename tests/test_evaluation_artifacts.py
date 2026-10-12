import json
import pandas as pd
from glossweaver.evaluation import evaluate
from glossweaver.model_registry import resolve_checkpoint
from glossweaver.utils import write_jsonl


def test_evaluations_keep_separate_snapshots_and_sample_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, 'ROOT', tmp_path)
    scores = dict(sacrebleu=10, rouge_l=.2, meteor=.3, content_word_recall=1,
                  copy_ratio=1, chrf=20, exact_match=0, sacrebleu_signature='test')
    monkeypatch.setattr(evaluate, 'compute_metrics', lambda *args, **kwargs: scores)
    monkeypatch.setattr(evaluate, 'grammar_diagnostics', lambda *args: {})
    source = tmp_path / 'data.jsonl'
    write_jsonl(source, [{'id': str(i), 'gloss': 'ME GO STORE', 'target': 'I go to the store.',
                         'corpus_origin': 'controlled' if i % 2 else 'public'} for i in range(10)])
    rows = [evaluate.evaluate_model('e0_copy', source, tmp_path / 'out.csv',
            dataset_name='fixture', split='dev', max_examples=4, include_bertscore=False)
            for _ in range(2)]
    assert rows[0]['evaluation_id'] != rows[1]['evaluation_id']
    assert rows[0]['sample_sha256'] == rows[1]['sample_sha256']
    assert len(pd.read_csv(tmp_path / 'results/metrics/model_metrics.csv')) == 2
    for row in rows:
        folder = tmp_path / 'results/evaluations' / row['evaluation_id']
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert len(manifest['ids']) == 4
        assert pd.read_csv(folder / 'predictions.csv').id.astype(str).tolist() == manifest['ids']
        assert manifest['metrics']['checkpoint_sha256'] == 'copy-baseline'


def test_latest_pointer_requires_weights_and_cannot_escape_parent(tmp_path):
    parent = tmp_path / 'model'; run = parent / 'run'; run.mkdir(parents=True)
    marker = parent / 'latest.json'
    marker.write_text(json.dumps({'run_directory': 'run'}))
    assert resolve_checkpoint(parent) == parent
    (run / 'model.safetensors').write_bytes(b'fixture')
    assert resolve_checkpoint(parent) == run.resolve()
    outside = tmp_path / 'outside'; outside.mkdir()
    (outside / 'model.safetensors').write_bytes(b'fixture')
    marker.write_text(json.dumps({'run_directory': '../outside'}))
    assert resolve_checkpoint(parent) == parent


def test_completed_deployment_and_study_checkpoints_are_selected_before_pretrained(tmp_path, monkeypatch):
    from dataclasses import replace
    from glossweaver.model_registry import MODELS, default_available_model
    (tmp_path / 'model.safetensors').write_bytes(b'fixture')
    monkeypatch.setitem(MODELS, 'reconstruction-flan', replace(MODELS['reconstruction-flan'], checkpoint_path=tmp_path / 'missing'))
    monkeypatch.setitem(MODELS, 'study-full', replace(MODELS['study-full'], checkpoint_path=tmp_path))
    monkeypatch.setitem(MODELS, 'balanced-flan', replace(MODELS['balanced-flan'], checkpoint_path=tmp_path))
    assert default_available_model().key == 'balanced-flan'
    monkeypatch.setitem(MODELS, 'balanced-flan', replace(MODELS['balanced-flan'], checkpoint_path=tmp_path / 'missing'))
    assert default_available_model().key == 'study-full'
