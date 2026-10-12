"""Actual tiny local training and inference; no Hub downloads or research scores."""
from dataclasses import replace
import pytest
from pathlib import Path
from transformers import T5Config, T5ForConditionalGeneration
from scripts.run_smoke_test import tiny_tokenizer, RECORDS
from glossweaver.settings import TrainConfig
from glossweaver.training.train import run_training
from glossweaver.utils import write_jsonl
from glossweaver.inference import Reconstructor

@pytest.mark.parametrize('grammar', [False, True])
def test_local_training_manifest_checkpoint_and_candidate_generation(tmp_path, grammar):
    tokenizer = tiny_tokenizer()
    tokenizer.model_input_names = ['input_ids', 'attention_mask']
    base = tmp_path/'base'
    model = T5ForConditionalGeneration(T5Config(vocab_size=len(tokenizer), d_model=8, d_ff=16, num_layers=1, num_decoder_layers=1, num_heads=2, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
    model.save_pretrained(base); tokenizer.save_pretrained(base)
    train = tmp_path/'train.jsonl'; validation = tmp_path/'validation.jsonl'
    write_jsonl(train, [RECORDS[0]]); write_jsonl(validation, [RECORDS[1]])
    config = TrainConfig(experiment_name='tiny-smoke', model_name=str(base), train_file=train, validation_file=validation, epochs=1, max_steps=1, batch_size=1, gradient_accumulation_steps=1, grammar_enabled=grammar, output_dir=tmp_path/'runs', device='cpu')
    metadata = run_training(config)
    checkpoint = Path(metadata['checkpoint_path'])
    assert (checkpoint/'run_manifest.json').exists()
    assert (checkpoint/'optimizer_state.pt').exists()
    reconstructor = Reconstructor(str(checkpoint), device='cpu')
    candidates, predictions = reconstructor.reconstruct_candidates('ME GO STORE YESTERDAY', num_beams=3, count=3)
    assert 1 <= len(candidates) <= 3
    assert len({c['text'].casefold() for c in candidates}) == len(candidates)
    assert bool(predictions) == grammar
    continued = run_training(replace(config, epochs=2, max_steps=2), resume=checkpoint)
    assert continued['checkpoint_path'] != str(checkpoint)
    assert Path(continued['checkpoint_path']).is_dir()
