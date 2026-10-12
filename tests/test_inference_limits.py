import pytest
import torch
from types import SimpleNamespace
from glossweaver.inference import Reconstructor


def test_token_overflow_is_rejected_before_generation():
    reconstructor = Reconstructor.__new__(Reconstructor)
    reconstructor.max_source_length = 3
    reconstructor.info = SimpleNamespace(input_formatter='shared_v1')
    reconstructor.device = 'cpu'
    reconstructor.tokenizer = lambda *args, **kwargs: {'attention_mask': torch.ones((1, 4))}
    with pytest.raises(ValueError, match='no text was discarded'):
        reconstructor._encode(['FOUR TOKENS'])


def test_pretrained_prompt_is_separate_from_fine_tuned_formatter():
    from glossweaver.model_registry import MODELS
    r = Reconstructor.__new__(Reconstructor)
    r.info = MODELS['pretrained_flan_t5']
    assert r.formatted_input('ME GO STORE YESTERDAY') == 'Make a complete English sentence from these words: me go store yesterday'
    assert r.formatted_input('Alice go store').endswith('Alice go store')
    r.info = MODELS['e2_flan']
    assert r.formatted_input('ME GO STORE').endswith('ME GO STORE')
