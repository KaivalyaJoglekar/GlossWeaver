import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from glossweaver.modeling.grammar_t5 import GrammarAwareT5


def test_grammar_model_forward_and_gloss_only_generation():
    config = transformers.T5Config(
        vocab_size=32, d_model=16, d_ff=32, num_layers=1, num_decoder_layers=1,
        num_heads=2, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
    )
    model = GrammarAwareT5(config)
    input_ids = torch.tensor([[4, 5, 6, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    labels = torch.tensor([[7, 8, 1]])
    grammar_labels = torch.tensor([[1, 0, 0, 1, 1, 0]], dtype=torch.float)
    output = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, grammar_labels=grammar_labels,
    )
    assert output.loss is not None
    assert output.grammar_logits.shape == (1, 6)
    generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=2)
    assert generated.shape[0] == 1

