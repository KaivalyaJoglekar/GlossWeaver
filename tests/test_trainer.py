from types import SimpleNamespace
import torch
import pytest
from glossweaver.training.trainer import train_epochs


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(0.0))
    def forward(self, **batch):
        return SimpleNamespace(loss=(self.weight - batch['value']).square().mean())
    def save_pretrained(self, path):
        pass


def test_final_partial_accumulation_uses_actual_group_size(tmp_path, monkeypatch):
    model = TinyModel()
    monkeypatch.setattr('glossweaver.training.trainer._validation_predictions', lambda *args: (['ok'], ['ok']))
    monkeypatch.setattr('glossweaver.training.trainer.compute_metrics', lambda *args, **kwargs: {'sacrebleu': 100, 'rouge_l': 1, 'meteor': 1})
    tokenizer = SimpleNamespace(save_pretrained=lambda path: None)
    # Three microbatches: final singleton must contribute a full second update.
    train_epochs(model, [{'value': torch.tensor([0.25])}] * 3, [{'value': torch.tensor([0.25])}],
                 torch.optim.SGD(model.parameters(), lr=0.1), tokenizer,
                 device='cpu', epochs=1, output_dir=tmp_path, history_csv=tmp_path/'history.csv', gradient_accumulation_steps=2)
    assert model.weight.item() == pytest.approx(0.09)

def test_resume_restores_optimizer_budget_and_history(tmp_path, monkeypatch):
    monkeypatch.setattr('glossweaver.training.trainer._validation_predictions', lambda *args: (['ok'], ['ok']))
    monkeypatch.setattr('glossweaver.training.trainer.compute_metrics', lambda *args, **kwargs: {'sacrebleu': 100, 'rouge_l': 1, 'meteor': 1})
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    tokenizer = SimpleNamespace(save_pretrained=lambda path: None)
    batch = [{'value': torch.tensor([.25])}]
    first = train_epochs(model, batch, batch, optimizer, tokenizer, device='cpu', epochs=1, output_dir=tmp_path, history_csv=tmp_path/'history.csv')
    state = torch.load(tmp_path/'optimizer_state.pt', weights_only=False)
    assert state['global_step'] == 1
    second = train_epochs(model, batch, batch, optimizer, tokenizer, device='cpu', epochs=2, output_dir=tmp_path/'resumed', history_csv=tmp_path/'resumed/history.csv', resume_state=state)
    assert [r['epoch'] for r in second] == [1,2]
    assert second[-1]['global_step'] == 2
    assert model.weight.item() == pytest.approx(.09)
