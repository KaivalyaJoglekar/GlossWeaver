from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration

from glossweaver.data.grammar_labels import LABEL_NAMES
from glossweaver.model_registry import MODELS, ModelInfo, checkpoint_has_weights, get_model
from glossweaver.modeling.baseline import CopyHeuristicBaseline
from glossweaver.modeling.grammar_t5 import GrammarAwareT5
from glossweaver.settings import GenerationConfig
from glossweaver.text_format import format_gloss_input, normalize_gloss
from glossweaver.utils import resolve_device


def _legacy_format(gloss: str) -> str:
    normalized = normalize_gloss(gloss)
    if not normalized:
        raise ValueError("Gloss input is empty")
    return "reconstruct gloss: " + normalized


class Reconstructor:
    def __init__(self, model: str, threshold: float = 0.5, device: str = "auto") -> None:
        self.threshold = threshold
        self.device = resolve_device(device)
        if model in MODELS:
            self.info = get_model(model)
        else:
            path = Path(model).resolve()
            self.info = ModelInfo(
                key=path.name, display_name=path.name, checkpoint_path=path,
                model_type="checkpoint", input_formatter="shared_v1",
                grammar_enabled=False, experiment_name=path.name,
                training_data_description="Direct checkpoint path.",
            )
        self.copy = self.info.model_type == "copy"
        if self.copy:
            self.model = CopyHeuristicBaseline()
            self.tokenizer = None
            self.grammar_aware = False
            return
        checkpoint = self.info.checkpoint_path
        if checkpoint is None or not checkpoint_has_weights(checkpoint):
            raise FileNotFoundError(
                f"Requested trained checkpoint is unavailable or has no weights: {checkpoint}"
            )
        config = AutoConfig.from_pretrained(checkpoint)
        self.grammar_aware = bool(getattr(config, "glossweaver_grammar_aware", False))
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = (
            GrammarAwareT5.from_pretrained(checkpoint)
            if self.grammar_aware else T5ForConditionalGeneration.from_pretrained(checkpoint)
        )
        self.model.to(self.device).eval()

    def formatted_input(self, gloss: str) -> str:
        if self.info.input_formatter == "legacy_reconstruct_gloss":
            return _legacy_format(gloss)
        return format_gloss_input(gloss)

    def reconstruct(
        self, gloss: str, num_beams: int = 4,
        generation: GenerationConfig | None = None,
    ) -> tuple[str, dict[str, float]]:
        if self.copy:
            return self.model.predict(gloss), {}
        settings = generation or GenerationConfig(num_beams=num_beams)
        encoded = self.tokenizer(
            self.formatted_input(gloss), return_tensors="pt", truncation=True, max_length=96,
        ).to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                num_beams=settings.num_beams,
                do_sample=settings.do_sample,
                early_stopping=settings.early_stopping if settings.num_beams > 1 else False,
                no_repeat_ngram_size=settings.no_repeat_ngram_size,
                length_penalty=settings.length_penalty,
                max_new_tokens=settings.max_new_tokens,
            )
            grammar: dict[str, float] = {}
            if self.grammar_aware:
                probabilities = torch.sigmoid(self.model.predict_grammar(**encoded))[0].cpu().tolist()
                names = LABEL_NAMES[:len(probabilities)]
                grammar = dict(zip(names, probabilities, strict=True))
        return self.tokenizer.decode(generated[0], skip_special_tokens=True), grammar

    def reconstruct_batch(
        self, glosses: list[str], num_beams: int = 4,
        generation: GenerationConfig | None = None,
    ) -> list[tuple[str, dict[str, float]]]:
        if not glosses:
            return []
        if self.copy:
            return [(self.model.predict(gloss), {}) for gloss in glosses]
        settings = generation or GenerationConfig(num_beams=num_beams)
        encoded = self.tokenizer(
            [self.formatted_input(gloss) for gloss in glosses], return_tensors="pt",
            padding=True, truncation=True, max_length=96,
        ).to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                **encoded, num_beams=settings.num_beams, do_sample=settings.do_sample,
                early_stopping=settings.early_stopping if settings.num_beams > 1 else False,
                no_repeat_ngram_size=settings.no_repeat_ngram_size,
                length_penalty=settings.length_penalty,
                max_new_tokens=settings.max_new_tokens,
            )
            grammar_rows: list[dict[str, float]] = [{} for _ in glosses]
            if self.grammar_aware:
                probabilities = torch.sigmoid(self.model.predict_grammar(**encoded)).cpu().tolist()
                grammar_rows = [
                    dict(zip(LABEL_NAMES[:len(values)], values, strict=True))
                    for values in probabilities
                ]
        predictions = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return list(zip(predictions, grammar_rows, strict=True))

    @property
    def debug_info(self) -> dict[str, object]:
        return self.info.debug_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct English from gloss-like input")
    parser.add_argument("--model", choices=sorted(MODELS))
    parser.add_argument("--checkpoint")
    parser.add_argument("--text", required=True)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if bool(args.model) == bool(args.checkpoint):
        parser.error("Provide exactly one of --model or --checkpoint")
    reconstructor = Reconstructor(args.model or args.checkpoint, device=args.device)
    prediction, grammar = reconstructor.reconstruct(args.text, num_beams=args.num_beams)
    print(f"Input: {args.text}")
    print(f"Prediction: {prediction}")
    print(f"Checkpoint: {reconstructor.debug_info['checkpoint_path']}")
    if grammar:
        print("Grammar: " + ", ".join(f"{key}={value:.3f}" for key, value in grammar.items()))


if __name__ == "__main__":
    main()
