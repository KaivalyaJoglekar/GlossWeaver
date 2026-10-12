from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration

from glossweaver.data.grammar_labels import LABEL_NAMES
from glossweaver.model_registry import MODELS, ModelInfo, checkpoint_has_weights, get_model, resolve_checkpoint
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
        checkpoint = resolve_checkpoint(self.info.checkpoint_path) if self.info.checkpoint_path else None
        if checkpoint:
            from dataclasses import replace
            self.info = replace(self.info, checkpoint_path=checkpoint)
        if checkpoint is None or not checkpoint_has_weights(checkpoint):
            raise FileNotFoundError(
                f"Requested trained checkpoint is unavailable or has no weights: {checkpoint}"
            )
        config = AutoConfig.from_pretrained(checkpoint)
        self.max_source_length = int(getattr(config, "glossweaver_max_source_length", 96))
        self.max_target_length = int(getattr(config, "glossweaver_max_target_length", 96))
        self.grammar_aware = bool(getattr(config, "glossweaver_grammar_aware", False))
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = (
            GrammarAwareT5.from_pretrained(checkpoint)
            if self.grammar_aware else T5ForConditionalGeneration.from_pretrained(checkpoint)
        )
        self.model.to(self.device).eval()

    def formatted_input(self, gloss: str) -> str:
        if self.info.input_formatter == "flan_grammar_v1":
            text = normalize_gloss(gloss)
            # All-capital glosses otherwise tend to be copied by the base model.
            text = text.lower() if text.isupper() else text
            return "Make a complete English sentence from these words: " + text
        if self.info.input_formatter == "legacy_reconstruct_gloss":
            return _legacy_format(gloss)
        return format_gloss_input(gloss)

    def reconstruct(
        self, gloss: str, num_beams: int = 4,
        generation: GenerationConfig | None = None,
    ) -> tuple[str, dict[str, float]]:
        self._validate_gloss(gloss)
        if self.copy:
            return self.model.predict(gloss), {}
        settings = generation or GenerationConfig(num_beams=num_beams)
        encoded = self._encode([gloss])
        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                num_beams=settings.num_beams,
                do_sample=settings.do_sample,
                early_stopping=settings.early_stopping if settings.num_beams > 1 else False,
                no_repeat_ngram_size=settings.no_repeat_ngram_size,
                length_penalty=settings.length_penalty,
                max_new_tokens=min(settings.max_new_tokens, self.max_target_length),
            )
            grammar: dict[str, float] = {}
            if self.grammar_aware:
                probabilities = torch.sigmoid(self.model.predict_grammar(**encoded))[0].cpu().tolist()
                names = LABEL_NAMES[:len(probabilities)]
                grammar = dict(zip(names, probabilities, strict=True))
        return self.tokenizer.decode(generated[0], skip_special_tokens=True), grammar

    def reconstruct_candidates(self, gloss: str, num_beams: int = 4, count: int = 3) -> tuple[list[dict], dict[str, float]]:
        """Return unique beam candidates, ranked by advisory warnings then beam order."""
        from glossweaver.semantics import validate_meaning
        if not 1 <= count <= num_beams <= 10:
            raise ValueError("Candidate count must be between 1 and beam width (at most 10)")
        self._validate_gloss(gloss)
        if self.copy:
            return [{"text": self.model.predict(gloss), "beam_rank": 1, "semantic": validate_meaning(gloss, self.model.predict(gloss))}], {}
        settings = GenerationConfig(num_beams=num_beams)
        encoded = self._encode([gloss])
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded, num_beams=num_beams, num_return_sequences=count,
                do_sample=False, early_stopping=num_beams > 1,
                no_repeat_ngram_size=settings.no_repeat_ngram_size,
                length_penalty=settings.length_penalty,
                max_new_tokens=min(settings.max_new_tokens, self.max_target_length),
            )
            grammar = {}
            if self.grammar_aware:
                values = torch.sigmoid(self.model.predict_grammar(**encoded))[0].cpu().tolist()
                grammar = dict(zip(LABEL_NAMES[:len(values)], values, strict=True))
        seen, candidates = set(), []
        for rank, text in enumerate(self.tokenizer.batch_decode(generated, skip_special_tokens=True), 1):
            key = " ".join(text.casefold().split())
            if not key or key in seen:
                continue
            seen.add(key)
            candidates.append({"text": text, "beam_rank": rank, "semantic": validate_meaning(gloss, text)})
        if not candidates:
            raise ValueError("The model generated no nonempty candidates")
        candidates.sort(key=lambda c: (sum(i["severity"] == "warning" for i in c["semantic"]["issues"]), c["beam_rank"]))
        return candidates, grammar

    def reconstruct_batch(
        self, glosses: list[str], num_beams: int = 4,
        generation: GenerationConfig | None = None,
    ) -> list[tuple[str, dict[str, float]]]:
        if not glosses:
            return []
        for gloss in glosses:
            self._validate_gloss(gloss)
        if self.copy:
            return [(self.model.predict(gloss), {}) for gloss in glosses]
        settings = generation or GenerationConfig(num_beams=num_beams)
        encoded = self._encode(glosses)
        with torch.no_grad():
            generated = self.model.generate(
                **encoded, num_beams=settings.num_beams, do_sample=settings.do_sample,
                early_stopping=settings.early_stopping if settings.num_beams > 1 else False,
                no_repeat_ngram_size=settings.no_repeat_ngram_size,
                length_penalty=settings.length_penalty,
                max_new_tokens=min(settings.max_new_tokens, self.max_target_length),
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

    @staticmethod
    def _validate_gloss(gloss: str) -> None:
        if not gloss.strip():
            raise ValueError("Input is empty")
        if len(gloss) > 2000:
            raise ValueError("Input must be at most 2,000 characters")

    def _encode(self, glosses: list[str]):
        texts = [self.formatted_input(gloss) for gloss in glosses]
        encoded = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
        encoded.pop("token_type_ids", None)  # T5 has no segment embeddings.
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        if any(length > self.max_source_length for length in lengths):
            raise ValueError(
                f"Input exceeds this model's {self.max_source_length}-token limit "
                "(including the instruction). Shorten the input; no text was discarded."
            )
        return encoded.to(self.device)

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
