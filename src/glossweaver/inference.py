from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration

from glossweaver.data.grammar_labels import LABEL_NAMES
from glossweaver.modeling.baseline import CopyHeuristicBaseline
from glossweaver.modeling.grammar_t5 import GrammarAwareT5
from glossweaver.utils import resolve_device


class Reconstructor:
    def __init__(self, checkpoint: str, threshold: float = 0.5, device: str = "auto") -> None:
        self.checkpoint = checkpoint
        self.threshold = threshold
        self.device = resolve_device(device)
        self.copy = checkpoint == "e0-copy"
        if self.copy:
            self.model = CopyHeuristicBaseline()
            self.tokenizer = None
            self.grammar_aware = False
            return
        config = AutoConfig.from_pretrained(checkpoint)
        self.grammar_aware = bool(getattr(config, "glossweaver_grammar_aware", False))
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = (
            GrammarAwareT5.from_pretrained(checkpoint)
            if self.grammar_aware else T5ForConditionalGeneration.from_pretrained(checkpoint)
        )
        self.model.to(self.device).eval()

    def reconstruct(self, gloss: str, num_beams: int = 4) -> tuple[str, dict[str, float]]:
        if self.copy:
            return self.model.predict(gloss), {}
        encoded = self.tokenizer(
            "reconstruct gloss: " + gloss,
            return_tensors="pt", truncation=True, max_length=128,
        ).to(self.device)
        with torch.no_grad():
            generated = self.model.generate(**encoded, num_beams=num_beams, max_new_tokens=128)
            grammar = {}
            if self.grammar_aware:
                probabilities = torch.sigmoid(self.model(**encoded).grammar_logits)[0].cpu().tolist()
                grammar = dict(zip(LABEL_NAMES, probabilities, strict=True))
        return self.tokenizer.decode(generated[0], skip_special_tokens=True), grammar


def display(gloss: str, prediction: str, grammar: dict[str, float], threshold: float) -> None:
    print(f"Input:\n{gloss}\n\nReconstructed:\n{prediction}")
    if grammar:
        active = [f"{name} ({value:.3f})" for name, value in grammar.items() if value >= threshold]
        print("\nPredicted recovery categories:\n" + ("\n".join(active) if active else "None above threshold"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct fluent English from gloss-only input")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if not args.text and not args.interactive:
        parser.error("Provide --text or --interactive")
    reconstructor = Reconstructor(args.checkpoint, args.threshold, args.device)
    if args.text:
        prediction, grammar = reconstructor.reconstruct(args.text)
        display(args.text, prediction, grammar, args.threshold)
    if args.interactive:
        while True:
            try:
                gloss = input("gloss> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not gloss:
                continue
            prediction, grammar = reconstructor.reconstruct(gloss)
            display(gloss, prediction, grammar, args.threshold)


if __name__ == "__main__":
    main()
