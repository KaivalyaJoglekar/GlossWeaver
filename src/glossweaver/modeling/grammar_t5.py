from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


@dataclass
class GrammarAwareSeq2SeqOutput(Seq2SeqLMOutput):
    generation_loss: torch.FloatTensor | None = None
    grammar_loss: torch.FloatTensor | None = None
    grammar_logits: torch.FloatTensor | None = None


class GrammarAwareT5(T5ForConditionalGeneration):
    """T5 with a training-only multi-label objective over shared encoder states."""

    def __init__(
        self,
        config,
        num_grammar_labels: int | None = None,
        lambda_grammar: float | None = None,
    ):
        super().__init__(config)
        self.num_grammar_labels = int(
            num_grammar_labels if num_grammar_labels is not None
            else getattr(config, "num_grammar_labels", 6)
        )
        self.lambda_grammar = float(
            lambda_grammar if lambda_grammar is not None
            else getattr(config, "lambda_grammar", 0.25)
        )
        self.grammar_head = nn.Linear(config.d_model, self.num_grammar_labels)
        self.grammar_loss_fn = nn.BCEWithLogitsLoss()
        self.post_init()

    def set_grammar_lambda(self, value: float) -> None:
        if value < 0:
            raise ValueError("lambda_grammar must be non-negative")
        self.lambda_grammar = value

    def _pool_encoder(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_state.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def predict_grammar(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """Return grammar logits from the encoder without invoking the decoder."""
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        pooled = self._pool_encoder(encoder_outputs.last_hidden_state, attention_mask)
        return self.grammar_head(pooled)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        grammar_labels: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], GrammarAwareSeq2SeqOutput]:
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        elif not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0])

        pooled = self._pool_encoder(encoder_outputs.last_hidden_state, attention_mask)
        grammar_logits = self.grammar_head(pooled)
        outputs = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        grammar_loss = None
        total_loss = outputs.loss
        if grammar_labels is not None:
            grammar_loss = self.grammar_loss_fn(
                grammar_logits.float(), grammar_labels.to(grammar_logits.dtype)
            )
            total_loss = grammar_loss * self.lambda_grammar if total_loss is None else total_loss + self.lambda_grammar * grammar_loss
        if not return_dict:
            values = outputs.to_tuple()
            return ((total_loss,) + values[1:] + (grammar_logits,)) if total_loss is not None else values + (grammar_logits,)
        return GrammarAwareSeq2SeqOutput(
            loss=total_loss,
            generation_loss=outputs.loss,
            grammar_loss=grammar_loss,
            grammar_logits=grammar_logits,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def save_pretrained(self, save_directory, *args, **kwargs):
        self.config.glossweaver_grammar_aware = True
        self.config.num_grammar_labels = self.num_grammar_labels
        self.config.lambda_grammar = self.lambda_grammar
        return super().save_pretrained(save_directory, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        num_labels = kwargs.pop("num_grammar_labels", None)
        grammar_lambda = kwargs.pop("lambda_grammar", None)
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if num_labels is not None and num_labels != model.num_grammar_labels:
            model.num_grammar_labels = num_labels
            model.grammar_head = nn.Linear(model.config.d_model, num_labels)
        if grammar_lambda is not None:
            model.set_grammar_lambda(grammar_lambda)
        return model
