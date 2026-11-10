"""Reusable inference engine for CVE-to-CWE classification."""

from dataclasses import dataclass
from typing import Any

from .modeling import DEFAULT_MODEL, environment_info, load_base_model, resolve_dtype, resolve_model_revision
from .parsing import parse_prediction
from .prompts import clean_description
from .tokenization import check_label_roundtrip, render_prompt


@dataclass(frozen=True)
class InferenceConfig:
    model: str = DEFAULT_MODEL
    model_revision: str | None = None
    adapter: str | None = None
    quantized: bool = True
    dtype: str = "auto"
    max_new_tokens: int = 12
    batch_size: int = 32
    max_description_chars: int = 1500


class CWEInferenceEngine:
    """Lazy-loading, deterministic inference engine shared by CLI and API consumers."""

    def __init__(self, config: InferenceConfig):
        if config.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if config.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.config = config
        self.tokenizer = None
        self.model = None
        self.revision: str | None = None
        self._label_roundtrip_checked = False

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self) -> "CWEInferenceEngine":
        if self.loaded:
            return self
        import torch
        from transformers import AutoTokenizer

        self.revision = resolve_model_revision(self.config.model, self.config.model_revision)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, revision=self.revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        dtype = resolve_dtype(torch, self.config.dtype)
        self.model = load_base_model(self.config.model, self.config.quantized, dtype, revision=self.revision)
        if self.config.adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.config.adapter)
        self.model.eval()
        return self

    def _validate_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("description must be a string")
        cleaned = clean_description(text, max_chars=self.config.max_description_chars)
        if not cleaned:
            raise ValueError("description must not be empty")
        return cleaned

    def predict_raw(self, descriptions: list[str]) -> list[str]:
        if not isinstance(descriptions, list):
            raise TypeError("descriptions must be a list")
        cleaned = [self._validate_text(text) for text in descriptions]
        if not cleaned:
            return []
        self.load()
        import torch

        if not self._label_roundtrip_checked:
            check_label_roundtrip(self.tokenizer, [f"CWE-{number}" for number in (22, 79, 89, 416, 787)])
            self._label_roundtrip_checked = True
        order = sorted(range(len(cleaned)), key=lambda index: len(cleaned[index]))
        outputs: list[str | None] = [None] * len(cleaned)
        for start in range(0, len(order), self.config.batch_size):
            batch = order[start : start + self.config.batch_size]
            prompts = [render_prompt(self.tokenizer, cleaned[index]) for index in batch]
            encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            texts = self.tokenizer.batch_decode(
                generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for index, text in zip(batch, texts, strict=True):
                outputs[index] = text.strip()
        return [text or "" for text in outputs]

    def predict(self, description: str) -> dict[str, Any]:
        raw = self.predict_raw([description])[0]
        return {"prediction": parse_prediction(raw), "raw": raw}

    def metadata(self) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "model_revision": self.revision or self.config.model_revision,
            "adapter": self.config.adapter,
            "quantized": self.config.quantized,
            "max_new_tokens": self.config.max_new_tokens,
            "batch_size": self.config.batch_size,
            "loaded": self.loaded,
            "environment": environment_info() if self.loaded else None,
        }
