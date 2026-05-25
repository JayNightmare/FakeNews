"""Prediction backends for the misinformation pipeline.

Provides a small provider abstraction so the experiment runner can swap
between local Hugging Face models and API-backed models without embedding
transport details in the orchestration layer.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Callable
from urllib.request import Request, urlopen

from src.schema import UnifiedRecord


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers=headers, method="POST")
    with urlopen(request) as response:  # nosec B310
        return json.load(response)


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Model returned an empty response")

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not extract JSON object from model response: {stripped[:200]}")

    return json.loads(stripped[start:end + 1])


def normalize_prediction(
    record: UnifiedRecord,
    context_mode: str,
    mode: str,
    parsed: dict[str, Any],
    *,
    model_name: str | None = None,
    raw_response: Any | None = None,
) -> dict[str, Any]:
    """Normalize provider-specific output into the pipeline schema."""
    classification = str(parsed.get("classification", "fake")).strip().lower()
    predicted_label = 0 if classification == "real" else 1
    predicted_label_name = "real" if predicted_label == 0 else "fake"

    normalized = {
        "id": record.sample_id,
        "dataset": record.dataset,
        "context_mode": context_mode,
        "mode": mode,
        "ground_truth_label": record.mapped_label,
        "ground_truth_label_name": record.mapped_label_name,
        "predicted_label": predicted_label,
        "predicted_label_name": predicted_label_name,
        "confidence": parsed.get("confidence", 0.5),
        "explanation": parsed.get("explanation", ""),
        "reasoning_signals": parsed.get("reasoning_signals", []),
        "requires_external_evidence": parsed.get("requires_external_evidence", True),
        "raw_response": parsed if raw_response is None else raw_response,
    }
    if model_name:
        normalized["model"] = model_name
    return normalized


class BasePredictor(ABC):
    """Common interface for prediction backends."""

    @abstractmethod
    def predict(
        self,
        record: UnifiedRecord,
        prompt: str,
        context_mode: str,
    ) -> dict[str, Any]:
        """Return a normalized prediction dictionary."""


class FunctionPredictor(BasePredictor):
    """Adapter for simple in-process predictor functions."""

    def __init__(self, predict_fn: Callable[[UnifiedRecord, str], dict[str, Any]]) -> None:
        self._predict_fn = predict_fn

    def predict(
        self,
        record: UnifiedRecord,
        prompt: str,
        context_mode: str,
    ) -> dict[str, Any]:
        del prompt
        return self._predict_fn(record, context_mode)


class OpenAICompatiblePredictor(BasePredictor):
    """Predictor backed by an OpenAI-compatible chat completions API."""

    def __init__(self) -> None:
        self._api_key = os.getenv("OPENAI_API_KEY")
        self._base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is required for openai-compatible mode")

    def predict(
        self,
        record: UnifiedRecord,
        prompt: str,
        context_mode: str,
    ) -> dict[str, Any]:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        response = _post_json(f"{self._base_url}/chat/completions", payload, headers)
        content = response["choices"][0]["message"]["content"]
        parsed = _extract_json_object(content)
        return normalize_prediction(
            record,
            context_mode,
            "openai-compatible",
            parsed,
            model_name=self._model,
            raw_response=response,
        )


class HuggingFacePredictor(BasePredictor):
    """Predictor backed by a local Hugging Face causal LM.

    Uses an instruction-tuned chat model and expects a JSON object response.
    Loading is lazy so unit tests can import the module without heavy optional
    dependencies being installed.
    """

    def __init__(self) -> None:
        self._model_name = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
        self._adapter_path = os.getenv("HF_ADAPTER_PATH")
        self._device_map = os.getenv("HF_DEVICE_MAP", "auto")
        self._max_new_tokens = int(os.getenv("HF_MAX_NEW_TOKENS", "256"))
        self._temperature = float(os.getenv("HF_TEMPERATURE", "0"))
        self._top_p = float(os.getenv("HF_TOP_P", "1.0"))
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for huggingface mode. Install it with the project dependencies."
            ) from exc

        model_kwargs: dict[str, Any] = {"device_map": self._device_map}
        dtype = os.getenv("HF_TORCH_DTYPE")
        if dtype:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("torch is required when HF_TORCH_DTYPE is set") from exc
            model_kwargs["torch_dtype"] = getattr(torch, dtype)

        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModelForCausalLM.from_pretrained(self._model_name, **model_kwargs)

        if self._adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "peft is required when HF_ADAPTER_PATH is configured. Install it with the project dependencies."
                ) from exc
            model = PeftModel.from_pretrained(model, self._adapter_path)

        self._tokenizer = tokenizer
        self._model = model

    def _render_inputs(self, prompt: str) -> Any:
        assert self._tokenizer is not None
        messages = [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self._tokenizer, "apply_chat_template"):
            rendered = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = "\n\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
        return self._tokenizer(rendered, return_tensors="pt")

    def predict(
        self,
        record: UnifiedRecord,
        prompt: str,
        context_mode: str,
    ) -> dict[str, Any]:
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None

        model_inputs = self._render_inputs(prompt)
        if hasattr(self._model, "device"):
            model_inputs = {key: value.to(self._model.device) for key, value in model_inputs.items()}

        generation_kwargs = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": self._temperature > 0,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "pad_token_id": self._tokenizer.eos_token_id,
        }

        output_ids = self._model.generate(**model_inputs, **generation_kwargs)
        prompt_length = model_inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_length:]
        content = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        parsed = _extract_json_object(content)
        return normalize_prediction(
            record,
            context_mode,
            "huggingface",
            parsed,
            model_name=self._model_name,
            raw_response={"generated_text": content},
        )


def create_predictor(
    mode: str,
    *,
    heuristic_predictor: Callable[[UnifiedRecord, str], dict[str, Any]] | None = None,
) -> BasePredictor:
    """Factory for prediction backends."""
    if mode == "heuristic":
        if heuristic_predictor is None:
            raise RuntimeError("A heuristic predictor must be provided for heuristic mode")
        return FunctionPredictor(heuristic_predictor)
    if mode == "openai-compatible":
        return OpenAICompatiblePredictor()
    if mode == "huggingface":
        return HuggingFacePredictor()
    raise ValueError(f"Unsupported predictor mode '{mode}'")