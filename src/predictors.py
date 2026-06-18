"""Prediction backends for the misinformation pipeline.

Provides a small provider abstraction so the experiment runner can swap
between local Hugging Face models and API-backed models without embedding
transport details in the orchestration layer.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Callable, cast
from urllib.request import Request, urlopen

from src.env import load_project_env
from src.schema import UnifiedRecord

load_project_env()


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
        self._hf_token = os.getenv("HF_TOKEN")
        self._adapter_path = os.getenv("HF_ADAPTER_PATH")
        self._device_map = os.getenv("HF_DEVICE_MAP")
        self._max_new_tokens = int(os.getenv("HF_MAX_NEW_TOKENS", "256"))
        self._temperature = float(os.getenv("HF_TEMPERATURE", "0"))
        self._top_p = float(os.getenv("HF_TOP_P", "1.0"))
        self._enable_images = os.getenv("HF_ENABLE_IMAGES", "1").strip().lower() not in {"0", "false", "no"}
        self._tokenizer = None
        self._processor = None
        self._model = None
        self._multimodal = False

    @property
    def _do_sample(self) -> bool:
        return self._temperature > 0 or self._top_p < 1.0

    def _normalize_generation_config(self, model: Any) -> None:
        generation_config = getattr(model, "generation_config", None)
        if generation_config is None:
            return

        generation_config.do_sample = self._do_sample
        if self._do_sample:
            generation_config.temperature = self._temperature if self._temperature > 0 else 1.0
            generation_config.top_p = self._top_p
            return

        generation_config.temperature = 1.0
        generation_config.top_p = 1.0

    def _ensure_loaded(self) -> None:
        if self._model is not None and (self._tokenizer is not None or self._processor is not None):
            return

        try:
            import transformers
            from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for huggingface mode. Install it with the project dependencies."
            ) from exc

        model_kwargs: dict[str, Any] = {}
        if self._device_map:
            if self._device_map == "auto":
                try:
                    import accelerate  # noqa: F401
                except ImportError:
                    self._device_map = None
                else:
                    model_kwargs["device_map"] = self._device_map
            else:
                model_kwargs["device_map"] = self._device_map
        dtype = os.getenv("HF_TORCH_DTYPE")
        if dtype:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("torch is required when HF_TORCH_DTYPE is set") from exc
            model_kwargs["torch_dtype"] = getattr(torch, dtype)

        load_kwargs: dict[str, Any] = {}
        if self._hf_token:
            load_kwargs["token"] = self._hf_token

        config = AutoConfig.from_pretrained(self._model_name, **load_kwargs)
        config_name = type(config).__name__.casefold()
        is_multimodal = any(token in config_name for token in ("vision", "vl", "multi"))

        tokenizer = None
        processor = None
        model = None
        if is_multimodal:
            model_loader = getattr(transformers, "AutoModelForImageTextToText", None)
            if model_loader is None:
                model_loader = getattr(transformers, "AutoModelForVision2Seq", None)
            if model_loader is None:
                raise RuntimeError(
                    "Your installed transformers version does not expose an auto-loader for image+text models. "
                    "Upgrade transformers or use a text-only HF_MODEL_ID."
                )
            processor = AutoProcessor.from_pretrained(self._model_name, **load_kwargs)
            model = model_loader.from_pretrained(self._model_name, **model_kwargs, **load_kwargs)
            self._multimodal = True
        else:
            tokenizer = AutoTokenizer.from_pretrained(self._model_name, **load_kwargs)
            model = AutoModelForCausalLM.from_pretrained(self._model_name, **model_kwargs, **load_kwargs)
            self._multimodal = False

        self._normalize_generation_config(model)

        if self._adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "peft is required when HF_ADAPTER_PATH is configured. Install it with the project dependencies."
                ) from exc
            model = PeftModel.from_pretrained(model, self._adapter_path)

        self._tokenizer = tokenizer
        self._processor = processor
        self._model = model

    def _resolve_image_url(self, record: UnifiedRecord) -> str | None:
        metadata = record.metadata
        direct_keys = ("image_url", "top_img", "thumbnail", "image")
        for key in direct_keys:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip().lower().startswith(("http://", "https://")):
                return value.strip()

        images = metadata.get("images")
        if isinstance(images, list):
            for candidate in images:
                if isinstance(candidate, str) and candidate.strip().lower().startswith(("http://", "https://")):
                    return candidate.strip()

        return None

    def _load_image(self, image_url: str) -> Any:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Pillow is required for multimodal image+text inference. Install it via requirements.txt."
            ) from exc

        request = Request(image_url, headers={"User-Agent": "Mozilla/5.0"}, method="GET")
        with urlopen(request) as response:  # nosec B310
            image_bytes = response.read()
        return Image.open(BytesIO(image_bytes)).convert("RGB")

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

    def _render_multimodal_inputs(self, prompt: str, image: Any | None) -> Any:
        assert self._processor is not None
        user_content: list[dict[str, Any]] = []
        if image is not None:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Return JSON only."}]},
            {"role": "user", "content": user_content},
        ]

        if hasattr(self._processor, "apply_chat_template"):
            rendered = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = f"system: Return JSON only.\n\nuser: {prompt}"

        if image is not None:
            return self._processor(text=[rendered], images=[image], return_tensors="pt")
        return self._processor(text=[rendered], return_tensors="pt")

    def predict(
        self,
        record: UnifiedRecord,
        prompt: str,
        context_mode: str,
    ) -> dict[str, Any]:
        self._ensure_loaded()
        assert self._model is not None

        model_inputs: dict[str, Any]
        decoder = self._tokenizer
        image_url = None
        image_loaded = False
        if self._multimodal:
            image = None
            image_url = self._resolve_image_url(record)
            if self._enable_images and image_url:
                try:
                    image = self._load_image(image_url)
                    image_loaded = True
                except Exception:
                    image = None
            model_inputs = self._render_multimodal_inputs(prompt, image)
            if self._processor is not None and hasattr(self._processor, "tokenizer"):
                decoder = self._processor.tokenizer
        else:
            assert self._tokenizer is not None
            model_inputs = self._render_inputs(prompt)

        if decoder is None:
            raise RuntimeError("No tokenizer available for decoding model output.")

        if hasattr(self._model, "device"):
            model_inputs = {key: value.to(self._model.device) for key, value in model_inputs.items()}

        generation_kwargs = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": self._do_sample,
            "pad_token_id": decoder.eos_token_id,
        }
        if self._do_sample:
            if self._temperature > 0:
                generation_kwargs["temperature"] = self._temperature
            if self._top_p < 1.0:
                generation_kwargs["top_p"] = self._top_p

        output_ids = cast(Any, self._model).generate(**model_inputs, **generation_kwargs)
        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("Could not locate input_ids in model inputs for decoding.")
        prompt_length = input_ids.shape[-1]
        generated_ids = output_ids[0][prompt_length:]
        content = decoder.decode(generated_ids, skip_special_tokens=True)
        parsed = _extract_json_object(content)
        return normalize_prediction(
            record,
            context_mode,
            "huggingface",
            parsed,
            model_name=self._model_name,
            raw_response={
                "generated_text": content,
                "multimodal": self._multimodal,
                "image_url": image_url,
                "image_loaded": image_loaded,
            },
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