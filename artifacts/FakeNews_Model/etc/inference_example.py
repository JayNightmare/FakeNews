#!/usr/bin/env python3
"""Load the base model + FakeNews adapter for local inference."""

from __future__ import annotations

from peft import PeftModel
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "adapter"


def main() -> int:
    config = AutoConfig.from_pretrained(BASE_MODEL_ID)
    config_name = type(config).__name__.casefold()
    is_vl = any(token in config_name for token in ("vision", "vl", "multi"))

    if is_vl:
        model_loader = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_loader is None:
            model_loader = getattr(transformers, "AutoModelForVision2Seq", None)
        if model_loader is None:
            raise RuntimeError("This transformers version does not support VL auto loaders.")
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
        tokenizer = processor.tokenizer
        model = model_loader.from_pretrained(BASE_MODEL_ID)
    else:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID)

    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    prompt = "Classify this claim as real or fake and explain: The moon is made of cheese."
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
