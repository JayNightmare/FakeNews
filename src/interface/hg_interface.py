import json
import os
import threading

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_REPO = os.getenv("MODEL_REPO", "JayNightmare/FakeNews")
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID")
ADAPTER_SUBFOLDER = os.getenv("ADAPTER_SUBFOLDER", "adapter")

ADAPTER_SUBFOLDER_CANDIDATES = tuple(
    dict.fromkeys(
        [
            ADAPTER_SUBFOLDER,
            "adapter",
            "FakeNews_Model/adapter",
            "artifacts/FakeNews_Model/adapter",
            "model_adapter",
            "FakeNews_Model/model_adapter",
            "artifacts/FakeNews_Model/model_adapter",
            "",
        ]
    )
)

_generation_lock = threading.Lock()


def _download_adapter_config():
    last_error = None

    for subfolder in ADAPTER_SUBFOLDER_CANDIDATES:
        try:
            path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename="adapter_config.json",
                subfolder=subfolder or None,
            )
            return path, subfolder or None
        except Exception as error:
            last_error = error

    raise RuntimeError(
        "Could not find adapter_config.json. Set ADAPTER_SUBFOLDER to the repo "
        "folder that contains adapter_config.json and adapter_model.safetensors. "
        f"Tried: {', '.join(subfolder or '<repo root>' for subfolder in ADAPTER_SUBFOLDER_CANDIDATES)}."
    ) from last_error


def _resolve_base_model_id(adapter_config_path):
    with open(adapter_config_path, "r", encoding="utf-8") as file:
        adapter_config = json.load(file)

    base_model_id = BASE_MODEL_ID or adapter_config.get("base_model_name_or_path")

    if not base_model_id:
        raise RuntimeError(
            "Could not resolve the base model. Set BASE_MODEL_ID in the Space environment."
        )

    if base_model_id.startswith(("/", "./", "../")) and not BASE_MODEL_ID:
        raise RuntimeError(
            "adapter_config.json points to a local base model path. "
            "Set BASE_MODEL_ID to the original Hugging Face base model id."
        )

    return base_model_id


def _load_tokenizer(base_model_id, adapter_subfolder):
    try:
        return AutoTokenizer.from_pretrained(
            MODEL_REPO,
            subfolder=adapter_subfolder,
            trust_remote_code=True,
        )
    except Exception:
        return AutoTokenizer.from_pretrained(
            base_model_id,
            trust_remote_code=True,
        )


def _load_model():
    adapter_config_path, adapter_subfolder = _download_adapter_config()
    base_model_id = _resolve_base_model_id(adapter_config_path)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = _load_tokenizer(base_model_id, adapter_subfolder)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=dtype,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        MODEL_REPO,
        subfolder=adapter_subfolder,
    )

    model.to(device)
    model.eval()

    return model, tokenizer, device, base_model_id, adapter_subfolder


model, tokenizer, device, base_model_id, adapter_subfolder = _load_model()


def _build_prompt(claim, context):
    context = context.strip() or "No additional context provided."

    messages = [
        {
            "role": "system",
            "content": (
                "You are FakeNews, a misinformation classifier. "
                "Classify the claim using the available context. "
                "Return a concise label and explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Claim:\n{claim.strip()}\n\n"
                f"Context:\n{context}\n\n"
                "Classify this as fake, real, misleading, or uncertain."
            ),
        },
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return (
            "System: You are FakeNews, a misinformation classifier.\n\n"
            f"User: Claim:\n{claim.strip()}\n\n"
            f"Context:\n{context}\n\n"
            "Classify this as fake, real, misleading, or uncertain.\n\n"
            "Assistant:"
        )


def classify(claim, context):
    if not claim or not claim.strip():
        return "Enter a claim to classify."

    prompt = _build_prompt(claim, context or "")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    with _generation_lock:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


description = (
    f"Loaded adapter from `{MODEL_REPO}`"
    + (f" subfolder `{adapter_subfolder}`" if adapter_subfolder else "")
    + f" on base model `{base_model_id}`."
)

demo = gr.Interface(
    fn=classify,
    inputs=[
        gr.Textbox(label="Claim", lines=3, placeholder="Enter a news claim..."),
        gr.Textbox(label="Optional context", lines=6, placeholder="Paste article/context here..."),
    ],
    outputs=gr.Textbox(label="FakeNews result", lines=8),
    title="FakeNews Classifier",
    description=description,
    examples=[
        [
            "The Eiffel Tower was originally built for the 1889 World's Fair.",
            "",
        ],
        [
            "A viral post claims drinking salt water cures all viral infections.",
            "No credible medical source supports this claim.",
        ],
    ],
)

if __name__ == "__main__":
    demo.launch()