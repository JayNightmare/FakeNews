---
title: FakeNews Classifier
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
license: mit
models:
  - JayNightmare/FakeNews
---

# FakeNews Classifier

Gradio interface for the `JayNightmare/FakeNews` PEFT LoRA adapter.

The model repo stores adapter files in `adapter/`, so the Space loads the base model from the adapter config and attaches the adapter with `subfolder="adapter"`.

Optional Space environment variables:

- `MODEL_REPO`: defaults to `JayNightmare/FakeNews`
- `ADAPTER_SUBFOLDER`: defaults to `adapter`
- `BASE_MODEL_ID`: optional override if the adapter config points to a local path
