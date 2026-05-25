PYTHON ?= python3
HF_MODEL_ID ?= Qwen/Qwen2.5-1.5B-Instruct

.PHONY: pilot pilot-claimreview pilot-fakeddit pilot-fakenewsnet pilot-mumin pilot-all
.PHONY: pilot-context-ablation pilot-context-minimal pilot-context-misleading pilot-grounded
.PHONY: train-qwen-prepare train-qwen-adapter clean-data summary evaluate setup help

# ── Quick start ────────────────────────────────────────────────
pilot:
	HF_MODEL_ID="$(HF_MODEL_ID)" $(PYTHON) src/run_experiment.py --dataset claimreview --limit 100 --mode huggingface --output-dir artifacts/pilot_run

# ── Per-dataset pilots ────────────────────────────────────────
pilot-claimreview:
	HF_MODEL_ID="$(HF_MODEL_ID)" $(PYTHON) src/run_experiment.py --dataset claimreview --limit 100 --mode huggingface --output-dir artifacts/claimreview_run

pilot-fakeddit:
	HF_MODEL_ID="$(HF_MODEL_ID)" $(PYTHON) src/run_experiment.py --dataset fakeddit --limit 100 --mode huggingface --output-dir artifacts/fakeddit_run

pilot-fakenewsnet:
	HF_MODEL_ID="$(HF_MODEL_ID)" $(PYTHON) src/run_experiment.py --dataset fakenewsnet --limit 100 --mode huggingface --output-dir artifacts/fakenewsnet_run

pilot-mumin:
	HF_MODEL_ID="$(HF_MODEL_ID)" $(PYTHON) src/run_experiment.py --dataset mumin --limit 100 --mode huggingface --output-dir artifacts/mumin_run

# ── Multi-dataset + context variants ─────────────────────────
pilot-all:
	HF_MODEL_ID="$(HF_MODEL_ID)" $(PYTHON) src/run_experiment.py --dataset all --limit 50 --mode huggingface --output-dir artifacts/all_datasets_run

pilot-context-minimal:
	HF_MODEL_ID="$(HF_MODEL_ID)" $(PYTHON) src/run_experiment.py --dataset claimreview --context-mode minimal --limit 100 --mode huggingface --output-dir artifacts/context_minimal

pilot-context-misleading:
	HF_MODEL_ID="$(HF_MODEL_ID)" $(PYTHON) src/run_experiment.py --dataset claimreview --context-mode misleading --limit 100 --mode huggingface --output-dir artifacts/context_misleading

pilot-context-ablation:
	HF_MODEL_ID="$(HF_MODEL_ID)" $(PYTHON) src/run_experiment.py --dataset claimreview --context-mode full --context-ablation-levels 1.0,0.75,0.5,0.25 --limit 100 --mode huggingface --output-dir artifacts/context_ablation

pilot-grounded:
	HF_MODEL_ID="$(HF_MODEL_ID)" $(PYTHON) src/run_experiment.py --dataset claimreview --context-mode full --limit 100 --mode huggingface --ground-with-google --output-dir artifacts/grounded_run

train-qwen-prepare:
	$(PYTHON) src/train_hf_adapter.py --dataset claimreview --model-id "$(HF_MODEL_ID)" --prepare-only --output-dir artifacts/training/qwen_adapter

train-qwen-adapter:
	$(PYTHON) src/train_hf_adapter.py --dataset claimreview --model-id "$(HF_MODEL_ID)" --output-dir artifacts/training/qwen_adapter

# ── Setup ─────────────────────────────────────────────────────
setup:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

# ── Help ──────────────────────────────────────────────────────
help:
	@echo "Available targets:"
	@echo "  pilot                  - ClaimReview with local Hugging Face model"
	@echo "  pilot-claimreview      - ClaimReview pilot"
	@echo "  pilot-fakeddit         - Fakeddit pilot"
	@echo "  pilot-fakenewsnet      - FakeNewsNet pilot"
	@echo "  pilot-mumin            - MuMiN pilot"
	@echo "  pilot-all              - All datasets pilot"
	@echo "  pilot-context-minimal  - ClaimReview with minimal context"
	@echo "  pilot-context-misleading - ClaimReview with misleading context"
	@echo "  pilot-context-ablation - ClaimReview across multiple context budgets"
	@echo "  pilot-grounded         - ClaimReview with cached Google grounding"
	@echo "  train-qwen-prepare     - Export deterministic train/eval chat corpus"
	@echo "  train-qwen-adapter     - Fine-tune a LoRA adapter on ClaimReview"
	@echo "  setup                  - Create venv and install deps"
	@echo "  help                   - Show this help"
