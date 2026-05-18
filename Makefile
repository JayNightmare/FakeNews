PYTHON ?= python3

.PHONY: pilot pilot-claimreview pilot-fakeddit pilot-fakenewsnet pilot-mumin pilot-all
.PHONY: pilot-llm clean-data summary evaluate setup help

# ── Quick start (backward-compatible) ─────────────────────────
pilot:
	$(PYTHON) src/run_experiment.py --dataset claimreview --limit 100 --mode heuristic --output-dir artifacts/pilot_run

pilot-llm:
	$(PYTHON) src/run_experiment.py --dataset claimreview --limit 100 --mode openai-compatible --output-dir artifacts/pilot_run_llm

# ── Per-dataset pilots ────────────────────────────────────────
pilot-claimreview:
	$(PYTHON) src/run_experiment.py --dataset claimreview --limit 100 --mode heuristic --output-dir artifacts/claimreview_run

pilot-fakeddit:
	$(PYTHON) src/run_experiment.py --dataset fakeddit --limit 100 --mode heuristic --output-dir artifacts/fakeddit_run

pilot-fakenewsnet:
	$(PYTHON) src/run_experiment.py --dataset fakenewsnet --limit 100 --mode heuristic --output-dir artifacts/fakenewsnet_run

pilot-mumin:
	$(PYTHON) src/run_experiment.py --dataset mumin --limit 100 --mode heuristic --output-dir artifacts/mumin_run

# ── Multi-dataset + context variants ─────────────────────────
pilot-all:
	$(PYTHON) src/run_experiment.py --dataset all --limit 50 --mode heuristic --output-dir artifacts/all_datasets_run

pilot-context-minimal:
	$(PYTHON) src/run_experiment.py --dataset claimreview --context-mode minimal --limit 100 --mode heuristic --output-dir artifacts/context_minimal

pilot-context-misleading:
	$(PYTHON) src/run_experiment.py --dataset claimreview --context-mode misleading --limit 100 --mode heuristic --output-dir artifacts/context_misleading

# ── Setup ─────────────────────────────────────────────────────
setup:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

# ── Help ──────────────────────────────────────────────────────
help:
	@echo "Available targets:"
	@echo "  pilot                  - ClaimReview heuristic baseline (backward-compat)"
	@echo "  pilot-llm              - ClaimReview with OpenAI-compatible API"
	@echo "  pilot-claimreview      - ClaimReview pilot"
	@echo "  pilot-fakeddit         - Fakeddit pilot"
	@echo "  pilot-fakenewsnet      - FakeNewsNet pilot"
	@echo "  pilot-mumin            - MuMiN pilot"
	@echo "  pilot-all              - All datasets pilot"
	@echo "  pilot-context-minimal  - ClaimReview with minimal context"
	@echo "  pilot-context-misleading - ClaimReview with misleading context"
	@echo "  setup                  - Create venv and install deps"
	@echo "  help                   - Show this help"
