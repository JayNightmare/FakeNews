"""Smoke tests for core pipeline components.

Run with: python3 -m pytest tests/ -v
Or without pytest: python3 tests/test_smoke.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.schema import UnifiedRecord


def _make_record(**overrides) -> UnifiedRecord:
    """Create a minimal valid UnifiedRecord for testing."""
    defaults = {
        "dataset": "TestDataset",
        "sample_id": "test_001",
        "text": "This is a test claim about a political figure.",
        "original_label": "false",
        "original_label_name": "false",
        "mapped_label": 1,
        "mapped_label_name": "fake",
        "split": "test",
        "source_fields_used": ["text"],
        "context_text": "Additional context about the claim.",
        "modality": "text",
        "has_image": False,
        "metadata": {"source": "test", "review_publisher": "TestChecker"},
        "cleaning_notes": [],
    }
    defaults.update(overrides)
    return UnifiedRecord(**defaults)


def test_schema_roundtrip():
    """UnifiedRecord serializes and deserializes correctly."""
    record = _make_record()
    d = record.to_dict()
    assert d["dataset"] == "TestDataset"
    assert d["mapped_label"] == 1
    assert d["mapped_label_name"] == "fake"
    assert isinstance(d["metadata"], dict)
    # Verify JSON serialization
    json_str = json.dumps(d)
    restored = json.loads(json_str)
    assert restored["sample_id"] == "test_001"


def test_cleaning_preserves_good_records():
    """Cleaning pipeline keeps valid records."""
    from src.cleaning import clean_record
    record = _make_record()
    result = clean_record(record)
    assert result is not None
    assert result.text == record.text


def test_cleaning_drops_short_text():
    """Cleaning pipeline drops records with < 5 words."""
    from src.cleaning import clean_record
    record = _make_record(text="Too short")
    result = clean_record(record)
    assert result is None


def test_cleaning_drops_empty_text():
    """Cleaning pipeline drops records with empty text."""
    from src.cleaning import clean_record
    record = _make_record(text="")
    result = clean_record(record)
    assert result is None


def test_cleaning_strips_whitespace():
    """Cleaning collapses excess whitespace."""
    from src.cleaning import clean_record
    record = _make_record(text="This   has   excessive   spacing   throughout   the   text")
    result = clean_record(record)
    assert result is not None
    assert "   " not in result.text


def test_cleaning_batch_report():
    """Batch cleaning produces a correct report."""
    from src.cleaning import clean_records
    records = [
        _make_record(sample_id="good_1"),
        _make_record(sample_id="short_1", text="Nope"),
        _make_record(sample_id="empty_1", text=""),
        _make_record(sample_id="good_2", text="This is another perfectly valid claim text"),
    ]
    cleaned, report = clean_records(records)
    assert len(cleaned) == 2
    assert report.total_input == 4
    assert report.total_output == 2
    assert report.total_dropped == 2


def test_prompts_minimal():
    """Minimal context prompt contains claim text only."""
    from src.prompts import build_prompt
    record = _make_record()
    prompt = build_prompt(record, "minimal")
    assert "This is a test claim" in prompt
    assert "Additional context" not in prompt


def test_prompts_full():
    """Full context prompt includes context and metadata."""
    from src.prompts import build_prompt
    record = _make_record()
    prompt = build_prompt(record, "full")
    assert "This is a test claim" in prompt
    assert "Additional context" in prompt


def test_prompts_full_context_budget_truncates_context():
    """Full context prompt applies deterministic context budgets."""
    from src.prompts import build_prompt

    record = _make_record(
        context_text=(
            "Sentence one provides supporting detail. "
            "Sentence two adds more evidence. "
            "Sentence three is extra background."
        )
    )
    prompt = build_prompt(record, "full", context_budget=0.34)
    assert "Sentence one provides supporting detail." in prompt
    assert "Sentence two adds more evidence." not in prompt


def test_prompts_misleading_is_label_blind():
    """Misleading context does NOT reference the ground-truth label."""
    from src.prompts import build_prompt
    fake_record = _make_record(mapped_label=1, mapped_label_name="fake")
    real_record = _make_record(mapped_label=0, mapped_label_name="real",
                               sample_id="test_001")  # Same id → same seed
    fake_prompt = build_prompt(fake_record, "misleading")
    real_prompt = build_prompt(real_record, "misleading")
    # Same sample_id means same seed → same misleading template
    # The important thing is neither prompt says "this is fake" or "this is real"
    assert "mapped_label" not in fake_prompt
    assert "mapped_label" not in real_prompt
    assert "ground_truth" not in fake_prompt


def test_evaluation_reports_context_budget_flip_thresholds():
    """Evaluation surfaces context-ablation flips by budget."""
    from src.evaluation import evaluate_predictions

    predictions = [
        {
            "id": "claim_1",
            "dataset": "TestDataset",
            "context_mode": "full",
            "context_budget": 1.0,
            "ground_truth_label": 0,
            "predicted_label": 0,
            "confidence": 0.9,
        },
        {
            "id": "claim_1",
            "dataset": "TestDataset",
            "context_mode": "full",
            "context_budget": 0.5,
            "ground_truth_label": 0,
            "predicted_label": 0,
            "confidence": 0.7,
        },
        {
            "id": "claim_1",
            "dataset": "TestDataset",
            "context_mode": "full",
            "context_budget": 0.25,
            "ground_truth_label": 0,
            "predicted_label": 1,
            "confidence": 0.62,
        },
    ]

    report = evaluate_predictions(predictions)
    assert "by_context_budget" in report
    assert report["context_ablation"]["prediction_flip_count"] == 1
    assert report["context_ablation"]["thresholds_to_fake"]["count"] == 1
    assert report["context_ablation"]["thresholds_to_fake"]["mean_context_budget"] == 0.25


def test_google_factcheck_cache_reuses_responses():
    """Google fact-check enrichment reuses cached responses across runs."""
    from src.google_factcheck import GoogleFactCheckClient, enrich_records_with_google_factcheck

    response = {
        "claims": [
            {
                "text": "This is a test claim about a political figure.",
                "claimReview": [
                    {
                        "title": "Fact check review",
                        "url": "https://example.com/review",
                        "textualRating": "False",
                        "publisher": {"name": "Example Checker"},
                    }
                ],
            }
        ]
    }
    calls: list[str] = []

    def fake_fetch(url: str):
        calls.append(url)
        return response

    client = GoogleFactCheckClient(api_key="test-key", fetch_json=fake_fetch)

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = Path(temp_dir) / "google_cache.json"
        first_records, first_report = enrich_records_with_google_factcheck(
            [_make_record()],
            cache_path=cache_path,
            client=client,
        )
        second_records, second_report = enrich_records_with_google_factcheck(
            [_make_record()],
            cache_path=cache_path,
            client=client,
        )

        assert first_report["live_fetches"] == 1
        assert second_report["cache_hits"] == 1
        assert len(calls) == 1
        assert "google_fact_check" in first_records[0].metadata
        assert second_records[0].metadata["google_fact_check"]["cache_status"] == "hit"


def test_training_example_exports_supervised_chat_messages():
    """Training export should produce a chat-format example with JSON supervision."""
    from src.training import build_training_example

    example = build_training_example(_make_record(mapped_label=0, mapped_label_name="real"))

    assert example.messages[0]["role"] == "system"
    assert example.messages[1]["role"] == "user"
    assert example.messages[2]["role"] == "assistant"
    response_payload = json.loads(example.response)
    assert response_payload["classification"] == "real"
    assert isinstance(response_payload["reasoning_signals"], list)


def test_training_partition_prefers_explicit_splits():
    """Training partitioning should preserve explicit train/eval dataset splits."""
    from src.training import partition_records_for_training

    records = [
        _make_record(sample_id="train_1", split="train"),
        _make_record(sample_id="train_2", split="train"),
        _make_record(sample_id="eval_1", split="validation"),
    ]

    train_records, eval_records = partition_records_for_training(records, eval_ratio=0.2, seed=7)
    assert [record.sample_id for record in train_records] == ["train_1", "train_2"]
    assert [record.sample_id for record in eval_records] == ["eval_1"]


def test_export_training_corpus_writes_manifest_and_examples():
    """Training corpus export writes deterministic train/eval JSONL files."""
    from src.training import export_training_corpus

    records = [
        _make_record(sample_id="train_1", split="train"),
        _make_record(sample_id="eval_1", split="validation"),
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        manifest = export_training_corpus(records, output_dir, eval_ratio=0.2)
        train_path = output_dir / "train_examples.jsonl"
        eval_path = output_dir / "eval_examples.jsonl"
        manifest_path = output_dir / "training_manifest.json"

        assert manifest["train_count"] == 1
        assert manifest["eval_count"] == 1
        assert train_path.exists()
        assert eval_path.exists()
        assert manifest_path.exists()
        assert "classification" in train_path.read_text(encoding="utf-8")


def test_evaluation_metrics():
    """Evaluation correctly computes binary metrics."""
    from src.evaluation import confusion_matrix, compute_metrics
    gt = [1, 1, 0, 0, 1, 0]
    pred = [1, 0, 0, 1, 1, 0]
    cm = confusion_matrix(gt, pred)
    assert cm["tp"] == 2
    assert cm["tn"] == 2
    assert cm["fp"] == 1
    assert cm["fn"] == 1
    metrics = compute_metrics(cm)
    assert 0 < metrics["accuracy"] < 1
    assert 0 < metrics["f1"] < 1


def test_summary_generation():
    """Summary generator produces expected fields."""
    from src.summary import generate_summary
    records = [_make_record(sample_id=f"s{i}") for i in range(5)]
    summary = generate_summary(records, "TestDataset", pre_cleaning_count=6)
    assert summary["dataset"] == "TestDataset"
    assert summary["sample_count"] == 5
    assert summary["pre_cleaning_count"] == 6
    assert "label_types" in summary
    assert "text_length_stats" in summary


def test_visualization_generation():
    """Visualization stage writes a dashboard and chart artifacts."""
    from src.summary import generate_summary
    from src.evaluation import evaluate_predictions
    from src.visualization import generate_run_visualizations

    records = [
        _make_record(sample_id="real_1", mapped_label=0, mapped_label_name="real", has_image=True),
        _make_record(sample_id="fake_1", mapped_label=1, mapped_label_name="fake", has_image=False),
    ]
    summary = generate_summary(records, "TestDataset", pre_cleaning_count=2)
    predictions = [
        {
            "id": "real_1",
            "dataset": "TestDataset",
            "context_mode": "full",
            "ground_truth_label": 0,
            "ground_truth_label_name": "real",
            "predicted_label": 0,
            "predicted_label_name": "real",
            "confidence": 0.88,
        },
        {
            "id": "fake_1",
            "dataset": "TestDataset",
            "context_mode": "full",
            "ground_truth_label": 1,
            "ground_truth_label_name": "fake",
            "predicted_label": 1,
            "predicted_label_name": "fake",
            "confidence": 0.91,
        },
    ]
    eval_report = evaluate_predictions(predictions)

    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        manifest = generate_run_visualizations(records, summary, eval_report, predictions, tmp_path)
        dashboard = tmp_path / "visualizations" / "dashboard.html"
        label_chart = tmp_path / "visualizations" / "label_distribution_pie.svg"
        report = tmp_path / "visualizations" / "visualization_report.md"

        assert dashboard.exists()
        assert label_chart.exists()
        assert report.exists()
        assert "dashboard" in manifest
        assert "report" in manifest
        assert "Mapped Label Distribution" in dashboard.read_text(encoding="utf-8")
        assert "Where it came from" in report.read_text(encoding="utf-8")


def test_aggregate_visualization_generation_handles_int_labels():
    """Aggregate visualization stage accepts integer label keys from evaluation output."""
    from src.visualization import generate_aggregate_visualizations

    run_summaries = [
        {"dataset": "ClaimReview", "records_after_cleaning": 4, "accuracy": 1.0, "f1": 1.0},
        {"dataset": "Fakeddit", "records_after_cleaning": 86, "accuracy": 0.5465, "f1": 0.7068},
    ]
    aggregate_eval = {
        "prediction_count": 90,
        "overall": {"accuracy": 0.5667, "f1": 0.7234},
        "label_distribution": {"ground_truth": {0: 12, 1: 78}},
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        manifest = generate_aggregate_visualizations(run_summaries, aggregate_eval, tmp_path)
        dashboard = tmp_path / "visualizations" / "dashboard.html"
        report = tmp_path / "visualizations" / "visualization_report.md"
        pie_chart = tmp_path / "visualizations" / "aggregate_labels_pie.svg"

        assert dashboard.exists()
        assert report.exists()
        assert pie_chart.exists()
        assert "report" in manifest
        assert "Aggregate Ground-Truth Labels" in report.read_text(encoding="utf-8")


def test_fakenewsnet_loader_reads_fetched_csv_format():
    """FakeNewsNet loader accepts the fetched CSV format written by fetch_datasets.py."""
    from src.datasets.fakenewsnet import FakeNewsNetLoader

    csv_text = "\n".join([
        "title,news_url,source_domain,tweet_num,real",
        "A verified story,https://example.com/story,example.com,12,1",
        "A fabricated story,https://example.com/fake,example.com,4,0",
    ])

    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        (data_dir / "fakenewsnet.csv").write_text(csv_text + "\n", encoding="utf-8")

        loader = FakeNewsNetLoader()
        records = loader.load(data_dir)

        assert len(records) == 2
        assert records[0].dataset == "FakeNewsNet"
        assert records[0].mapped_label_name == "real"
        assert records[1].mapped_label_name == "fake"


def test_mumin_loader_reads_fetched_csv_format():
    """MuMiN loader accepts the fetched CSV format written by fetch_datasets.py."""
    from src.datasets.mumin import MuMiNLoader

    csv_text = "\n".join([
        "id,claim,claim_en,verdict,train_mask,val_mask,test_mask",
        "9,Texto original,Translated claim,factual,True,False,False",
        "10,Outro texto,Another translated claim,misinformation,False,False,True",
    ])

    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        (data_dir / "mumin.csv").write_text(csv_text + "\n", encoding="utf-8")

        loader = MuMiNLoader()
        records = loader.load(data_dir)

        assert len(records) == 2
        assert records[0].dataset == "MuMiN"
        assert records[0].text == "Translated claim"
        assert records[0].mapped_label_name == "real"
        assert records[0].split == "train"
        assert records[1].mapped_label_name == "fake"
        assert records[1].split == "test"


def _run_all():
    """Run all tests without pytest."""
    import traceback
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = failed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"  ✓ {test_fn.__name__}")
            passed += 1
        except Exception:
            print(f"  ✗ {test_fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
