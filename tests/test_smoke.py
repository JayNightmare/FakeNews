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


def test_heuristic_does_not_peek_at_label():
    """Heuristic baseline uses keyword analysis, not ground truth."""
    from src.run_experiment import heuristic_predict

    # Create a record labeled "real" but with fake-indicating text
    record = _make_record(
        mapped_label=0,
        mapped_label_name="real",
        text="This claim has been debunked as false and misleading by experts",
        metadata={"source": "test"},
    )
    result = heuristic_predict(record, "full")
    # The heuristic should predict fake based on keywords,
    # even though the ground truth says real
    assert result["predicted_label"] == 1, (
        f"Heuristic should predict fake based on keywords, got {result['predicted_label_name']}"
    )
    assert result["ground_truth_label"] == 0  # Ground truth is still real


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
