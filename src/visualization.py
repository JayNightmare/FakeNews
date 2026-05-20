"""Static visualization helpers for experiment artifacts."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

from src.schema import UnifiedRecord


_PALETTE = [
    "#1d4ed8",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#4b5563",
    "#db2777",
]


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _fmt_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _table_html(title: str, headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_rows: list[str] = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(_fmt_number(value))}</td>" for value in row)
        body_rows.append(f"<tr>{cells}</tr>")
    body = "".join(body_rows) if body_rows else (
        f"<tr><td colspan=\"{len(headers)}\">No data available</td></tr>"
    )
    return (
        "<section class=\"card table-card\">"
        f"<h2>{html.escape(title)}</h2>"
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
        "</section>"
    )


def _bar_chart_svg(title: str, data: dict[str, int | float], *, width: int = 760, height: int = 360) -> str:
    chart_left = 90
    chart_right = width - 32
    chart_top = 52
    chart_bottom = height - 72
    chart_width = chart_right - chart_left
    chart_height = chart_bottom - chart_top
    items = list(data.items())
    if not items:
        items = [("No data", 0)]

    max_value = max(float(value) for _, value in items) if items else 1.0
    max_value = max(max_value, 1.0)
    bar_gap = 18
    bar_width = max(24.0, (chart_width - (len(items) - 1) * bar_gap) / max(len(items), 1))

    bars: list[str] = []
    for index, (label, value) in enumerate(items):
        numeric_value = float(value)
        bar_height = (numeric_value / max_value) * chart_height if max_value else 0.0
        x = chart_left + index * (bar_width + bar_gap)
        y = chart_bottom - bar_height
        color = _PALETTE[index % len(_PALETTE)]
        bars.append(
            f"<rect x=\"{x:.1f}\" y=\"{y:.1f}\" width=\"{bar_width:.1f}\" "
            f"height=\"{bar_height:.1f}\" rx=\"8\" fill=\"{color}\" />"
        )
        bars.append(
            f"<text x=\"{x + bar_width / 2:.1f}\" y=\"{chart_bottom + 18:.1f}\" "
            "text-anchor=\"middle\" class=\"label\">"
            f"{html.escape(label)}</text>"
        )
        bars.append(
            f"<text x=\"{x + bar_width / 2:.1f}\" y=\"{max(y - 8, chart_top + 14):.1f}\" "
            "text-anchor=\"middle\" class=\"value\">"
            f"{html.escape(_fmt_number(value))}</text>"
        )

    grid: list[str] = []
    for step in range(5):
        y = chart_top + (chart_height / 4) * step
        value = max_value - (max_value / 4) * step
        grid.append(
            f"<line x1=\"{chart_left}\" y1=\"{y:.1f}\" x2=\"{chart_right}\" y2=\"{y:.1f}\" "
            "stroke=\"#dbe4f0\" stroke-width=\"1\" />"
        )
        grid.append(
            f"<text x=\"{chart_left - 10}\" y=\"{y + 4:.1f}\" text-anchor=\"end\" class=\"axis\">"
            f"{html.escape(_fmt_number(round(value, 2)))}</text>"
        )

    return (
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" "
        f"viewBox=\"0 0 {width} {height}\">"
        "<style>"
        ".title{font:700 22px Arial,sans-serif;fill:#0f172a}"
        ".axis{font:12px Arial,sans-serif;fill:#475569}"
        ".label{font:12px Arial,sans-serif;fill:#1e293b}"
        ".value{font:12px Arial,sans-serif;fill:#0f172a}"
        "</style>"
        f"<rect width=\"{width}\" height=\"{height}\" fill=\"#ffffff\"/>"
        f"<text x=\"24\" y=\"32\" class=\"title\">{html.escape(title)}</text>"
        f"<line x1=\"{chart_left}\" y1=\"{chart_bottom}\" x2=\"{chart_right}\" y2=\"{chart_bottom}\" stroke=\"#94a3b8\" stroke-width=\"2\" />"
        f"<line x1=\"{chart_left}\" y1=\"{chart_top}\" x2=\"{chart_left}\" y2=\"{chart_bottom}\" stroke=\"#94a3b8\" stroke-width=\"2\" />"
        f"{''.join(grid)}"
        f"{''.join(bars)}"
        "</svg>"
    )


def _pie_chart_svg(title: str, data: dict[str, int | float], *, width: int = 760, height: int = 360) -> str:
    items = [(label, float(value)) for label, value in data.items() if float(value) > 0]
    if not items:
        items = [("No data", 1.0)]

    cx, cy, radius = 180.0, height / 2, 110.0
    total = sum(value for _, value in items)
    start_angle = -math.pi / 2
    segments: list[str] = []
    legend: list[str] = []

    for index, (label, value) in enumerate(items):
        sweep = (value / total) * math.tau if total else 0.0
        end_angle = start_angle + sweep
        x1 = cx + radius * math.cos(start_angle)
        y1 = cy + radius * math.sin(start_angle)
        x2 = cx + radius * math.cos(end_angle)
        y2 = cy + radius * math.sin(end_angle)
        large_arc = 1 if sweep > math.pi else 0
        color = _PALETTE[index % len(_PALETTE)]
        if abs(sweep - math.tau) < 1e-6:
            segments.append(
                f"<circle cx=\"{cx}\" cy=\"{cy}\" r=\"{radius}\" fill=\"{color}\" />"
            )
        else:
            segments.append(
                f"<path d=\"M {cx:.2f} {cy:.2f} L {x1:.2f} {y1:.2f} A {radius:.2f} {radius:.2f} 0 {large_arc} 1 {x2:.2f} {y2:.2f} Z\" fill=\"{color}\" />"
            )
        percent = (value / total) * 100 if total else 0.0
        legend_y = 86 + index * 30
        legend.append(
            f"<rect x=\"410\" y=\"{legend_y - 12}\" width=\"16\" height=\"16\" rx=\"4\" fill=\"{color}\" />"
            f"<text x=\"436\" y=\"{legend_y}\" class=\"label\">{html.escape(_fmt_number(label))}: {html.escape(_fmt_number(value))} ({percent:.1f}%)</text>"
        )
        start_angle = end_angle

    return (
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" "
        f"viewBox=\"0 0 {width} {height}\">"
        "<style>"
        ".title{font:700 22px Arial,sans-serif;fill:#0f172a}"
        ".label{font:14px Arial,sans-serif;fill:#1e293b}"
        "</style>"
        f"<rect width=\"{width}\" height=\"{height}\" fill=\"#ffffff\"/>"
        f"<text x=\"24\" y=\"32\" class=\"title\">{html.escape(title)}</text>"
        f"{''.join(segments)}"
        f"{''.join(legend)}"
        "</svg>"
    )


def _dashboard_html(
    title: str,
    cards: list[tuple[str, Any]],
    chart_files: list[tuple[str, str]],
    table_sections: list[str],
) -> str:
    stat_cards = "".join(
        "<div class=\"stat-card\">"
        f"<div class=\"stat-label\">{html.escape(label)}</div>"
        f"<div class=\"stat-value\">{html.escape(_fmt_number(value))}</div>"
        "</div>"
        for label, value in cards
    )
    chart_cards = "".join(
        "<section class=\"card chart-card\">"
        f"<h2>{html.escape(chart_title)}</h2>"
        f"<img src=\"{html.escape(filename)}\" alt=\"{html.escape(chart_title)}\" />"
        "</section>"
        for chart_title, filename in chart_files
    )
    tables = "".join(table_sections)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --card: #ffffff;
      --ink: #0f172a;
      --muted: #475569;
      --border: #dbe4f0;
      --shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 32px; font-family: Arial, sans-serif; background: linear-gradient(180deg, #eef5ff 0%, var(--bg) 100%); color: var(--ink); }}
    .shell {{ max-width: 1380px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; }}
    p.lede {{ margin: 0 0 24px; color: var(--muted); font-size: 16px; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .stat-card, .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 18px; box-shadow: var(--shadow); }}
    .stat-card {{ padding: 18px 20px; }}
    .stat-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px; }}
    .stat-value {{ font-size: 28px; font-weight: 700; }}
    .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; }}
    .card {{ padding: 20px; margin-bottom: 20px; }}
    .card h2 {{ margin: 0 0 14px; font-size: 20px; }}
    .chart-card img {{ width: 100%; border-radius: 12px; border: 1px solid var(--border); background: #fff; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 14px; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 700; text-transform: uppercase; font-size: 12px; letter-spacing: 0.04em; }}
    @media (max-width: 720px) {{ body {{ padding: 16px; }} .charts {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main class=\"shell\">
    <h1>{html.escape(title)}</h1>
    <p class=\"lede\">Static dashboard generated by the experiment pipeline for easier inspection than raw JSON or CSV outputs.</p>
    <section class=\"stats\">{stat_cards}</section>
    <section class=\"charts\">{chart_cards}</section>
    {tables}
  </main>
</body>
</html>
"""


def _sorted_items(data: dict[str, int | float]) -> list[tuple[str, float]]:
    return sorted(((label, float(value)) for label, value in data.items()), key=lambda item: item[1], reverse=True)


def _describe_distribution(data: dict[str, int | float], *, noun: str) -> str:
    items = _sorted_items(data)
    if not items:
        return f"No {noun} distribution was available for this run."
    if len(items) == 1:
        return f"All observed {noun} values fell into `{items[0][0]}`."

    top_label, top_value = items[0]
    second_label, second_value = items[1]
    total = sum(value for _, value in items)
    top_share = (top_value / total * 100) if total else 0.0
    gap = top_value - second_value
    if top_share >= 70:
        return f"`{top_label}` dominates the {noun} mix at {top_share:.1f}%, well above `{second_label}` by {gap:.1f}."
    if abs(top_value - second_value) < max(total * 0.1, 1.0):
        return f"The {noun} mix is relatively balanced, with `{top_label}` only slightly ahead of `{second_label}`."
    return f"`{top_label}` is the largest {noun} segment, ahead of `{second_label}` by {gap:.1f}."


def _describe_metric_trend(metrics: dict[str, int | float]) -> str:
    items = _sorted_items(metrics)
    if not items:
        return "No evaluation metrics were available for this run."
    strongest_label, strongest_value = items[0]
    weakest_label, weakest_value = items[-1]
    if strongest_value >= 0.9 and weakest_value >= 0.8:
        return f"All top-line metrics are strong; `{strongest_label}` leads at {strongest_value:.3f} and the weakest metric is still {weakest_value:.3f}."
    return f"`{strongest_label}` is the strongest metric at {strongest_value:.3f}, while `{weakest_label}` is the main constraint at {weakest_value:.3f}."


def _markdown_section(title: str, image_name: str, source: str, trend: str, description: str) -> str:
    return "\n".join([
        f"## {title}",
        "",
        f"![{title}]({image_name})",
        "",
        f"**What this visual is:** {description}",
        "",
        f"**Where it came from:** {source}",
        "",
        f"**Trend seen:** {trend}",
        "",
    ])


def _run_visualization_markdown(summary: dict[str, Any], eval_report: dict[str, Any]) -> str:
    label_dist = summary.get("label_types", {}).get("mapped_labels", {})
    modality_dist = summary.get("modality_distribution", {})
    context_dist = {
        "with context": summary.get("records_with_context", 0),
        "without context": max(summary.get("sample_count", 0) - summary.get("records_with_context", 0), 0),
    }
    metrics = {
        key: eval_report.get("overall", {}).get(key, 0)
        for key in ["accuracy", "precision", "recall", "f1"]
    }

    sections = [
        f"# {summary.get('dataset', 'Dataset')} Visualization Report",
        "",
        "This file explains the generated visualization artifacts for the current run and summarizes the main pattern visible in each chart.",
        "",
        _markdown_section(
            "Mapped Label Distribution",
            "label_distribution_pie.svg",
            "Derived from `dataset_summary.json -> label_types.mapped_labels` after cleaning and any optional balanced sampling.",
            _describe_distribution(label_dist, noun="label"),
            "A pie chart showing how many cleaned samples map to each binary target label.",
        ),
        _markdown_section(
            "Modality Distribution",
            "modality_distribution_bar.svg",
            "Derived from `dataset_summary.json -> modality_distribution`, using the normalized records written by the pipeline.",
            _describe_distribution(modality_dist, noun="modality"),
            "A bar chart showing the count of text-only versus multimodal records that survived cleaning.",
        ),
        _markdown_section(
            "Context Coverage",
            "context_coverage_pie.svg",
            "Derived from `dataset_summary.json -> records_with_context` and `sample_count`.",
            _describe_distribution(context_dist, noun="context coverage"),
            "A pie chart comparing records that include context text with records that do not.",
        ),
        _markdown_section(
            "Overall Metrics",
            "overall_metrics_bar.svg",
            "Derived from `evaluation_report.json -> overall` after prediction and scoring.",
            _describe_metric_trend(metrics),
            "A bar chart comparing the top-line evaluation metrics for the current run.",
        ),
    ]
    return "\n".join(sections)


def _aggregate_visualization_markdown(
    run_summaries: list[dict[str, Any]],
    aggregate_eval: dict[str, Any],
) -> str:
    counts = {
        summary.get("dataset", f"dataset_{index}"): summary.get("records_after_cleaning", 0)
        for index, summary in enumerate(run_summaries)
    }
    f1_scores = {
        summary.get("dataset", f"dataset_{index}"): summary.get("f1", 0) or 0
        for index, summary in enumerate(run_summaries)
    }
    accuracy_scores = {
        summary.get("dataset", f"dataset_{index}"): summary.get("accuracy", 0) or 0
        for index, summary in enumerate(run_summaries)
    }
    label_dist = aggregate_eval.get("label_distribution", {}).get("ground_truth", {})

    sections = [
        "# Aggregate Visualization Report",
        "",
        "This file explains the cross-dataset visualization artifacts created for a multi-dataset run.",
        "",
        _markdown_section(
            "Records After Cleaning",
            "dataset_sizes_bar.svg",
            "Derived from `aggregate_summary.json -> per_dataset[].records_after_cleaning`.",
            _describe_distribution(counts, noun="dataset size"),
            "A bar chart comparing how many records remained in each dataset after cleaning and optional balancing.",
        ),
        _markdown_section(
            "F1 by Dataset",
            "dataset_f1_bar.svg",
            "Derived from `aggregate_summary.json -> per_dataset[].f1`.",
            _describe_distribution(f1_scores, noun="F1 score"),
            "A bar chart comparing the final F1 score achieved on each dataset.",
        ),
        _markdown_section(
            "Accuracy by Dataset",
            "dataset_accuracy_bar.svg",
            "Derived from `aggregate_summary.json -> per_dataset[].accuracy`.",
            _describe_distribution(accuracy_scores, noun="accuracy score"),
            "A bar chart comparing overall accuracy across datasets.",
        ),
        _markdown_section(
            "Aggregate Ground-Truth Labels",
            "aggregate_labels_pie.svg",
            "Derived from `aggregate_evaluation.json -> label_distribution.ground_truth`.",
            _describe_distribution(label_dist, noun="ground-truth label"),
            "A pie chart showing the combined label mix across all datasets included in the run.",
        ),
    ]
    return "\n".join(sections)


def generate_run_visualizations(
    records: list[UnifiedRecord],
    summary: dict[str, Any],
    eval_report: dict[str, Any],
    predictions: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, str]:
    """Write static charts and an HTML dashboard for a single pipeline run."""
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    label_dist = summary.get("label_types", {}).get("mapped_labels", {})
    modality_dist = summary.get("modality_distribution", {})
    context_dist = {
        "with context": summary.get("records_with_context", 0),
        "without context": max(summary.get("sample_count", 0) - summary.get("records_with_context", 0), 0),
    }
    metrics = {
        key: eval_report.get("overall", {}).get(key, 0)
        for key in ["accuracy", "precision", "recall", "f1"]
    }

    charts = {
        "label_distribution_pie.svg": _pie_chart_svg("Mapped Label Distribution", label_dist),
        "modality_distribution_bar.svg": _bar_chart_svg("Modality Distribution", modality_dist),
        "context_coverage_pie.svg": _pie_chart_svg("Context Coverage", context_dist),
        "overall_metrics_bar.svg": _bar_chart_svg("Overall Metrics", metrics),
    }
    for filename, content in charts.items():
        _write_text(viz_dir / filename, content)

    metadata_rows = [
        [field, count]
        for field, count in list(summary.get("metadata_fields_available", {}).items())[:12]
    ]
    prediction_rows = [
        [
            pred.get("id", ""),
            pred.get("ground_truth_label_name", pred.get("ground_truth_label", "")),
            pred.get("predicted_label_name", pred.get("predicted_label", "")),
            pred.get("confidence", ""),
        ]
        for pred in predictions[:12]
    ]
    label_rows = [
        [label, count]
        for label, count in label_dist.items()
    ]

    dashboard = _dashboard_html(
        title=f"{summary.get('dataset', 'Dataset')} Visualization Dashboard",
        cards=[
            ("Samples", summary.get("sample_count", 0)),
            ("Retention", summary.get("retention_rate", "n/a")),
            ("Accuracy", eval_report.get("overall", {}).get("accuracy", "n/a")),
            ("F1", eval_report.get("overall", {}).get("f1", "n/a")),
            ("With context", summary.get("records_with_context", 0)),
            ("With image", summary.get("records_with_image", 0)),
        ],
        chart_files=[
            ("Mapped Label Distribution", "label_distribution_pie.svg"),
            ("Modality Distribution", "modality_distribution_bar.svg"),
            ("Context Coverage", "context_coverage_pie.svg"),
            ("Overall Metrics", "overall_metrics_bar.svg"),
        ],
        table_sections=[
            _table_html("Mapped Labels", ["Label", "Count"], label_rows),
            _table_html("Metadata Field Coverage", ["Field", "Records"], metadata_rows),
            _table_html("Prediction Sample", ["Sample ID", "Ground Truth", "Predicted", "Confidence"], prediction_rows),
        ],
    )
    _write_text(viz_dir / "dashboard.html", dashboard)
    _write_text(viz_dir / "visualization_report.md", _run_visualization_markdown(summary, eval_report))

    manifest = {
        "dashboard": str(viz_dir / "dashboard.html"),
        "report": str(viz_dir / "visualization_report.md"),
        "charts": [str(viz_dir / filename) for filename in charts],
    }
    _write_text(viz_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def generate_aggregate_visualizations(
    run_summaries: list[dict[str, Any]],
    aggregate_eval: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    """Write static charts and an HTML dashboard for multi-dataset runs."""
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    counts = {summary.get("dataset", f"dataset_{index}"): summary.get("records_after_cleaning", 0)
              for index, summary in enumerate(run_summaries)}
    f1_scores = {summary.get("dataset", f"dataset_{index}"): summary.get("f1", 0) or 0
                 for index, summary in enumerate(run_summaries)}
    accuracy_scores = {summary.get("dataset", f"dataset_{index}"): summary.get("accuracy", 0) or 0
                       for index, summary in enumerate(run_summaries)}
    label_dist = aggregate_eval.get("label_distribution", {}).get("ground_truth", {})

    charts = {
        "dataset_sizes_bar.svg": _bar_chart_svg("Records After Cleaning", counts),
        "dataset_f1_bar.svg": _bar_chart_svg("F1 by Dataset", f1_scores),
        "dataset_accuracy_bar.svg": _bar_chart_svg("Accuracy by Dataset", accuracy_scores),
        "aggregate_labels_pie.svg": _pie_chart_svg("Aggregate Ground-Truth Labels", label_dist),
    }
    for filename, content in charts.items():
        _write_text(viz_dir / filename, content)

    rows = [
        [
            summary.get("dataset", ""),
            summary.get("records_after_cleaning", 0),
            summary.get("accuracy", "n/a"),
            summary.get("f1", "n/a"),
        ]
        for summary in run_summaries
    ]
    dashboard = _dashboard_html(
        title="Aggregate Visualization Dashboard",
        cards=[
            ("Datasets", len(run_summaries)),
            ("Predictions", aggregate_eval.get("prediction_count", 0)),
            ("Accuracy", aggregate_eval.get("overall", {}).get("accuracy", "n/a")),
            ("F1", aggregate_eval.get("overall", {}).get("f1", "n/a")),
        ],
        chart_files=[
            ("Records After Cleaning", "dataset_sizes_bar.svg"),
            ("F1 by Dataset", "dataset_f1_bar.svg"),
            ("Accuracy by Dataset", "dataset_accuracy_bar.svg"),
            ("Aggregate Ground-Truth Labels", "aggregate_labels_pie.svg"),
        ],
        table_sections=[
            _table_html("Per-Dataset Metrics", ["Dataset", "Records", "Accuracy", "F1"], rows),
        ],
    )
    _write_text(viz_dir / "dashboard.html", dashboard)
    _write_text(
        viz_dir / "visualization_report.md",
        _aggregate_visualization_markdown(run_summaries, aggregate_eval),
    )

    manifest = {
        "dashboard": str(viz_dir / "dashboard.html"),
        "report": str(viz_dir / "visualization_report.md"),
        "charts": [str(viz_dir / filename) for filename in charts],
    }
    _write_text(viz_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest