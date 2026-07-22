#!/usr/bin/env python3
"""Join model accuracy and timing summaries into one comparison CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--timing", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-model", default="baseline_asr")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_timing_reports(paths: list[Path]) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        model = str(report.get("model_name", ""))
        if not model:
            raise SystemExit(f"{path}: missing model_name")
        if model in reports:
            raise SystemExit(f"Duplicate timing report for model: {model}")
        reports[model] = report
    return reports


def relative_reduction(model_value: str, baseline_value: str) -> str:
    model = float(model_value)
    baseline = float(baseline_value)
    if baseline == 0:
        return ""
    return f"{(baseline - model) / baseline:.6f}"


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {args.output}; pass --overwrite explicitly.")

    with args.evaluation.open("r", encoding="utf-8", newline="") as handle:
        evaluation_rows = list(csv.DictReader(handle))
    if not evaluation_rows:
        raise SystemExit("Evaluation report has no data rows.")
    if len({row["model"] for row in evaluation_rows}) != len(evaluation_rows):
        raise SystemExit(
            "Evaluation report contains multiple rows per model. Use the *_by_language.csv report "
            "for this single-language benchmark."
        )

    by_model = {row["model"]: row for row in evaluation_rows}
    baseline = by_model.get(args.baseline_model)
    if not baseline:
        raise SystemExit(f"Baseline model {args.baseline_model!r} was not found in the evaluation report.")
    timing_reports = load_timing_reports(args.timing)

    missing_timings = sorted(set(by_model) - {args.baseline_model} - set(timing_reports))
    if missing_timings:
        raise SystemExit(f"Missing timing report(s) for: {', '.join(missing_timings)}")

    fieldnames = [
        "model",
        "language",
        "rows",
        "wer",
        "wer_relative_reduction_vs_baseline",
        "cer",
        "cer_relative_reduction_vs_baseline",
        "corpus_wer",
        "corpus_wer_relative_reduction_vs_baseline",
        "corpus_cer",
        "corpus_cer_relative_reduction_vs_baseline",
        "exact_norm",
        "chrf",
        "chrf_pp",
        "backend",
        "hardware",
        "mean_request_seconds",
        "median_request_seconds",
        "p95_request_seconds",
        "mean_generation_seconds",
        "overall_tokens_per_second",
        "max_peak_memory_gib",
    ]
    output_rows: list[dict[str, str]] = []
    for evaluation in evaluation_rows:
        model = evaluation["model"]
        timing = timing_reports.get(model, {})
        timing_summary = timing.get("summary", {})
        is_baseline = model == args.baseline_model
        output_rows.append(
            {
                "model": model,
                "language": evaluation["language"],
                "rows": evaluation["rows"],
                "wer": evaluation["wer"],
                "wer_relative_reduction_vs_baseline": ""
                if is_baseline
                else relative_reduction(evaluation["wer"], baseline["wer"]),
                "cer": evaluation["cer"],
                "cer_relative_reduction_vs_baseline": ""
                if is_baseline
                else relative_reduction(evaluation["cer"], baseline["cer"]),
                "corpus_wer": evaluation["corpus_wer"],
                "corpus_wer_relative_reduction_vs_baseline": ""
                if is_baseline
                else relative_reduction(evaluation["corpus_wer"], baseline["corpus_wer"]),
                "corpus_cer": evaluation["corpus_cer"],
                "corpus_cer_relative_reduction_vs_baseline": ""
                if is_baseline
                else relative_reduction(evaluation["corpus_cer"], baseline["corpus_cer"]),
                "exact_norm": evaluation["exact_norm"],
                "chrf": evaluation["chrf"],
                "chrf_pp": evaluation["chrf_pp"],
                "backend": str(timing.get("backend", "")),
                "hardware": str(timing.get("hardware", "")),
                "mean_request_seconds": str(timing_summary.get("mean_request_seconds", "")),
                "median_request_seconds": str(timing_summary.get("median_request_seconds", "")),
                "p95_request_seconds": str(timing_summary.get("p95_request_seconds", "")),
                "mean_generation_seconds": str(timing_summary.get("mean_generation_seconds", "")),
                "overall_tokens_per_second": str(timing_summary.get("overall_tokens_per_second", "")),
                "max_peak_memory_gib": str(timing_summary.get("max_peak_memory_gib", "")),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote combined accuracy/speed comparison to {args.output}")


if __name__ == "__main__":
    main()
