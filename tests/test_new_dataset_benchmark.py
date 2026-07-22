from __future__ import annotations

import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from evaluate_outputs import add_metric, summarize as summarize_metrics  # noqa: E402
from build_comparison_report import relative_reduction  # noqa: E402
from prepare_freedom_ai_labs_dataset import load_source_rows  # noqa: E402
from run_qwen3_dataset import clean_completion, summarize as summarize_timing  # noqa: E402


class DatasetConversionTests(unittest.TestCase):
    def test_reads_expected_workbook_columns(self) -> None:
        from openpyxl import Workbook

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["transcript", "transcript_pred"])
            sheet.append(["Дұрыс мәтін.", "Дурыс мәтін"])
            workbook.save(path)

            rows = load_source_rows(path, None, "transcript", "transcript_pred")

        self.assertEqual(rows, [(2, "Дұрыс мәтін.", "Дурыс мәтін")])


class EvaluationTests(unittest.TestCase):
    def test_reports_macro_and_corpus_error_rates(self) -> None:
        grouped = defaultdict(list)
        add_metric(grouped, "model", "mix", "asr", "a b c", "a b c")
        add_metric(grouped, "model", "mix", "asr", "wrong words", "x")

        row = summarize_metrics(grouped)[0]

        self.assertEqual(row["wer"], "1.000000")
        self.assertEqual(row["corpus_wer"], "0.500000")

    def test_relative_error_reduction_is_positive_for_improvement(self) -> None:
        self.assertEqual(relative_reduction("0.15", "0.20"), "0.250000")


class RunnerHelperTests(unittest.TestCase):
    def test_removes_thinking_and_preamble(self) -> None:
        value = '<think>ignore this</think> Here is the cleaned transcript: "Сәлем!"'
        self.assertEqual(clean_completion(value), "Сәлем!")

    def test_timing_summary_uses_aggregate_throughput(self) -> None:
        runs = [
            {
                "request_seconds": 1.1,
                "generation_seconds": 1.0,
                "generated_tokens": 10,
                "peak_memory_gib": 8.0,
            },
            {
                "request_seconds": 2.2,
                "generation_seconds": 2.0,
                "generated_tokens": 40,
                "peak_memory_gib": 9.0,
            },
        ]

        summary = summarize_timing(runs)

        self.assertAlmostEqual(summary["overall_tokens_per_second"], 50 / 3)
        self.assertEqual(summary["max_peak_memory_gib"], 9.0)


if __name__ == "__main__":
    unittest.main()
