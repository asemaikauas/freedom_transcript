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
from run_qwen3_dataset import (  # noqa: E402
    apply_structured_edits,
    build_prompt,
    clean_completion,
    parse_edit_response,
    summarize as summarize_timing,
)


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

    def test_parses_json_object_and_markdown_fence(self) -> None:
        value = """```json
        {"edits":[{"from":"тіркеден","to":"тіркеуден","type":"grammar"}]}
        ```"""

        edits, error = parse_edit_response(value)

        self.assertIsNone(error)
        self.assertEqual(edits[0]["from"], "тіркеден")
        self.assertEqual(edits[0]["to"], "тіркеуден")

    def test_parses_json_array_after_preamble(self) -> None:
        value = (
            'Proposed edits: [{"from":"тіркеден","to":"тіркеуден",'
            '"type":"grammar"}]'
        )

        edits, error = parse_edit_response(value)

        self.assertIsNone(error)
        self.assertEqual(len(edits), 1)

    def test_mixed_prompt_requires_language_preservation(self) -> None:
        class CaptureTokenizer:
            messages = None

            def apply_chat_template(self, messages, **_kwargs):
                self.messages = messages
                return "rendered"

        tokenizer = CaptureTokenizer()
        row = {
            "language": "mix",
            "raw_transcript": "Мен сегодня жұмыс істеймін.",
        }

        rendered = build_prompt(tokenizer, "system", row)

        self.assertEqual(rendered, "rendered")
        user_message = tokenizer.messages[1]["content"]
        self.assertIn("Preserve the language of every source segment", user_message)
        self.assertNotIn("Target language:", user_message)

    def test_applies_safe_local_grammar_edit(self) -> None:
        original = "Қазір тіркеден өткен болатынмын."
        response = '{"edits":[{"from":"тіркеден","to":"тіркеуден","type":"grammar"}]}'

        result = apply_structured_edits(original, response)

        self.assertEqual(result["output_text"], "Қазір тіркеуден өткен болатынмын.")
        self.assertEqual(len(result["applied_edits"]), 1)
        self.assertFalse(result["rejected_edits"])

    def test_applies_multiple_edits_from_right_to_left(self) -> None:
        original = "эээ мне нужэн новый пароль"
        response = (
            '{"edits":[{"from":"эээ ","to":"","type":"filler"},'
            '{"from":"нужэн","to":"нужен","type":"spelling"}]}'
        )

        result = apply_structured_edits(original, response)

        self.assertEqual(result["output_text"], "мне нужен новый пароль")
        self.assertEqual(len(result["applied_edits"]), 2)

    def test_invalid_json_falls_back_to_original(self) -> None:
        original = "Қазір тіркеден өткен болатынмын."

        result = apply_structured_edits(original, "Қазір тіркеуден өткен болатынмын.")

        self.assertEqual(result["output_text"], original)
        self.assertTrue(result["parse_error"])

    def test_ambiguous_source_requires_occurrence(self) -> None:
        original = "ыыы мәтін ыыы"
        ambiguous = '{"edits":[{"from":"ыыы","to":"","type":"filler"}]}'
        specific = (
            '{"edits":[{"from":"ыыы","to":"","type":"filler","occurrence":2}]}'
        )

        ambiguous_result = apply_structured_edits(original, ambiguous)
        specific_result = apply_structured_edits(original, specific)

        self.assertEqual(ambiguous_result["output_text"], original)
        self.assertEqual(ambiguous_result["rejected_edits"][0]["reason"], "source_not_unique")
        self.assertEqual(specific_result["output_text"], "ыыы мәтін ")

    def test_rejects_sentence_rewrite_over_change_budget(self) -> None:
        original = "Это исходное предложение остается на русском языке."
        response = (
            '{"edits":[{"from":"Это исходное предложение остается на русском языке.",'
            '"to":"Бұл сөйлем қазақ тіліне толық аударылды.",'
            '"type":"grammar"}]}'
        )

        result = apply_structured_edits(original, response)

        self.assertEqual(result["output_text"], original)
        self.assertEqual(result["rejected_edits"][0]["reason"], "change_budget_exceeded")

    def test_rejects_unexpected_language_change(self) -> None:
        original = "Если это возможно?"
        response = (
            '{"edits":[{"from":"Если это возможно?",'
            '"to":"Егер бұл мүмкін болса?","type":"grammar"}]}'
        )

        result = apply_structured_edits(original, response, max_change_ratio=1.0)

        self.assertEqual(result["output_text"], original)
        self.assertEqual(result["safety_fallback"], "unexpected_kazakh_translation")

    def test_rejects_number_change(self) -> None:
        original = "Мен 1995 жылы дүниеге келдім."
        response = '{"edits":[{"from":"1995","to":"1994","type":"grammar"}]}'

        result = apply_structured_edits(original, response)

        self.assertEqual(result["output_text"], original)
        self.assertEqual(result["safety_fallback"], "protected_number_changed")

    def test_rejects_repetition_loop(self) -> None:
        original = "Бұл жауап өте жақсы болды, мен оны кейін қайта тексеріп шығамын."
        response = (
            '{"edits":[{"from":"жақсы",'
            '"to":"қайта қайта қайта қайта қайта қайта","type":"grammar"}]}'
        )

        result = apply_structured_edits(original, response, max_change_ratio=1.0)

        self.assertEqual(result["output_text"], original)
        self.assertEqual(result["safety_fallback"], "repetition_loop")


if __name__ == "__main__":
    unittest.main()
