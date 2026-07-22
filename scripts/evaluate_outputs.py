#!/usr/bin/env python3
"""Evaluate transcript-correction outputs against clean references."""

from __future__ import annotations

import argparse
import csv
import string
import unicodedata
from collections import defaultdict
from pathlib import Path

from sacrebleu.metrics import CHRF


DEFAULT_BENCHMARK = Path("data/llm_correction_benchmark.csv")
DEFAULT_SUMMARY_OUTPUT = Path("reports/llm_correction_eval_summary.csv")
DEFAULT_LANGUAGES = ("kk", "ru", "mix")
LANGUAGE_CHOICES = ("en", "kk", "ru", "mix")
CHRF_SCORER = CHRF(word_order=0)
CHRF_PP_SCORER = CHRF(word_order=2)


def normalize_text(text: str) -> str:
    kept: list[str] = []
    for char in text.lower():
        category = unicodedata.category(char)
        if char.isspace():
            kept.append(" ")
        elif category[0] in {"L", "N"}:
            kept.append(char)
        elif char in string.punctuation:
            kept.append(" ")
        else:
            kept.append(" ")
    return " ".join("".join(kept).split())


def edit_distance(left: list[str] | str, right: list[str] | str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            substitution = previous[j - 1] + int(left_item != right_item)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def wer(prediction: str, reference: str) -> float:
    pred_words = normalize_text(prediction).split()
    ref_words = normalize_text(reference).split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    return edit_distance(pred_words, ref_words) / len(ref_words)


def cer(prediction: str, reference: str) -> float:
    pred_chars = normalize_text(prediction).replace(" ", "")
    ref_chars = normalize_text(reference).replace(" ", "")
    if not ref_chars:
        return 0.0 if not pred_chars else 1.0
    return edit_distance(pred_chars, ref_chars) / len(ref_chars)


def chrf(prediction: str, reference: str) -> float:
    return CHRF_SCORER.sentence_score(normalize_text(prediction), [normalize_text(reference)]).score


def chrf_pp(prediction: str, reference: str) -> float:
    return CHRF_PP_SCORER.sentence_score(normalize_text(prediction), [normalize_text(reference)]).score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--outputs",
        type=Path,
        action="append",
        default=None,
        help="Prediction CSV. Repeat --outputs to compare multiple models in one report.",
    )
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument(
        "--language-summary-output",
        type=Path,
        default=None,
        help="One-row-per-language report path (default: <summary-output>_by_language.csv).",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=LANGUAGE_CHOICES,
        default=list(DEFAULT_LANGUAGES),
        help="Languages to evaluate (default: kk ru mix; English is skipped).",
    )
    parser.add_argument("--model", default=None, help="Model name if outputs CSV has no model column.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permit prediction files that do not cover every selected benchmark row.",
    )
    return parser.parse_args()


def load_benchmark(path: Path, languages: list[str]) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["language"] in languages]
    benchmark = {row["id"]: row for row in rows}
    if len(benchmark) != len(rows):
        raise SystemExit("Benchmark CSV contains duplicate IDs in the selected languages.")
    return benchmark


def add_metric(
    grouped: dict[tuple[str, str, str], list[dict[str, float | int]]],
    model: str,
    language: str,
    asr_model: str,
    prediction: str,
    reference: str,
) -> None:
    pred_words = normalize_text(prediction).split()
    ref_words = normalize_text(reference).split()
    pred_chars = normalize_text(prediction).replace(" ", "")
    ref_chars = normalize_text(reference).replace(" ", "")
    grouped[(model, language, asr_model)].append(
        {
            "wer": wer(prediction, reference),
            "cer": cer(prediction, reference),
            "word_edits": edit_distance(pred_words, ref_words),
            "reference_words": len(ref_words),
            "char_edits": edit_distance(pred_chars, ref_chars),
            "reference_chars": len(ref_chars),
            "exact_norm": float(normalize_text(prediction) == normalize_text(reference)),
            "chrf": chrf(prediction, reference),
            "chrf_pp": chrf_pp(prediction, reference),
        }
    )


def corpus_error_rate(values: list[dict[str, float | int]], edits_key: str, units_key: str) -> float:
    edits = sum(int(item[edits_key]) for item in values)
    units = sum(int(item[units_key]) for item in values)
    if not units:
        return 0.0 if not edits else 1.0
    return edits / units


def summarize(
    grouped: dict[tuple[str, str, str], list[dict[str, float | int]]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for (model, language, asr_model), values in sorted(grouped.items()):
        count = len(values)
        rows.append(
            {
                "model": model,
                "language": language,
                "asr_model": asr_model,
                "rows": str(count),
                "wer": f"{sum(item['wer'] for item in values) / count:.6f}",
                "cer": f"{sum(item['cer'] for item in values) / count:.6f}",
                "corpus_wer": f"{corpus_error_rate(values, 'word_edits', 'reference_words'):.6f}",
                "corpus_cer": f"{corpus_error_rate(values, 'char_edits', 'reference_chars'):.6f}",
                "exact_norm": f"{sum(item['exact_norm'] for item in values) / count:.6f}",
                "chrf": f"{sum(item['chrf'] for item in values) / count:.6f}",
                "chrf_pp": f"{sum(item['chrf_pp'] for item in values) / count:.6f}",
            }
        )
    return rows


def summarize_by_language(
    grouped: dict[tuple[str, str, str], list[dict[str, float | int]]],
) -> list[dict[str, str]]:
    by_language: dict[tuple[str, str], list[dict[str, float | int]]] = defaultdict(list)
    for (model, language, _asr_model), values in grouped.items():
        by_language[(model, language)].extend(values)

    rows: list[dict[str, str]] = []
    for (model, language), values in sorted(by_language.items()):
        count = len(values)
        rows.append(
            {
                "model": model,
                "language": language,
                "rows": str(count),
                "wer": f"{sum(item['wer'] for item in values) / count:.6f}",
                "cer": f"{sum(item['cer'] for item in values) / count:.6f}",
                "corpus_wer": f"{corpus_error_rate(values, 'word_edits', 'reference_words'):.6f}",
                "corpus_cer": f"{corpus_error_rate(values, 'char_edits', 'reference_chars'):.6f}",
                "exact_norm": f"{sum(item['exact_norm'] for item in values) / count:.6f}",
                "chrf": f"{sum(item['chrf'] for item in values) / count:.6f}",
                "chrf_pp": f"{sum(item['chrf_pp'] for item in values) / count:.6f}",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if len(set(args.languages)) != len(args.languages):
        raise SystemExit("--languages must not contain duplicates.")
    benchmark = load_benchmark(args.benchmark, args.languages)
    if not benchmark:
        raise SystemExit(f"No benchmark rows found for languages: {', '.join(args.languages)}")
    grouped: dict[tuple[str, str, str], list[dict[str, float | int]]] = defaultdict(list)

    for row in benchmark.values():
        add_metric(
            grouped,
            "baseline_asr",
            row["language"],
            row["asr_model"],
            row["raw_transcript"],
            row["reference_clean"],
        )

    prediction_keys: set[tuple[str, str]] = set()
    if args.outputs:
        for output_path in args.outputs:
            if not output_path.exists():
                raise SystemExit(f"Prediction file does not exist: {output_path}")
            selected_ids_seen: set[str] = set()
            ids_seen_in_file: set[str] = set()
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                required = {"id", "cleaned_text"}
                if not reader.fieldnames or not required.issubset(reader.fieldnames):
                    raise SystemExit(f"{output_path}: outputs must contain at least: id, cleaned_text")

                for output_row in reader:
                    row_id = output_row["id"]
                    if row_id in ids_seen_in_file:
                        raise SystemExit(f"{output_path}: duplicate prediction ID: {row_id}")
                    ids_seen_in_file.add(row_id)
                    benchmark_row = benchmark.get(row_id)
                    if not benchmark_row:
                        continue
                    selected_ids_seen.add(row_id)
                    model = output_row.get("model") or args.model
                    if not model:
                        raise SystemExit("Provide a model column in outputs CSV or pass --model.")
                    prediction_key = (model, row_id)
                    if prediction_key in prediction_keys:
                        raise SystemExit(f"Duplicate model prediction across output files: {model} / {row_id}")
                    prediction_keys.add(prediction_key)
                    add_metric(
                        grouped,
                        model,
                        benchmark_row["language"],
                        benchmark_row["asr_model"],
                        output_row["cleaned_text"],
                        benchmark_row["reference_clean"],
                    )

            missing_ids = sorted(set(benchmark) - selected_ids_seen)
            if missing_ids and not args.allow_partial:
                preview = ", ".join(missing_ids[:5])
                raise SystemExit(
                    f"{output_path}: missing {len(missing_ids)} selected benchmark IDs "
                    f"(first: {preview}). Pass --allow-partial only for smoke tests."
                )

    summary_rows = summarize(grouped)
    language_summary_rows = summarize_by_language(grouped)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            lineterminator="\n",
            fieldnames=[
                "model",
                "language",
                "asr_model",
                "rows",
                "wer",
                "cer",
                "corpus_wer",
                "corpus_cer",
                "exact_norm",
                "chrf",
                "chrf_pp",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    language_summary_output = args.language_summary_output
    if language_summary_output is None:
        language_summary_output = args.summary_output.with_name(
            f"{args.summary_output.stem}_by_language{args.summary_output.suffix}"
        )
    language_summary_output.parent.mkdir(parents=True, exist_ok=True)
    with language_summary_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            lineterminator="\n",
            fieldnames=[
                "model",
                "language",
                "rows",
                "wer",
                "cer",
                "corpus_wer",
                "corpus_cer",
                "exact_norm",
                "chrf",
                "chrf_pp",
            ],
        )
        writer.writeheader()
        writer.writerows(language_summary_rows)

    print(f"Wrote evaluation summary to {args.summary_output}")
    print(f"Wrote language summary to {language_summary_output}")
    print(f"Evaluated languages: {', '.join(args.languages)} ({len(benchmark)} benchmark rows)")
    for row in summary_rows[:12]:
        print(
            f"{row['model']}\t{row['language']}\t{row['asr_model']}\t"
            f"WER={row['wer']}\tCER={row['cer']}\t"
            f"corpus_WER={row['corpus_wer']}\tcorpus_CER={row['corpus_cer']}\t"
            f"chrF++={row['chrf_pp']}\tN={row['rows']}"
        )


if __name__ == "__main__":
    main()
