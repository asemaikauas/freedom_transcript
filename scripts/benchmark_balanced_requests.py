#!/usr/bin/env python3
"""Benchmark balanced transcript-cleanup requests across languages and ASR systems."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

from benchmark_single_request import (
    DEFAULT_INPUT,
    DEFAULT_MODEL_ROOT,
    DEFAULT_PROMPT,
    LANGUAGE_NAMES,
    MODEL_SPECS,
    build_prompt,
    generate_once,
    generation_kwargs,
    load_tokenizer_and_model,
)


DEFAULT_LANGUAGES = ("kk", "ru")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_SPECS)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--model-path", type=Path, help="Override the complete path for this model.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=LANGUAGE_NAMES,
        default=list(DEFAULT_LANGUAGES),
        help="Target languages to benchmark (default: kk ru).",
    )
    parser.add_argument(
        "--rows-per-group",
        type=int,
        default=5,
        help="Rows selected from each language x ASR group.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing JSON result. Disabled by default.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str | int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [{**row, "_row_index": index} for index, row in enumerate(rows)]


def evenly_spaced_positions(length: int, count: int) -> list[int]:
    if count == 1:
        return [length // 2]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def select_balanced_rows(
    rows: list[dict[str, str | int]],
    languages: list[str],
    rows_per_group: int,
) -> list[dict[str, str | int]]:
    grouped: dict[tuple[str, str], list[dict[str, str | int]]] = defaultdict(list)
    for row in rows:
        language = str(row["language"])
        if language in languages:
            grouped[(language, str(row["asr_model"]))].append(row)

    if not grouped:
        raise SystemExit(f"No rows found for languages: {', '.join(languages)}")

    selected: list[dict[str, str | int]] = []
    for (language, asr_model), values in sorted(grouped.items()):
        if len(values) < rows_per_group:
            raise SystemExit(
                f"Group {language}/{asr_model} has {len(values)} rows; "
                f"cannot select {rows_per_group}."
            )
        # Cover short through long requests rather than drawing a length-biased
        # random sample. Row identity remains deterministic across all models.
        values.sort(key=lambda row: (len(str(row["raw_transcript"])), int(row["_row_index"])))
        positions = evenly_spaced_positions(len(values), rows_per_group)
        selected.extend(values[position] for position in positions)

    expected_groups = len(languages) * len({str(row["asr_model"]) for row in rows})
    actual_groups = len(grouped)
    if actual_groups != expected_groups:
        raise SystemExit(f"Expected {expected_groups} language/ASR groups, found {actual_groups}.")
    return selected


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(runs: list[dict]) -> dict[str, float | int]:
    latencies = [float(run["latency_seconds"]) for run in runs]
    total_seconds = sum(latencies)
    total_tokens = sum(int(run["generated_tokens"]) for run in runs)
    return {
        "requests": len(runs),
        "mean_latency_seconds": statistics.mean(latencies),
        "median_latency_seconds": statistics.median(latencies),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "total_generation_seconds": total_seconds,
        "total_generated_tokens": total_tokens,
        "overall_tokens_per_second": total_tokens / total_seconds if total_seconds else 0.0,
        "max_peak_memory_gib": max(float(run["peak_memory_gib"]) for run in runs),
    }


def grouped_summaries(runs: list[dict], fields: tuple[str, ...]) -> dict[str, dict]:
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[tuple(str(run[field]) for field in fields)].append(run)
    return {" / ".join(key): summarize(values) for key, values in sorted(grouped.items())}


def main() -> None:
    args = parse_args()
    if args.rows_per_group < 1:
        raise SystemExit("--rows-per-group must be at least 1.")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be at least 1.")
    if args.warmup < 0:
        raise SystemExit("--warmup cannot be negative.")
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be at least 1.")
    if len(set(args.languages)) != len(args.languages):
        raise SystemExit("--languages must not contain duplicates.")
    if args.json_output.exists() and not args.overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing result: {args.json_output}\n"
            "Choose another path or pass --overwrite explicitly."
        )
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU allocation is required.")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise SystemExit("This GPU does not support bfloat16; retry with --dtype float16.")

    spec = MODEL_SPECS[args.model]
    model_path = args.model_path or args.model_root / str(spec["directory"])
    if not model_path.exists():
        raise SystemExit(f"Model path does not exist: {model_path}")

    all_rows = load_rows(args.input)
    selected_rows = select_balanced_rows(all_rows, args.languages, args.rows_per_group)
    system_prompt = args.prompt_file.read_text(encoding="utf-8").strip()

    print(
        f"Selected {len(selected_rows)} rows: languages={','.join(args.languages)}, "
        f"rows_per_language_asr_group={args.rows_per_group}"
    )
    for row in selected_rows:
        print(
            f"SELECTED row={row['_row_index']} id={row['id']} language={row['language']} "
            f"asr={row['asr_model']} chars={len(str(row['raw_transcript']))}"
        )

    load_started = time.perf_counter()
    tokenizer, model = load_tokenizer_and_model(args.model, model_path, dtype)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    gen_kwargs = generation_kwargs(args.model, tokenizer, args.max_new_tokens)
    print(f"Loaded {spec['name']} in {load_seconds:.3f} s")

    def prepare(row: dict[str, str | int]):
        prompt = build_prompt(tokenizer, args.model, system_prompt, row)  # type: ignore[arg-type]
        return tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=args.model != "gemma3",
        ).to(model.device)

    warmup_inputs = prepare(selected_rows[0])
    for current in range(args.warmup):
        elapsed, tokens, _, _ = generate_once(model, warmup_inputs, gen_kwargs)
        print(f"Warmup {current + 1}/{args.warmup}: {elapsed:.3f} s, {tokens} tokens")

    runs: list[dict] = []
    total_requests = len(selected_rows) * args.repetitions
    request_number = 0
    for row in selected_rows:
        inputs = prepare(row)
        input_tokens = int(inputs["input_ids"].shape[-1])
        for repetition in range(args.repetitions):
            request_number += 1
            elapsed, tokens, peak_memory_gib, completion = generate_once(model, inputs, gen_kwargs)
            tokens_per_second = tokens / elapsed if elapsed else 0.0
            output_text = tokenizer.decode(completion, skip_special_tokens=True).strip()
            run = {
                "request_number": request_number,
                "repetition": repetition + 1,
                "row_index": int(row["_row_index"]),
                "row_id": row["id"],
                "language": row["language"],
                "asr_model": row["asr_model"],
                "raw_transcript": row["raw_transcript"],
                "reference_clean": row["reference_clean"],
                "input_tokens": input_tokens,
                "generated_tokens": tokens,
                "latency_seconds": elapsed,
                "tokens_per_second": tokens_per_second,
                "peak_memory_gib": peak_memory_gib,
                "output_text": output_text,
            }
            runs.append(run)
            print(
                f"Request {request_number}/{total_requests}: id={row['id']} "
                f"language={row['language']} asr={row['asr_model']} latency={elapsed:.3f}s "
                f"output_tokens={tokens} speed={tokens_per_second:.2f}tok/s"
            )

    result = {
        "benchmark": "balanced_transcript_cleanup_speed",
        "model_key": args.model,
        "model_name": spec["name"],
        "model_path": str(model_path),
        "gpu": torch.cuda.get_device_name(),
        "dtype": args.dtype,
        "languages": args.languages,
        "asr_models": sorted({str(row["asr_model"]) for row in selected_rows}),
        "rows_per_language_asr_group": args.rows_per_group,
        "selected_unique_rows": len(selected_rows),
        "repetitions_per_row": args.repetitions,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "model_load_seconds": load_seconds,
        "summary": summarize(runs),
        "by_language": grouped_summaries(runs, ("language",)),
        "by_asr_model": grouped_summaries(runs, ("asr_model",)),
        "by_language_asr": grouped_summaries(runs, ("language", "asr_model")),
        "runs": runs,
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("SUMMARY_JSON=" + json.dumps(result["summary"], ensure_ascii=False))
    print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
