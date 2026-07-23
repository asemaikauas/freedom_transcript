#!/usr/bin/env python3
"""Run Qwen3-8B with Transformers or a llama.cpp GGUF server.

Both backends use the same prompt rendered by the same Hugging Face tokenizer.
The runner writes evaluator-compatible predictions plus per-request timing data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/freedom_ai_labs_benchmark.csv")
DEFAULT_PROMPT = Path("prompts/transcript_cleanup.txt")

LANGUAGE_NAMES = {
    "en": "English",
    "kk": "Kazakh",
    "ru": "Russian",
    "mix": "Kazakh-Russian mixed (code-switched)",
}

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
LEADING_THINK_CLOSE = re.compile(r"^\s*</think>\s*", re.IGNORECASE)
PREAMBLE_RE = re.compile(
    r"^(?:here(?:'s| is)|this is|the (?:cleaned|corrected)(?: up)? transcript(?: is)?)"
    r"\b[^:\n]{0,60}:\s*",
    re.IGNORECASE,
)
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
WORD_RE = re.compile(r"\w+", re.UNICODE)
DIGIT_RE = re.compile(r"\d+(?:[.,]\d+)?")
KAZAKH_SPECIFIC_LETTERS = frozenset("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
NUMBER_WORDS = frozenset(
    {
        "ноль",
        "один",
        "одна",
        "одно",
        "два",
        "две",
        "три",
        "четыре",
        "пять",
        "шесть",
        "семь",
        "восемь",
        "девять",
        "десять",
        "одиннадцать",
        "двенадцать",
        "тринадцать",
        "четырнадцать",
        "пятнадцать",
        "шестнадцать",
        "семнадцать",
        "восемнадцать",
        "девятнадцать",
        "двадцать",
        "тридцать",
        "сорок",
        "пятьдесят",
        "шестьдесят",
        "семьдесят",
        "восемьдесят",
        "девяносто",
        "сто",
        "тысяча",
        "тысячи",
        "тысяч",
        "миллион",
        "миллиона",
        "миллионов",
        "миллиард",
        "миллиарда",
        "миллиардов",
        "нөл",
        "бір",
        "екі",
        "үш",
        "төрт",
        "бес",
        "алты",
        "жеті",
        "сегіз",
        "тоғыз",
        "он",
        "жиырма",
        "отыз",
        "қырық",
        "елу",
        "алпыс",
        "жетпіс",
        "сексен",
        "тоқсан",
        "жүз",
        "мың",
    }
)
ALLOWED_EDIT_TYPES = frozenset(
    {
        "grammar",
        "spelling",
        "phonetic",
        "filler",
        "punctuation",
        "capitalization",
        "spacing",
        "repetition",
    }
)
SAFE_FILLER_FORMS = frozenset(
    {
        "ыыы",
        "ы-ы-ы",
        "эээ",
        "э-э-э",
        "ммм",
        "м-м-м",
        "і",
        "іі",
        "ііі",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        required=True,
        choices=("transformers", "llama-cpp", "llama-server"),
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-output", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Local Hugging Face model directory (required for Transformers).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="Qwen3 tokenizer directory (defaults to --model-path).",
    )
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8080",
        help="Base URL of an already-running llama-server.",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument(
        "--max-edits",
        type=int,
        default=8,
        help="Maximum structured edits accepted from one model response.",
    )
    parser.add_argument(
        "--max-change-ratio",
        type=float,
        default=0.15,
        help="Maximum fraction of source characters covered by accepted edit spans.",
    )
    parser.add_argument(
        "--max-edit-span-chars",
        type=int,
        default=80,
        help="Maximum source or replacement length for one local edit.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean_completion(text: str) -> str:
    text = THINK_BLOCK.sub("", text).strip()
    text = LEADING_THINK_CLOSE.sub("", text).strip()
    text = PREAMBLE_RE.sub("", text).strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] in "\"'":
        text = text[1:-1].strip()
    return text


def _load_json_response(text: str) -> Any:
    cleaned = CODE_FENCE_RE.sub("", clean_completion(text)).strip()
    candidates = [cleaned]
    fragments: list[tuple[int, str]] = []
    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")
    if object_start >= 0 and object_end > object_start:
        fragments.append((object_start, cleaned[object_start : object_end + 1]))
    array_start = cleaned.find("[")
    array_end = cleaned.rfind("]")
    if array_start >= 0 and array_end > array_start:
        fragments.append((array_start, cleaned[array_start : array_end + 1]))
    candidates.extend(fragment for _start, fragment in sorted(fragments))

    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def parse_edit_response(text: str) -> tuple[list[dict[str, Any]], str | None]:
    try:
        payload = _load_json_response(text)
    except json.JSONDecodeError as exc:
        return [], f"invalid_json:{exc.msg}"

    if isinstance(payload, dict):
        raw_edits = payload.get("edits")
    elif isinstance(payload, list):
        raw_edits = payload
    else:
        return [], "response_must_be_object_or_array"
    if not isinstance(raw_edits, list):
        return [], "edits_must_be_array"

    edits: list[dict[str, Any]] = []
    for index, item in enumerate(raw_edits):
        if not isinstance(item, dict):
            return [], f"edit_{index}_must_be_object"
        source = item.get("from")
        replacement = item.get("to")
        edit_type = item.get("type", "unspecified")
        occurrence = item.get("occurrence")
        if not isinstance(source, str) or not source:
            return [], f"edit_{index}_from_must_be_nonempty_string"
        if not isinstance(replacement, str):
            return [], f"edit_{index}_to_must_be_string"
        if not isinstance(edit_type, str) or not edit_type:
            return [], f"edit_{index}_type_must_be_string"
        if occurrence is not None and (
            not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 1
        ):
            return [], f"edit_{index}_occurrence_must_be_positive_integer"
        edits.append(
            {
                "from": source,
                "to": replacement,
                "type": edit_type,
                "occurrence": occurrence,
            }
        )
    return edits, None


def _occurrence_starts(text: str, substring: str) -> list[int]:
    starts: list[int] = []
    offset = 0
    while True:
        start = text.find(substring, offset)
        if start < 0:
            return starts
        starts.append(start)
        offset = start + len(substring)


def _number_signature(text: str) -> tuple[list[str], Counter[str]]:
    digits = DIGIT_RE.findall(text)
    words = Counter(token for token in WORD_RE.findall(text.casefold()) if token in NUMBER_WORDS)
    return digits, words


def _has_repetition_loop(text: str, threshold: int = 5) -> bool:
    previous = None
    run_length = 0
    for token in WORD_RE.findall(text.casefold()):
        if token == previous:
            run_length += 1
        else:
            previous = token
            run_length = 1
        if run_length >= threshold:
            return True
    return False


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + int(left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _alphanumeric_signature(text: str) -> str:
    return "".join(char.casefold() for char in text if char.isalnum())


def _typed_edit_error(original: str, edit: dict[str, Any], start: int) -> str | None:
    source = edit["from"]
    replacement = edit["to"]
    edit_type = edit["type"]

    if edit_type not in ALLOWED_EDIT_TYPES:
        return "unsupported_edit_type"

    if edit_type == "filler":
        if replacement:
            return "filler_must_be_deletion"
        if source.strip().casefold() not in SAFE_FILLER_FORMS:
            return "filler_not_allowlisted"
        return None

    if edit_type == "punctuation":
        if _alphanumeric_signature(source) != _alphanumeric_signature(replacement):
            return "punctuation_changed_text"
        return None

    if edit_type == "capitalization":
        if source.casefold() != replacement.casefold():
            return "capitalization_changed_text"
        return None

    if edit_type == "spacing":
        if "".join(source.split()) != "".join(replacement.split()):
            return "spacing_changed_text"
        return None

    if edit_type == "repetition":
        return "type_requires_review"

    if not WORD_RE.fullmatch(source) or not WORD_RE.fullmatch(replacement):
        return "unsafe_multiword_edit"
    if source[0].isdigit() or replacement[0].isdigit():
        return "protected_number_edit"
    if len(replacement) < len(source):
        return "unsafe_word_shortening"
    if _edit_distance(source.casefold(), replacement.casefold()) > 1:
        return "word_edit_distance_too_large"

    prefix = original[:start].rstrip()
    if source[0].isupper() and prefix and prefix[-1] not in ".!?\n":
        return "protected_capitalized_token"
    return None


def _final_safety_error(original: str, candidate: str) -> str | None:
    if not candidate.strip():
        return "empty_output"
    if _number_signature(original) != _number_signature(candidate):
        return "protected_number_changed"
    original_kazakh = sum(char in KAZAKH_SPECIFIC_LETTERS for char in original)
    candidate_kazakh = sum(char in KAZAKH_SPECIFIC_LETTERS for char in candidate)
    if original_kazakh == 0 and candidate_kazakh >= 2:
        return "unexpected_kazakh_translation"
    if original_kazakh >= 3 and candidate_kazakh < max(1, original_kazakh // 2):
        return "unexpected_kazakh_removal"
    if _has_repetition_loop(candidate):
        return "repetition_loop"
    return None


def apply_structured_edits(
    original: str,
    response: str,
    max_edits: int = 8,
    max_change_ratio: float = 0.15,
    max_edit_span_chars: int = 80,
) -> dict[str, Any]:
    proposed, parse_error = parse_edit_response(response)
    result: dict[str, Any] = {
        "output_text": original,
        "model_response": clean_completion(response),
        "proposed_edits": proposed,
        "applied_edits": [],
        "rejected_edits": [],
        "parse_error": parse_error,
        "safety_fallback": None,
    }
    if parse_error:
        return result
    if len(proposed) > max_edits:
        result["parse_error"] = f"too_many_edits:{len(proposed)}>{max_edits}"
        return result

    change_budget = max(24, math.ceil(len(original) * max_change_ratio))
    used_budget = 0
    spans: list[tuple[int, int]] = []
    accepted: list[dict[str, Any]] = []

    for edit in proposed:
        rejection: str | None = None
        source = edit["from"]
        replacement = edit["to"]
        if source == replacement:
            rejection = "no_op"
        elif max(len(source), len(replacement)) > max_edit_span_chars:
            rejection = "edit_span_too_large"

        starts = _occurrence_starts(original, source) if rejection is None else []
        occurrence = edit["occurrence"]
        if rejection is None:
            if occurrence is None and len(starts) != 1:
                rejection = "source_not_unique"
            elif occurrence is not None and occurrence > len(starts):
                rejection = "occurrence_not_found"
            elif not starts:
                rejection = "source_not_found"

        start = -1
        end = -1
        if rejection is None:
            start = starts[0] if occurrence is None else starts[occurrence - 1]
            end = start + len(source)
            if any(start < prior_end and prior_start < end for prior_start, prior_end in spans):
                rejection = "overlapping_edit"

        if rejection is None:
            rejection = _typed_edit_error(original, edit, start)

        edit_cost = max(len(source), len(replacement))
        if rejection is None and used_budget + edit_cost > change_budget:
            rejection = "change_budget_exceeded"

        if rejection is not None:
            result["rejected_edits"].append({"edit": edit, "reason": rejection})
            continue

        accepted_edit = {**edit, "start": start, "end": end}
        accepted.append(accepted_edit)
        spans.append((start, end))
        used_budget += edit_cost

    candidate = original
    for edit in sorted(accepted, key=lambda item: int(item["start"]), reverse=True):
        before = candidate[: edit["start"]]
        after = candidate[edit["end"] :]
        if edit["type"] == "filler" and not edit["to"]:
            if after[:1] in {",", ";", ":"}:
                after = after[1:]
            if before and after and before[-1].isspace() and after[0].isspace():
                after = after[1:]
            elif not before and after[:1].isspace():
                after = after[1:]
        candidate = before + edit["to"] + after

    safety_error = _final_safety_error(original, candidate)
    if safety_error:
        result["safety_fallback"] = safety_error
        result["rejected_edits"].extend(
            {"edit": edit, "reason": f"global:{safety_error}"} for edit in accepted
        )
        return result

    result["output_text"] = candidate
    result["applied_edits"] = accepted
    return result


def load_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "language", "raw_transcript", "reference_clean"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit(f"Benchmark CSV must contain: {', '.join(sorted(required))}")
        rows = list(reader)
    if len({row["id"] for row in rows}) != len(rows):
        raise SystemExit("Benchmark CSV contains duplicate IDs.")
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise SystemExit("No benchmark rows were selected.")
    return rows


def load_tokenizer(path: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required. Install requirements-inference.txt.") from exc
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_prompt(tokenizer, system_prompt: str, row: dict[str, str]) -> str:
    language = LANGUAGE_NAMES.get(row["language"], row["language"])
    if row["language"] == "mix":
        language_instruction = (
            "Language policy: the source may contain Kazakh, Russian, or both. "
            "Preserve the language of every source segment; never translate."
        )
    else:
        language_instruction = f"Target language: {language}."
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"{language_instruction}\n"
                "Input transcript as a JSON string:\n"
                f"{json.dumps(row['raw_transcript'], ensure_ascii=False)}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


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


def summarize(runs: list[dict[str, Any]]) -> dict[str, float | int | None]:
    request_seconds = [float(run["request_seconds"]) for run in runs]
    generation_seconds = [float(run["generation_seconds"]) for run in runs]
    generated_tokens = sum(int(run["generated_tokens"]) for run in runs)
    generation_total = sum(generation_seconds)
    peaks = [float(run["peak_memory_gib"]) for run in runs if run["peak_memory_gib"] is not None]
    result: dict[str, float | int | None] = {
        "requests": len(runs),
        "mean_request_seconds": statistics.mean(request_seconds),
        "median_request_seconds": statistics.median(request_seconds),
        "p95_request_seconds": percentile(request_seconds, 0.95),
        "mean_generation_seconds": statistics.mean(generation_seconds),
        "median_generation_seconds": statistics.median(generation_seconds),
        "p95_generation_seconds": percentile(generation_seconds, 0.95),
        "total_generation_seconds": generation_total,
        "total_generated_tokens": generated_tokens,
        "overall_tokens_per_second": generated_tokens / generation_total if generation_total else 0.0,
        "max_peak_memory_gib": max(peaks) if peaks else None,
    }
    if any("proposed_edits" in run for run in runs):
        result.update(
            {
                "edit_parse_failures": sum(bool(run.get("parse_error")) for run in runs),
                "safety_fallbacks": sum(bool(run.get("safety_fallback")) for run in runs),
                "proposed_edits": sum(len(run.get("proposed_edits", [])) for run in runs),
                "applied_edits": sum(len(run.get("applied_edits", [])) for run in runs),
                "rejected_edits": sum(len(run.get("rejected_edits", [])) for run in runs),
                "unchanged_outputs": sum(
                    run.get("output_text") == run.get("raw_transcript") for run in runs
                ),
            }
        )
    return result


def get_json(url: str, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not query {url}: {exc}") from exc


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama-server returned HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"llama-server request failed: {exc}") from exc


def gpu_name() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return os.environ.get("CUDA_VISIBLE_DEVICES", "unknown")
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return ", ".join(names) if names else os.environ.get("CUDA_VISIBLE_DEVICES", "unknown")


class TransformersBackend:
    def __init__(self, model_path: Path, tokenizer, dtype_name: str, seed: int):
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise SystemExit("PyTorch and transformers are required for this backend.") from exc

        if not torch.cuda.is_available():
            raise SystemExit("The Transformers backend requires a CUDA GPU allocation.")
        dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise SystemExit("This GPU does not support bfloat16; retry with --dtype float16.")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.torch = torch
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="cuda",
            local_files_only=True,
        )
        self.model.eval()
        self.hardware = torch.cuda.get_device_name()

    def run(self, prompt: str, max_new_tokens: int, seed: int) -> dict[str, Any]:
        torch = self.torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        request_started = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        torch.cuda.synchronize()
        generation_started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - generation_started
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        completion = generated[0, prompt_tokens:]
        text = self.tokenizer.decode(completion, skip_special_tokens=True)
        request_seconds = time.perf_counter() - request_started
        generated_tokens = int(completion.shape[-1])
        return {
            "text": text,
            "input_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "request_seconds": request_seconds,
            "generation_seconds": generation_seconds,
            "tokens_per_second": generated_tokens / generation_seconds if generation_seconds else 0.0,
            "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        }


class LlamaCppBackend:
    def __init__(
        self,
        model_path: Path,
        n_ctx: int,
        n_batch: int,
        n_gpu_layers: int,
        n_threads: int,
        seed: int,
    ):
        try:
            from llama_cpp import Llama
        except (ImportError, RuntimeError) as exc:
            raise SystemExit(
                "llama-cpp-python could not load. On Jubail, prepend the existing CUDA runtime "
                "directory to LD_LIBRARY_PATH before running the GGUF backend."
            ) from exc
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            seed=seed,
            verbose=False,
        )
        self.hardware = gpu_name()

    def run(self, prompt: str, max_new_tokens: int, seed: int) -> dict[str, Any]:
        # Clear the KV cache so every row pays its full prompt-prefill cost, just
        # like the Transformers backend. This prevents shared system-prompt
        # prefix caching from making the GGUF results artificially faster.
        self.model.reset()
        request_started = time.perf_counter()
        response = self.model.create_completion(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,
            seed=seed,
            stop=["<|im_end|>", "<|endoftext|>"],
            echo=False,
        )
        request_seconds = time.perf_counter() - request_started
        usage = response.get("usage") or {}
        choices = response.get("choices") or []
        text = choices[0].get("text", "") if choices else ""
        generated_tokens = int(usage.get("completion_tokens") or 0)
        input_tokens = int(usage.get("prompt_tokens") or 0)
        return {
            "text": text,
            "input_tokens": input_tokens,
            "generated_tokens": generated_tokens,
            "request_seconds": request_seconds,
            "generation_seconds": request_seconds,
            "tokens_per_second": generated_tokens / request_seconds if request_seconds else 0.0,
            "peak_memory_gib": None,
            "finish_reason": choices[0].get("finish_reason") if choices else None,
        }


class LlamaServerBackend:
    def __init__(self, server_url: str, tokenizer, timeout: float):
        self.base_url = server_url.rstrip("/")
        self.tokenizer = tokenizer
        self.timeout = timeout
        get_json(f"{self.base_url}/health", timeout)
        self.hardware = os.environ.get("CUDA_VISIBLE_DEVICES", "llama-server-managed")

    def run(self, prompt: str, max_new_tokens: int, seed: int) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "n_predict": max_new_tokens,
            "temperature": 0.0,
            "seed": seed,
            # Disable prefix/KV reuse so llama-server does the same full prompt
            # prefill work that the Transformers backend performs per row.
            "cache_prompt": False,
            "stop": ["<|im_end|>", "<|endoftext|>"],
        }
        request_started = time.perf_counter()
        response = post_json(f"{self.base_url}/completion", payload, self.timeout)
        request_seconds = time.perf_counter() - request_started
        timings = response.get("timings") or {}
        prompt_seconds = float(timings.get("prompt_ms", 0.0)) / 1000.0
        predicted_seconds = float(timings.get("predicted_ms", 0.0)) / 1000.0
        generation_seconds = prompt_seconds + predicted_seconds
        if generation_seconds <= 0:
            generation_seconds = request_seconds
        generated_tokens = int(
            timings.get("predicted_n")
            or response.get("tokens_predicted")
            or len(self.tokenizer.encode(response.get("content", ""), add_special_tokens=False))
        )
        input_tokens = int(
            timings.get("prompt_n")
            or response.get("tokens_evaluated")
            or len(self.tokenizer.encode(prompt, add_special_tokens=False))
        )
        return {
            "text": response.get("content", ""),
            "input_tokens": input_tokens,
            "generated_tokens": generated_tokens,
            "request_seconds": request_seconds,
            "generation_seconds": generation_seconds,
            "tokens_per_second": generated_tokens / generation_seconds if generation_seconds else 0.0,
            "peak_memory_gib": None,
            "server_stop_type": response.get("stop_type"),
            "server_stopped_eos": response.get("stopped_eos"),
            "server_stopped_limit": response.get("stopped_limit"),
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be at least 1.")
    if args.warmup < 0:
        raise SystemExit("--warmup cannot be negative.")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1.")
    if args.backend in {"transformers", "llama-cpp"} and args.model_path is None:
        raise SystemExit(f"--model-path is required for the {args.backend} backend.")
    if args.n_ctx < 512:
        raise SystemExit("--n-ctx must be at least 512.")
    if args.n_batch < 1:
        raise SystemExit("--n-batch must be at least 1.")
    if args.n_threads < 1:
        raise SystemExit("--n-threads must be at least 1.")
    if args.max_edits < 1:
        raise SystemExit("--max-edits must be at least 1.")
    if not 0 < args.max_change_ratio <= 1:
        raise SystemExit("--max-change-ratio must be greater than 0 and at most 1.")
    if args.max_edit_span_chars < 1:
        raise SystemExit("--max-edit-span-chars must be at least 1.")
    tokenizer_path = args.tokenizer_path
    if tokenizer_path is None and args.backend == "transformers":
        tokenizer_path = args.model_path
    if tokenizer_path is None:
        raise SystemExit(f"--tokenizer-path is required for the {args.backend} backend.")
    if not tokenizer_path.exists():
        raise SystemExit(f"Tokenizer path does not exist: {tokenizer_path}")

    runs_jsonl = args.timing_output.with_suffix(".runs.jsonl")
    existing = [path for path in (args.output, args.timing_output, runs_jsonl) if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "Refusing to overwrite existing result file(s):\n"
            + "\n".join(str(path) for path in existing)
            + "\nPass --overwrite to replace them explicitly."
        )

    rows = load_rows(args.input, args.limit)
    system_prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    tokenizer = load_tokenizer(tokenizer_path)
    prompts = [build_prompt(tokenizer, system_prompt, row) for row in rows]

    startup_started = time.perf_counter()
    if args.backend == "transformers":
        assert args.model_path is not None
        if not args.model_path.exists():
            raise SystemExit(f"Model path does not exist: {args.model_path}")
        backend = TransformersBackend(args.model_path, tokenizer, args.dtype, args.seed)
    elif args.backend == "llama-cpp":
        assert args.model_path is not None
        if not args.model_path.is_file():
            raise SystemExit(f"GGUF model file does not exist: {args.model_path}")
        backend = LlamaCppBackend(
            args.model_path,
            args.n_ctx,
            args.n_batch,
            args.n_gpu_layers,
            args.n_threads,
            args.seed,
        )
    else:
        backend = LlamaServerBackend(args.server_url, tokenizer, args.request_timeout)
    startup_seconds = time.perf_counter() - startup_started

    print(
        f"Backend={args.backend} model={args.model_name} rows={len(rows)} "
        f"hardware={backend.hardware} startup={startup_seconds:.3f}s"
    )
    for current in range(args.warmup):
        result = backend.run(prompts[0], args.max_new_tokens, args.seed)
        print(
            f"Warmup {current + 1}/{args.warmup}: "
            f"{result['generation_seconds']:.3f}s, {result['generated_tokens']} tokens"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.timing_output.parent.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    with (
        args.output.open("w", encoding="utf-8", newline="") as output_handle,
        runs_jsonl.open("w", encoding="utf-8") as timing_handle,
    ):
        writer = csv.DictWriter(
            output_handle,
            fieldnames=["id", "model", "cleaned_text"],
            lineterminator="\n",
        )
        writer.writeheader()
        for request_number, (row, prompt) in enumerate(zip(rows, prompts), start=1):
            backend_result = backend.run(prompt, args.max_new_tokens, args.seed)
            model_response = str(backend_result.pop("text"))
            edit_result = apply_structured_edits(
                row["raw_transcript"],
                model_response,
                max_edits=args.max_edits,
                max_change_ratio=args.max_change_ratio,
                max_edit_span_chars=args.max_edit_span_chars,
            )
            cleaned_text = str(edit_result["output_text"])
            writer.writerow({"id": row["id"], "model": args.model_name, "cleaned_text": cleaned_text})
            output_handle.flush()
            run = {
                "request_number": request_number,
                "row_id": row["id"],
                "language": row["language"],
                "raw_transcript": row["raw_transcript"],
                "reference_clean": row["reference_clean"],
                "output_text": cleaned_text,
                "model_response": edit_result["model_response"],
                "proposed_edits": edit_result["proposed_edits"],
                "applied_edits": edit_result["applied_edits"],
                "rejected_edits": edit_result["rejected_edits"],
                "parse_error": edit_result["parse_error"],
                "safety_fallback": edit_result["safety_fallback"],
                **backend_result,
            }
            runs.append(run)
            timing_handle.write(json.dumps(run, ensure_ascii=False) + "\n")
            timing_handle.flush()
            print(
                f"Request {request_number}/{len(rows)} id={row['id']} "
                f"latency={run['request_seconds']:.3f}s "
                f"generation={run['generation_seconds']:.3f}s "
                f"speed={run['tokens_per_second']:.2f}tok/s "
                f"edits={len(run['applied_edits'])}/{len(run['proposed_edits'])}"
            )

    report = {
        "benchmark": "freedom_ai_labs_transcript_cleanup",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "model_name": args.model_name,
        "model_path": str(args.model_path) if args.model_path else None,
        "tokenizer_path": str(tokenizer_path),
        "server_url": args.server_url if args.backend == "llama-server" else None,
        "hardware": backend.hardware,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dtype": args.dtype if args.backend == "transformers" else None,
        "n_ctx": args.n_ctx if args.backend == "llama-cpp" else None,
        "n_batch": args.n_batch if args.backend == "llama-cpp" else None,
        "n_gpu_layers": args.n_gpu_layers if args.backend == "llama-cpp" else None,
        "n_threads": args.n_threads if args.backend == "llama-cpp" else None,
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "rows": len(rows),
        "max_new_tokens": args.max_new_tokens,
        "max_edits": args.max_edits,
        "max_change_ratio": args.max_change_ratio,
        "max_edit_span_chars": args.max_edit_span_chars,
        "warmup": args.warmup,
        "seed": args.seed,
        "startup_seconds": startup_seconds,
        "summary": summarize(runs),
        "runs": runs,
    }
    args.timing_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("SUMMARY_JSON=" + json.dumps(report["summary"], ensure_ascii=False))
    print(f"Wrote predictions to {args.output}")
    print(f"Wrote timing report to {args.timing_output}")


if __name__ == "__main__":
    main()
