#!/usr/bin/env python3
"""Benchmark one transcript-cleanup request on one local model."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


DEFAULT_INPUT = Path("data/llm_correction_benchmark.csv")
DEFAULT_PROMPT = Path("prompts/transcript_cleanup.txt")
DEFAULT_MODEL_ROOT = Path("/scratch/azk2021/models")

MODEL_SPECS = {
    "kazllm": {
        "name": "KazLLM-1.0-8B",
        "directory": "LLama-3.1-KazLLM-1.0-8B",
        "kind": "causal",
    },
    "qwen2.5": {
        "name": "Qwen2.5-7B-Instruct",
        "directory": "Qwen2.5-7B-Instruct",
        "kind": "causal",
    },
    "gemma3": {
        "name": "Gemma-3-4B-IT",
        "directory": "gemma-3-4b-it",
        "kind": "image_text",
    },
    "sherkala": {
        "name": "Sherkala-Chat-8B",
        "directory": "Llama-3.1-Sherkala-8B-Chat",
        "kind": "causal",
    },
    "qwen3": {
        "name": "Qwen3-8B",
        "directory": "Qwen3-8B",
        "kind": "causal",
    },
}

LANGUAGE_NAMES = {
    "en": "English",
    "kk": "Kazakh",
    "ru": "Russian",
    "mix": "Kazakh-Russian mixed (code-switched)",
}

# Sherkala's tokenizer does not provide a chat template. This is the template
# used by the repository's full inference script and the model card.
SHERKALA_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role']+'<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %} {% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_SPECS)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help=f"Directory containing all model folders (default: {DEFAULT_MODEL_ROOT}).",
    )
    parser.add_argument("--model-path", type=Path, help="Override the complete path for this model.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="Zero-based benchmark row used as the single request.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_row(path: Path, row_index: int) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not 0 <= row_index < len(rows):
        raise SystemExit(f"--row-index must be between 0 and {len(rows) - 1}")
    return rows[row_index]


def build_prompt(tokenizer, model_key: str, system_prompt: str, row: dict[str, str]) -> str:
    language = LANGUAGE_NAMES.get(row["language"], row["language"])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Target language: {language}\n\n{row['raw_transcript']}"},
    ]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if model_key == "qwen3":
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def load_tokenizer_and_model(model_key: str, model_path: Path, dtype: torch.dtype):
    spec = MODEL_SPECS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if model_key == "sherkala" and tokenizer.chat_template is None:
        tokenizer.chat_template = SHERKALA_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    common = {
        "device_map": "cuda",
        "local_files_only": True,
    }
    if spec["kind"] == "image_text":
        model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=dtype, **common)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, **common)
    model.eval()
    return tokenizer, model


def generation_kwargs(model_key: str, tokenizer, max_new_tokens: int) -> dict:
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if model_key == "sherkala":
        possible_ids = (
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        )
        eos_ids = sorted({token_id for token_id in possible_ids if isinstance(token_id, int) and token_id >= 0})
        kwargs["eos_token_id"] = eos_ids
    return kwargs


def generate_once(model, inputs, kwargs: dict) -> tuple[float, int, float, torch.Tensor]:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    prompt_tokens = inputs["input_ids"].shape[-1]
    generated_tokens = output.shape[-1] - prompt_tokens
    peak_memory_gib = torch.cuda.max_memory_allocated() / (1024**3)
    return elapsed, generated_tokens, peak_memory_gib, output[0, prompt_tokens:]


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be at least 1.")
    if args.warmup < 0:
        raise SystemExit("--warmup cannot be negative.")
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be at least 1.")
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU allocation is required.")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise SystemExit("This GPU does not support bfloat16; retry with --dtype float16.")

    spec = MODEL_SPECS[args.model]
    model_path = args.model_path or args.model_root / spec["directory"]
    if not model_path.exists():
        raise SystemExit(f"Model path does not exist: {model_path}")

    row = load_row(args.input, args.row_index)
    system_prompt = args.prompt_file.read_text(encoding="utf-8").strip()

    load_started = time.perf_counter()
    tokenizer, model = load_tokenizer_and_model(args.model, model_path, dtype)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    prompt = build_prompt(tokenizer, args.model, system_prompt, row)
    tokenize_kwargs = {
        "return_tensors": "pt",
        "add_special_tokens": args.model != "gemma3",
    }
    inputs = tokenizer(prompt, **tokenize_kwargs).to(model.device)
    gen_kwargs = generation_kwargs(args.model, tokenizer, args.max_new_tokens)

    print(
        f"Model={spec['name']} row={args.row_index} id={row['id']} "
        f"language={row['language']} input_tokens={inputs['input_ids'].shape[-1]}"
    )
    print(f"Model load time: {load_seconds:.3f} s")

    for current in range(args.warmup):
        elapsed, tokens, _, _ = generate_once(model, inputs, gen_kwargs)
        print(f"Warmup {current + 1}/{args.warmup}: {elapsed:.3f} s, {tokens} tokens")

    runs: list[dict[str, float | int]] = []
    last_completion = None
    for current in range(args.repetitions):
        elapsed, tokens, peak_memory_gib, last_completion = generate_once(model, inputs, gen_kwargs)
        tokens_per_second = tokens / elapsed if elapsed else 0.0
        runs.append(
            {
                "latency_seconds": elapsed,
                "generated_tokens": tokens,
                "tokens_per_second": tokens_per_second,
                "peak_memory_gib": peak_memory_gib,
            }
        )
        print(
            f"Run {current + 1}/{args.repetitions}: {elapsed:.3f} s, "
            f"{tokens} tokens, {tokens_per_second:.2f} tokens/s, "
            f"peak={peak_memory_gib:.2f} GiB"
        )

    result = {
        "model_key": args.model,
        "model_name": spec["name"],
        "model_path": str(model_path),
        "gpu": torch.cuda.get_device_name(),
        "dtype": args.dtype,
        "row_index": args.row_index,
        "row_id": row["id"],
        "language": row["language"],
        "input_tokens": inputs["input_ids"].shape[-1],
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "model_load_seconds": load_seconds,
        "median_latency_seconds": statistics.median(run["latency_seconds"] for run in runs),
        "mean_latency_seconds": statistics.mean(run["latency_seconds"] for run in runs),
        "mean_tokens_per_second": statistics.mean(run["tokens_per_second"] for run in runs),
        "max_peak_memory_gib": max(run["peak_memory_gib"] for run in runs),
        "runs": runs,
    }

    if last_completion is not None:
        result["last_output"] = tokenizer.decode(last_completion, skip_special_tokens=True).strip()

    print("RESULT_JSON=" + json.dumps(result, ensure_ascii=False))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
