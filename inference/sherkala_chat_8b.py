#!/usr/bin/env python3
"""Run ISSAI Sherkala-Chat-8B over the benchmark CSV to produce cleaned transcripts."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_PATH = Path("models/Llama-3.1-Sherkala-8B-Chat")
DEFAULT_INPUT = Path("data/llm_correction_benchmark.csv")
DEFAULT_OUTPUT = Path("outputs/sherkala-chat-8b.csv")
DEFAULT_PROMPT = Path("prompts/transcript_cleanup.txt")
MODEL_NAME = "Sherkala-Chat-8B"

LANGUAGE_NAMES = {
    "en": "English",
    "kk": "Kazakh",
    "ru": "Russian",
    "mix": "Kazakh-Russian mixed (code-switched)",
}
DEFAULT_LANGUAGES = ("kk", "ru", "mix")

PREAMBLE_RE = re.compile(
    r"^(?:here(?:'s| is)|this is|the (?:cleaned|corrected)(?: up)? transcript(?: is)?)\b[^:\n]{0,60}:\s*",
    re.IGNORECASE,
)


def strip_preamble(text: str) -> str:
    text = PREAMBLE_RE.sub("", text).strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] in "\"'":
        text = text[1:-1].strip()
    return text

# Sherkala's tokenizer ships without a chat_template; this is the Llama-3.1 template
# the model card instructs users to set manually.
CHAT_TEMPLATE = (
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
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=LANGUAGE_NAMES,
        default=list(DEFAULT_LANGUAGES),
        help="Languages to process (default: kk ru mix; English is skipped).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (for smoke tests).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def load_rows(path: Path, languages: list[str], limit: int | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["language"] in languages]
    return rows[:limit] if limit else rows


def build_prompt(tokenizer, system_prompt: str, raw_transcript: str, language: str) -> str:
    language_name = LANGUAGE_NAMES.get(language, language)
    user_content = f"Target language: {language_name}\n\n{raw_transcript}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def batched(items: list[dict[str, str]], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main() -> None:
    args = parse_args()
    system_prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    rows = load_rows(args.input, args.languages, args.limit)
    print(f"Selected {len(rows)} rows for languages: {', '.join(args.languages)}")

    print(f"Loading tokenizer/model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # generation_config.json ships a stale eos_token_id (2) that matches nothing in this
    # vocab, so generation never stops on its own; supply the real turn-end tokens instead.
    eos_token_ids = sorted({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")})

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "model", "cleaned_text"])
        writer.writeheader()

        for batch in batched(rows, args.batch_size):
            prompts = [
                build_prompt(tokenizer, system_prompt, row["raw_transcript"], row["language"]) for row in batch
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_token_ids,
                )

            for row, output_ids in zip(batch, generated):
                completion_ids = output_ids[prompt_len:]
                cleaned_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
                cleaned_text = strip_preamble(cleaned_text)
                writer.writerow({"id": row["id"], "model": MODEL_NAME, "cleaned_text": cleaned_text})

            processed += len(batch)
            handle.flush()
            print(f"Processed {processed}/{len(rows)}")

    print(f"Wrote outputs to {args.output}")


if __name__ == "__main__":
    main()
