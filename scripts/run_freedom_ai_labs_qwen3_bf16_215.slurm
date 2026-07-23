#!/bin/bash
#SBATCH -J freedom-qwen3-bf16-215
#SBATCH -o freedom-qwen3-bf16-215-%j.out
#SBATCH -e freedom-qwen3-bf16-215-%j.err
#SBATCH -t 01:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-/scratch/azk2021/models}"
HF_MODEL_DIR="${HF_MODEL_DIR:-${MODEL_ROOT}/Qwen3-8B}"
VENV_PATH="${VENV_PATH:-/scratch/azk2021/freedom-transcript/.venv}"

cd "$PROJECT_DIR"
source "$VENV_PATH/bin/activate"

BENCHMARK="data/freedom_ai_labs_benchmark.csv"
PREDICTIONS="outputs/freedom_ai_labs/qwen3-8b-bfloat16-215.csv"
TIMING="reports/freedom_ai_labs/qwen3-8b-bfloat16-215-timing.json"
EVALUATION="reports/freedom_ai_labs/comparison_bf16_215.csv"
LANGUAGE_EVALUATION="reports/freedom_ai_labs/comparison_bf16_215_by_language.csv"
COMPARISON="reports/freedom_ai_labs/comparison_bf16_215_with_speed.csv"

ROW_COUNT="$(
    python -c \
        'import csv; print(sum(1 for _ in csv.DictReader(open("data/freedom_ai_labs_benchmark.csv", encoding="utf-8"))))'
)"
if [[ "$ROW_COUNT" != "215" ]]; then
    echo "Expected 215 benchmark rows, found $ROW_COUNT" >&2
    exit 1
fi

mkdir -p outputs/freedom_ai_labs reports/freedom_ai_labs

echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Project: $PROJECT_DIR"
echo "Benchmark rows: $ROW_COUNT"
echo "Model: $HF_MODEL_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python -u scripts/run_qwen3_dataset.py \
    --backend transformers \
    --model-path "$HF_MODEL_DIR" \
    --model-name Qwen3-8B-bfloat16 \
    --input "$BENCHMARK" \
    --output "$PREDICTIONS" \
    --timing-output "$TIMING" \
    --dtype bfloat16 \
    --warmup 2 \
    --overwrite

python scripts/evaluate_outputs.py \
    --benchmark "$BENCHMARK" \
    --languages mix \
    --outputs "$PREDICTIONS" \
    --summary-output "$EVALUATION" \
    --language-summary-output "$LANGUAGE_EVALUATION"

python scripts/build_comparison_report.py \
    --evaluation "$LANGUAGE_EVALUATION" \
    --timing "$TIMING" \
    --output "$COMPARISON" \
    --overwrite

echo "Completed the 215-row Qwen3-8B BF16 benchmark."
echo "Final comparison: $COMPARISON"
cat "$COMPARISON"
