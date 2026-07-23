#!/bin/bash
#SBATCH -J freedom-norm-bf16
#SBATCH -o freedom-norm-bf16-%j.out
#SBATCH -e freedom-norm-bf16-%j.err
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

BENCHMARK="data/freedom_ai_labs_normalized_benchmark.csv"
PREDICTIONS="outputs/freedom_ai_labs_normalized/qwen3-8b-bfloat16-normalized-215.csv"
TIMING="reports/freedom_ai_labs_normalized/qwen3-8b-bfloat16-normalized-215.json"
EVALUATION="reports/freedom_ai_labs_normalized/qwen3-8b-bfloat16-normalized-215-eval.csv"
LANGUAGE_EVALUATION="reports/freedom_ai_labs_normalized/qwen3-8b-bfloat16-normalized-215-eval-by-language.csv"

cd "$PROJECT_DIR"
source "$VENV_PATH/bin/activate"

ROW_COUNT="$(
    python -c \
        'import csv, sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1], encoding="utf-8"))))' \
        "$BENCHMARK"
)"
if [[ "$ROW_COUNT" != "215" ]]; then
    echo "Expected 215 normalized benchmark rows, found $ROW_COUNT" >&2
    exit 1
fi

mkdir -p outputs/freedom_ai_labs_normalized reports/freedom_ai_labs_normalized

echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Project: $PROJECT_DIR"
echo "Benchmark: $BENCHMARK ($ROW_COUNT rows)"
echo "Model: $HF_MODEL_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python -u scripts/run_qwen3_dataset.py \
    --backend transformers \
    --model-path "$HF_MODEL_DIR" \
    --model-name Qwen3-8B-bfloat16-normalized-edits-v2 \
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

echo "Completed normalized 215-row Qwen3-8B BF16 benchmark."
cat "$LANGUAGE_EVALUATION"
