#!/bin/bash
#SBATCH -J freedom-norm-eval
#SBATCH -o freedom-norm-eval-%j.out
#SBATCH -e freedom-norm-eval-%j.err
#SBATCH -t 00:10:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
VENV_PATH="${VENV_PATH:-/scratch/azk2021/freedom-transcript/.venv}"

BENCHMARK="data/freedom_ai_labs_normalized_benchmark.csv"
BF16_PREDICTIONS="outputs/freedom_ai_labs_normalized/qwen3-8b-bfloat16-normalized-215.csv"
Q8_PREDICTIONS="outputs/freedom_ai_labs_normalized/qwen3-8b-q8_0-normalized-215.csv"
BF16_TIMING="reports/freedom_ai_labs_normalized/qwen3-8b-bfloat16-normalized-215.json"
Q8_TIMING="reports/freedom_ai_labs_normalized/qwen3-8b-q8_0-normalized-215.json"
EVALUATION="reports/freedom_ai_labs_normalized/comparison.csv"
LANGUAGE_EVALUATION="reports/freedom_ai_labs_normalized/comparison_by_language.csv"
COMPARISON="reports/freedom_ai_labs_normalized/comparison_with_speed.csv"

cd "$PROJECT_DIR"
source "$VENV_PATH/bin/activate"

for required_file in \
    "$BENCHMARK" \
    "$BF16_PREDICTIONS" \
    "$Q8_PREDICTIONS" \
    "$BF16_TIMING" \
    "$Q8_TIMING"
do
    if [[ ! -f "$required_file" ]]; then
        echo "Required input was not found: $required_file" >&2
        exit 1
    fi
done

python scripts/evaluate_outputs.py \
    --benchmark "$BENCHMARK" \
    --languages mix \
    --outputs "$BF16_PREDICTIONS" \
    --outputs "$Q8_PREDICTIONS" \
    --summary-output "$EVALUATION" \
    --language-summary-output "$LANGUAGE_EVALUATION"

python scripts/build_comparison_report.py \
    --evaluation "$LANGUAGE_EVALUATION" \
    --timing "$BF16_TIMING" \
    --timing "$Q8_TIMING" \
    --output "$COMPARISON" \
    --overwrite

echo "Completed normalized BF16 versus Q8_0 comparison."
cat "$COMPARISON"
