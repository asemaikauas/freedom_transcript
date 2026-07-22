#!/bin/bash
#SBATCH -J freedom-qwen3
#SBATCH -o freedom-qwen3-%j.out
#SBATCH -e freedom-qwen3-%j.err
#SBATCH -t 04:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G

set -euo pipefail

# These defaults match the paths already present for azk2021. Override any
# variable on the sbatch command line if the files move.
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
MODEL_ROOT="${MODEL_ROOT:-/scratch/azk2021/models}"
HF_MODEL_DIR="${HF_MODEL_DIR:-${MODEL_ROOT}/Qwen3-8B}"
GGUF_Q8_MODEL_FILE="${GGUF_Q8_MODEL_FILE:-${MODEL_ROOT}/Qwen3-8B-GGUF/Qwen3-8B-Q8_0.gguf}"
GGUF_Q5_MODEL_FILE="${GGUF_Q5_MODEL_FILE:-${MODEL_ROOT}/Qwen3-8B-GGUF/Qwen3-8B-Q5_K_M.gguf}"
VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/.venv}"
LLAMA_CPP_CUDA_LIB="${LLAMA_CPP_CUDA_LIB:-/scratch/azk2021/lib/ollama/cuda_v13}"

cd "$PROJECT_DIR"
mkdir -p outputs/freedom_ai_labs reports/freedom_ai_labs
source "$VENV_PATH/bin/activate"

# Run all three checkpoints sequentially inside one allocation so they use the
# exact same GPU model and scheduler environment.
python scripts/run_qwen3_dataset.py \
    --backend transformers \
    --model-path "$HF_MODEL_DIR" \
    --model-name Qwen3-8B-bfloat16 \
    --input data/freedom_ai_labs_benchmark.csv \
    --output outputs/freedom_ai_labs/qwen3-8b-bfloat16.csv \
    --timing-output reports/freedom_ai_labs/qwen3-8b-bfloat16-timing.json \
    --dtype bfloat16 \
    --warmup 2 \
    --overwrite

env LD_LIBRARY_PATH="${LLAMA_CPP_CUDA_LIB}:${LD_LIBRARY_PATH:-}" \
python scripts/run_qwen3_dataset.py \
    --backend llama-cpp \
    --model-path "$GGUF_Q8_MODEL_FILE" \
    --tokenizer-path "$HF_MODEL_DIR" \
    --model-name Qwen3-8B-GGUF-Q8_0 \
    --input data/freedom_ai_labs_benchmark.csv \
    --output outputs/freedom_ai_labs/qwen3-8b-q8_0.csv \
    --timing-output reports/freedom_ai_labs/qwen3-8b-q8_0-timing.json \
    --n-gpu-layers -1 \
    --n-threads 8 \
    --warmup 2 \
    --overwrite

env LD_LIBRARY_PATH="${LLAMA_CPP_CUDA_LIB}:${LD_LIBRARY_PATH:-}" \
python scripts/run_qwen3_dataset.py \
    --backend llama-cpp \
    --model-path "$GGUF_Q5_MODEL_FILE" \
    --tokenizer-path "$HF_MODEL_DIR" \
    --model-name Qwen3-8B-GGUF-Q5_K_M \
    --input data/freedom_ai_labs_benchmark.csv \
    --output outputs/freedom_ai_labs/qwen3-8b-q5_k_m.csv \
    --timing-output reports/freedom_ai_labs/qwen3-8b-q5_k_m-timing.json \
    --n-gpu-layers -1 \
    --n-threads 8 \
    --warmup 2 \
    --overwrite

python scripts/evaluate_outputs.py \
    --benchmark data/freedom_ai_labs_benchmark.csv \
    --languages mix \
    --outputs outputs/freedom_ai_labs/qwen3-8b-bfloat16.csv \
    --outputs outputs/freedom_ai_labs/qwen3-8b-q8_0.csv \
    --outputs outputs/freedom_ai_labs/qwen3-8b-q5_k_m.csv \
    --summary-output reports/freedom_ai_labs/comparison.csv \
    --language-summary-output reports/freedom_ai_labs/comparison_by_language.csv

python scripts/build_comparison_report.py \
    --evaluation reports/freedom_ai_labs/comparison_by_language.csv \
    --timing reports/freedom_ai_labs/qwen3-8b-bfloat16-timing.json \
    --timing reports/freedom_ai_labs/qwen3-8b-q8_0-timing.json \
    --timing reports/freedom_ai_labs/qwen3-8b-q5_k_m-timing.json \
    --output reports/freedom_ai_labs/comparison_with_speed.csv \
    --overwrite

echo "Finished full-precision, Q8_0, Q5_K_M, and supplied-ASR baseline comparison."
