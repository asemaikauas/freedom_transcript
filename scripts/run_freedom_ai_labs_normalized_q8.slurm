#!/bin/bash
#SBATCH -J freedom-norm-q8
#SBATCH -o freedom-norm-q8-%j.out
#SBATCH -e freedom-norm-q8-%j.err
#SBATCH -t 01:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-/scratch/azk2021/models}"
HF_MODEL_DIR="${HF_MODEL_DIR:-${MODEL_ROOT}/Qwen3-8B}"
GGUF_MODEL="${GGUF_MODEL:-${MODEL_ROOT}/Qwen3-8B-GGUF/Qwen3-8B-Q8_0.gguf}"
VENV_PATH="${VENV_PATH:-/scratch/azk2021/freedom-transcript/.venv}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-/scratch/azk2021/llama.cpp-v100/build-v100/bin/llama-server}"
CUDA12_LIB_DIR="${CUDA12_LIB_DIR:-/scratch/azk2021/lib/ollama/cuda_v12}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8080}"

BENCHMARK="data/freedom_ai_labs_normalized_benchmark.csv"
PREDICTIONS="outputs/freedom_ai_labs_normalized/qwen3-8b-q8_0-normalized-215.csv"
TIMING="reports/freedom_ai_labs_normalized/qwen3-8b-q8_0-normalized-215.json"
EVALUATION="reports/freedom_ai_labs_normalized/qwen3-8b-q8_0-normalized-215-eval.csv"
LANGUAGE_EVALUATION="reports/freedom_ai_labs_normalized/qwen3-8b-q8_0-normalized-215-eval-by-language.csv"
SERVER_LOG="reports/freedom_ai_labs_normalized/llama-server-q8-v100.log"

cd "$PROJECT_DIR"
source "$VENV_PATH/bin/activate"

if [[ ! -x "$LLAMA_SERVER_BIN" ]]; then
    echo "llama-server is not executable: $LLAMA_SERVER_BIN" >&2
    exit 1
fi
if [[ ! -f "$GGUF_MODEL" ]]; then
    echo "GGUF model was not found: $GGUF_MODEL" >&2
    exit 1
fi
if [[ ! -d "$CUDA12_LIB_DIR" ]]; then
    echo "CUDA 12 runtime directory was not found: $CUDA12_LIB_DIR" >&2
    exit 1
fi

export LD_LIBRARY_PATH="${CUDA12_LIB_DIR}:${LD_LIBRARY_PATH:-}"
if ldd "$LLAMA_SERVER_BIN" | grep -q "not found"; then
    echo "llama-server has unresolved shared libraries:" >&2
    ldd "$LLAMA_SERVER_BIN" | grep "not found" >&2
    exit 1
fi

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
echo "Model: $GGUF_MODEL"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID"
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

"$LLAMA_SERVER_BIN" \
    --model "$GGUF_MODEL" \
    --ctx-size 4096 \
    --n-gpu-layers 99 \
    --host 127.0.0.1 \
    --port 8080 \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

SERVER_READY=0
for _attempt in $(seq 1 60); do
    if curl -fsS "${SERVER_URL}/health" >/dev/null; then
        SERVER_READY=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "llama-server exited during startup:" >&2
        tail -80 "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 2
done
if [[ "$SERVER_READY" != "1" ]]; then
    echo "llama-server did not become healthy within 120 seconds:" >&2
    tail -80 "$SERVER_LOG" >&2
    exit 1
fi

python -u scripts/run_qwen3_dataset.py \
    --backend llama-server \
    --server-url "$SERVER_URL" \
    --tokenizer-path "$HF_MODEL_DIR" \
    --model-name Qwen3-8B-GGUF-Q8_0-normalized-edits-v2 \
    --input "$BENCHMARK" \
    --output "$PREDICTIONS" \
    --timing-output "$TIMING" \
    --warmup 2 \
    --overwrite

python scripts/evaluate_outputs.py \
    --benchmark "$BENCHMARK" \
    --languages mix \
    --outputs "$PREDICTIONS" \
    --summary-output "$EVALUATION" \
    --language-summary-output "$LANGUAGE_EVALUATION"

echo "Completed normalized 215-row Qwen3-8B Q8_0 benchmark."
cat "$LANGUAGE_EVALUATION"
