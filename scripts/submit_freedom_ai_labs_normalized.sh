#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_DIR"

BF16_JOB_ID="$(sbatch --parsable scripts/run_freedom_ai_labs_normalized_bf16.slurm)"
Q8_JOB_ID="$(sbatch --parsable scripts/run_freedom_ai_labs_normalized_q8.slurm)"
EVAL_JOB_ID="$(
    sbatch \
        --parsable \
        --dependency="afterok:${BF16_JOB_ID}:${Q8_JOB_ID}" \
        scripts/evaluate_freedom_ai_labs_normalized.slurm
)"

echo "BF16 A100 job: $BF16_JOB_ID"
echo "Q8_0 V100 job: $Q8_JOB_ID"
echo "Dependent comparison job: $EVAL_JOB_ID"
