# LLM Transcript Correction Benchmark

LLM output files should use this CSV schema:

```text
id,model,cleaned_text
```

Then evaluate:

```bash
python3 scripts/evaluate_outputs.py --outputs outputs/my_model.csv
```

Generation and evaluation default to Kazakh (`kk`), Russian (`ru`), and
Kazakh-Russian mixed (`mix`). English rows are skipped. The evaluator writes
both a detailed language-by-ASR report and a separate one-row-per-language
report whose filename ends in `_by_language.csv`.

To select languages explicitly:

```bash
python3 scripts/evaluate_outputs.py \
  --outputs outputs/my_model.csv \
  --languages kk ru mix
```

You can also run the evaluator before any LLM outputs exist to get baseline ASR metrics:

```bash
python3 scripts/evaluate_outputs.py
```

Install the evaluation dependency first:

```bash
python3 -m pip install -r requirements.txt
```

The evaluator reports normalized WER, CER, exact-match rate, chrF, and chrF++. Normalization lowercases text and ignores punctuation so a model is not punished just for adding sentence punctuation.

## Freedom AI Labs ground-truth dataset

The source workbook has two columns:

- `transcript`: human ground truth (`reference_clean`)
- `transcript_pred`: supplied ASR prediction (`raw_transcript` / baseline)

Convert the workbook to the canonical benchmark schema without modifying the
source file:

```bash
python3 scripts/prepare_freedom_ai_labs_dataset.py \
  --input "/path/to/GT, freedom ai labs.xlsx" \
  --output data/freedom_ai_labs_benchmark.csv
```

The workbook does not contain row-level language labels. The converter therefore
marks every row as `mix` (Kazakh-Russian mixed) unless `--language` is supplied.
Do not infer separate Kazakh and Russian scores from this dataset without adding
reviewed language labels first.

The generated benchmark contains 215 rows. To evaluate its supplied ASR baseline:

```bash
python3 scripts/evaluate_outputs.py \
  --benchmark data/freedom_ai_labs_benchmark.csv \
  --languages mix \
  --summary-output reports/freedom_ai_labs/baseline.csv
```

### Qwen3-8B versus Qwen3-8B-GGUF

`scripts/run_qwen3_dataset.py` supports Transformers and GGUF backends. Both use the tokenizer from
the full Qwen3-8B directory to render exactly the same chat prompt with thinking
disabled. Requests are processed serially with batch size 1 for comparable
latency and tokens-per-second measurements.

The model does not write the final transcript directly. It proposes exact local
replacements as JSON, and the runner applies only edits that pass validation.
The evaluator-compatible output remains:

```text
id,model,cleaned_text
```

The runner treats the model-supplied edit type as untrusted. It verifies filler,
punctuation, capitalization, and spacing edits mechanically; protects
capitalized names; and rejects normal-word deletion, multiword grammar changes,
word shortening, ordinary repetition deletion, and word changes larger than
one character. It also rejects malformed JSON, non-local or overlapping
replacements, ambiguous repeated substrings, excessive changes, modified
numbers, unexpected Kazakh/Russian language shifts, and repetition loops. A
rejected edit leaves that source span unchanged. Per-row timing JSONL records
the raw model response, proposed edits, applied edits, rejected edits, parse
errors, and safety fallback reason.

The safety limits can be adjusted when running controlled experiments:

```text
--max-edits 8
--max-change-ratio 0.15
--max-edit-span-chars 80
```

Full-precision Transformers run:

```bash
python3 scripts/run_qwen3_dataset.py \
  --backend transformers \
  --model-path /scratch/<NetID>/models/Qwen3-8B \
  --model-name Qwen3-8B-bfloat16 \
  --output outputs/freedom_ai_labs/qwen3-8b-bfloat16.csv \
  --timing-output reports/freedom_ai_labs/qwen3-8b-bfloat16-timing.json
```

The existing Jubail environment includes GPU-enabled `llama-cpp-python`. Run a
GGUF checkpoint directly with:

```bash
env LD_LIBRARY_PATH=/scratch/<NetID>/lib/ollama/cuda_v13:${LD_LIBRARY_PATH:-} \
python scripts/run_qwen3_dataset.py \
  --backend llama-cpp \
  --model-path /scratch/<NetID>/models/Qwen3-8B-GGUF/Qwen3-8B-Q8_0.gguf \
  --tokenizer-path /scratch/<NetID>/models/Qwen3-8B \
  --model-name Qwen3-8B-GGUF-Q8_0 \
  --output outputs/freedom_ai_labs/qwen3-8b-q8_0.csv \
  --timing-output reports/freedom_ai_labs/qwen3-8b-q8_0-timing.json \
  --n-gpu-layers -1
```

The provided Slurm job runs bfloat16, Q8_0, and Q5_K_M sequentially on the same
Jubail GPU and then evaluates all three:

```bash
sbatch scripts/run_freedom_ai_labs_qwen3.slurm
```

The timing JSON contains mean, median, and p95 request/generation latency,
aggregate output tokens per second, backend initialization time, hardware
metadata, and every per-request result. Model loading is excluded from request
latency. The Slurm job also
writes `reports/freedom_ai_labs/comparison_with_speed.csv`, which joins accuracy,
relative improvement over the supplied ASR baseline, latency, throughput, and
hardware into one table.

### Normalized 215-row BF16 versus Q8_0 comparison

`data/freedom_ai_labs_normalized_benchmark.csv` contains the lowercase,
punctuation-normalized version of the same 215 source/reference pairs. It uses
separate IDs and output directories so it cannot overwrite the original
benchmark.

Submit both hardware-specific runs and a dependent comparison job from the
repository root:

```bash
bash scripts/submit_freedom_ai_labs_normalized.sh
```

The BF16 job requests an A100. The Q8_0 job requests a V100 and starts the
existing V100 `llama-server` with its CUDA 12 runtime. When both model jobs
succeed, the comparison job writes:

```text
reports/freedom_ai_labs_normalized/comparison_with_speed.csv
```

## Models

The following models are benchmarked:

- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Gemma 3 4B IT](https://huggingface.co/google/gemma-3-4b-it)
- [ISSAI KazLLM-1.0-8B](https://huggingface.co/issai/LLama-3.1-KazLLM-1.0-8B)
- [ISSAI Sherkala-Chat-8B](https://huggingface.co/inceptionai/Llama-3.1-Sherkala-8B-Chat)

## Evaluation Metrics

Evaluation compares each prediction against the `reference_clean` column.

Before scoring, both prediction and reference are normalized:

- text is lowercased
- Unicode letters and numbers are kept
- whitespace, punctuation, and symbols are converted to spaces
- repeated spaces are collapsed

The evaluator reports:

- `WER`: word error rate, calculated as word edit distance divided by the number of reference words. Lower is better; `0.0` is perfect.
- `CER`: character error rate, calculated as character edit distance divided by the number of reference characters after removing spaces. Lower is better; `0.0` is perfect.
- `corpus_wer`: total word edits divided by total reference words. This is the standard corpus-level (micro-averaged) WER.
- `corpus_cer`: total character edits divided by total reference characters. This is the corpus-level (micro-averaged) CER.
- `exact_norm`: normalized exact-match rate. `1.0` means every normalized prediction exactly matched the normalized reference.
- `chrF`: character n-gram F-score from SacreBLEU, reported on the standard `0-100` scale. Higher is better; `100.0` is perfect.
- `chrF++`: chrF with word n-grams included, reported on the standard `0-100` scale. Higher is better; `100.0` is perfect.

The report is grouped by:

```text
model,language,asr_model
```

Running without an LLM output file evaluates the original ASR predictions as `baseline_asr`:

```text
raw_transcript vs reference_clean
```

Running with an output file evaluates model corrections:

```text
cleaned_text vs reference_clean
```
