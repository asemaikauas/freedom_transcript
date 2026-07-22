# LLM Correction Benchmark Summary

- Source rows: 2229
- Correction tasks: 2229
- Exact unique references: 686
- Languages: en, kk, mix, ru
- ASR systems: fastconformer_kk_ru, fastconformer_v1.2-cpt-1850000, whisper-v2.2-ct2-int8

## Rows By Language

| language | rows | correction_tasks | exact_unique_references | asr_changed_rate |
| --- | ---: | ---: | ---: | ---: |
| en | 555 | 555 | 231 | 0.888 |
| kk | 588 | 588 | 158 | 0.522 |
| mix | 519 | 519 | 145 | 0.753 |
| ru | 567 | 567 | 152 | 0.780 |

## Rows By ASR System

| asr_model | rows | asr_changed_rate |
| --- | ---: | ---: |
| fastconformer_kk_ru | 743 | 0.785 |
| fastconformer_v1.2-cpt-1850000 | 743 | 0.766 |
| whisper-v2.2-ct2-int8 | 743 | 0.647 |

## Output Schema For LLM Runs

Create a CSV with columns:

```text
id,model,cleaned_text
```

`id` must match the benchmark `id` column.
