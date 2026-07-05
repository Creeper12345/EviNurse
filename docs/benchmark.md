# NursData-MCQ Benchmark

NursData-MCQ is a Chinese nursing multiple-choice benchmark used for automated evaluation in the EviNurse study.

The full benchmark is hosted on Hugging Face:

- https://huggingface.co/datasets/Agnania/NursData-MCQ

This GitHub repository keeps only a small sample file at:

- `data/samples/mcq_sample.json`

The sample exists to make the evaluator runnable without downloading the complete benchmark. Full reporting should use the Hugging Face dataset.

## Schema

Each record has:

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Unique question id |
| `question` | string | Question stem |
| `options` | object | Option map, usually A-E |
| `answer` | string | Standard answer letter |

## Metric

The default metric is exact-match accuracy after extracting one final option letter from model output.

