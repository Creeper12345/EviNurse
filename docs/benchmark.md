# NursData-MCQ Benchmark

NursData-MCQ is a Chinese nursing multiple-choice benchmark used for automated evaluation in the EviNurse study.

The full benchmark is hosted on Hugging Face:

- https://huggingface.co/datasets/Agnania/NursData-MCQ

This GitHub repository keeps only a small sample file at:

- `data/samples/mcq_sample.json`

The sample exists to make the evaluator runnable without downloading the complete benchmark. Full reporting should use the Hugging Face dataset.

## Schema

Each record in `data/samples/mcq_sample.json` and the full Hugging Face benchmark is a JSON object. Required fields:

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Unique question id |
| `question` | string | Question stem |
| `options` | object | Option map, usually A-E. Keys are option letters and values are option text. |
| `answer` | string | Standard answer letter |

Optional fields used by context-aware evaluation:

| Field | Type | Description |
| --- | --- | --- |
| `context` | array | Retrieved evidence snippets. Each item may include `doc_name`, `source`, `content`, or `text`. |
| `metadata` | object | Additional source or benchmark metadata if available. |

Minimal example:

```json
{
  "id": "2011-专业实务-1",
  "question": "心脏自身的血液供应主要来自于",
  "options": {
    "A": "主动脉",
    "B": "锁骨下动脉",
    "C": "冠状动脉",
    "D": "肺动脉",
    "E": "肺静脉"
  },
  "answer": "C"
}
```

Context-aware example:

```json
{
  "id": "example-with-context",
  "question": "问题文本",
  "options": {"A": "选项A", "B": "选项B", "C": "选项C", "D": "选项D"},
  "answer": "A",
  "context": [
    {
      "doc_name": "指南或证据总结名称",
      "content": "与问题相关的证据片段"
    }
  ]
}
```

## Prediction Output

Evaluation scripts write a JSON list. Each output record keeps the original fields and appends:

| Field | Type | Description |
| --- | --- | --- |
| `model_output` | string | Raw model response |
| `predicted_answer` | string or null | Extracted answer option |
| `is_correct` | boolean | Whether `predicted_answer` equals `answer` |
| `error` | string or null | Request or parsing error, if any |
| `response_time` | number | Present in `evaluate_mcq_with_context.py`; request duration in seconds |

## Metric

The default metric is exact-match accuracy after extracting one final option letter from model output.
