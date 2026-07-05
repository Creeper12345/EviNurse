# Reproducibility Notes

## Automated MCQ Evaluation

1. Install dependencies.
2. Download `Agnania/NursData-MCQ` from Hugging Face.
3. Start an OpenAI-compatible model endpoint.
4. Run `scripts/evaluate_mcq.py`.
5. Report exact-match accuracy and include the prediction JSON.

## Model Availability

The EviNurse model link is not included yet. Once the model is available, update:

- `README.md`
- `configs/model_server.example.yaml`

## Recommended Reporting

For each evaluated model, report:

- Model name and version.
- Endpoint framework, such as vLLM or SGLang.
- Prompt template.
- Decoding parameters.
- Correct count, total count, and accuracy.
- Prediction file or checksum.

