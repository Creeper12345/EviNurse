# Reproducibility Notes

## Automated MCQ Evaluation

1. Install dependencies with `pip install -r requirements.txt`.
2. Download `Agnania/NursData-MCQ` from Hugging Face.
3. Install vLLM serving dependencies with `pip install -r requirements-vllm.txt` if launching the released model locally.
4. Start an OpenAI-compatible model endpoint.
5. Run `scripts/evaluate_mcq.py`.
6. Report exact-match accuracy and include the prediction JSON.

## Model Availability

The EviNurse model is released separately at:

https://huggingface.co/Agnania/EviNurse-32B

## Recommended Reporting

For each evaluated model, report:

- Model name and version.
- Endpoint framework, such as vLLM or SGLang.
- Prompt template.
- Decoding parameters.
- Correct count, total count, and accuracy.
- Prediction file or checksum.
