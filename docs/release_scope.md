# Release Scope

This repository is designed to address reproducibility concerns for nursing AI evaluation.

The release includes:

- A public benchmark link: `Agnania/NursData-MCQ`.
- A local sample file for quick code checks.
- MCQ evaluation code for OpenAI-compatible endpoints.
- Answer extraction and exact-match accuracy logic.
- Model-serving examples for vLLM.
- A de-identified RAG API example that preserves the OpenAI-compatible interface, query rewrite step, source-level retrieval hook, passage-level retrieval hook, and evidence prompt flow.
- Configuration templates for evaluation and model serving.
- Dataset validation utilities.
- Requirements and Apache-2.0 license.

The release is intentionally separated from the original working directory to avoid publishing temporary experiment files, local server scripts with hard-coded paths, or credentials.

Model weights are released separately at `Agnania/EviNurse-32B`.
