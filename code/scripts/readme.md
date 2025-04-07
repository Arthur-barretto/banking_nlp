# Local LLM Processing and Evaluation Pipeline

This folder contains scripts for running **structured topic modeling and sentiment analysis** using local LLMs. The current workflow supports both **Llama3** and **Qwen2** models. All processing is done locally to avoid API costs and enable experimentation with open-source models.

## Overview

The workflow consists of three main stages:

1. **Topic and Sentiment Extraction**  
2. **Model Evaluation via LLM-as-a-Judge**  
3. **Model-Level Assessment**

These stages are implemented identically for both Llama3 and Qwen2, with only the script names and model references varying.

---

## 1. Topic and Sentiment Extraction

Each script processes a batch of transcripts and performs two tasks:

- **Task 1**: Extract up to 10 key topics in bullet-point format.
- **Task 2**: Assign a sentiment classification (`positivo` or `negativo`).

**Scripts:**
- `LLM_llama3_unsupervised.py`: Runs extraction using the Llama3 model.
- `LLM_qwen_unsupervised.py`: Runs extraction using the Qwen2 model.

---

## 2. Model Evaluation (LLM-as-a-Judge)

Each model evaluates the outputs of the other models by reviewing their topic and sentiment responses.

**Evaluation Criteria:**
- Adherence to the original task prompt.
- Relevance and clarity of extracted topics.
- Compliance with requested output format.

**Scripts:**
- `LLM_as_a_judge_llama_contextualized.py`: Llama3 evaluates Qwen2 and ChatGPT.
- `LLM_as_a_judge_qwen_contextualized.py`: Qwen2 evaluates Llama3 and ChatGPT.

---

## 3. Model-Level Assessment

This stage aggregates evaluation scores and comments to produce high-level insights about each model.

**Assessment Includes:**
- Overall strengths and weaknesses
- Quality and consistency of outputs
- Suggestions for improvement

**Scripts:**
- `judge_llama_model_assessment.py`: Summarizes Llama3's judgments.
- `judge_qwen_model_assessment.py`: Summarizes Qwen2's judgments.

---

## Key Notes

- **Model-Agnostic Design**: The pipeline is built to be reusable. The same three-stage logic applies to any new LLM integrated into the workflow.
- **Offline & Customizable**: All processing happens locally, allowing full control over model loading and input preprocessing.
- **Integration with Main Project**: These scripts support the structured analysis and model benchmarking tasks described in the main project’s README.
