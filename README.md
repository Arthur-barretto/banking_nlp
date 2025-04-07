# Topic Modeling and Sentiment Analysis on Brazilian Bank Earnings Calls  
Benchmarking LDA, BERTopic, GPT-4-turbo, Llama3, and Qwen2 (2006–2023)

This project explores the use of Natural Language Processing (NLP) for analyzing quarterly earnings call transcripts from Brazilian banks. It benchmarks traditional topic modeling methods (LDA, BERTopic) against modern Large Language Models (LLMs), such as GPT-4-turbo, Llama3, and Qwen2. Tasks include both topic modeling and sentiment analysis, applied to a dataset spanning 2006 to 2023.

## Project Structure

This research is organized into three benchmark tasks:

### 1. Unstructured Topic Modeling
- Compare traditional models (LDA, BERTopic) with GPT-4-turbo on full or Q&A transcript sections.
- Evaluate coherence, relevance, and output diversity.

### 2. LLM Benchmarking (Unstructured)
- Use the novel LLM-as-a-Judge framework to compare GPT-4-turbo, Llama3, and Qwen2.
- Each LLM scores the outputs of others on clarity, structure, and relevance.

### 3. Structured Topic Modeling & Sentiment Analysis
- Label 30 documents with expert-reviewed ground truth.
- Use LLMs to classify presence and sentiment of 16 predefined financial topics.

## Repository Contents

### Notebooks

These notebooks contain the full end-to-end workflow of the analysis:

- `1.EDA.ipynb`  
  Purpose: Data cleaning and exploratory analysis.  
  Includes distribution analysis, token counts, and Q&A vs. presentation segmentation.

- `2.Traditional_NLP_Workflow.ipynb`  
  Purpose: Apply LDA and BERTopic for document-level topic modeling.  
  Includes coherence scoring and embedding visualizations.

- `3.Separate_Q&A_Workflow.ipynb`  
  Purpose: Use GPT to separate transcripts into Q&A and presentation sections.  
  Includes prompt logic and justification for Q&A focus.

- `4.ChatGPT_Workflow.ipynb`  
  Purpose: Topic modeling using GPT-4-turbo at the document level.  
  Includes topic generation and qualitative evaluation.

- `5.Topic_Modeling_QnA.ipynb`  
  Purpose: Compare traditional models and GPT-4-turbo on Q&A-only data.  
  Includes coherence metrics and qualitative comparisons.

- `6.TopicGPT_QnA.ipynb`  
  Purpose: Consolidate LLM outputs and compare trends.  
  Includes sentiment trends and correlation with stock data.

- `7.LLM_as_a_judge_chatgpt.ipynb`  
  Purpose: Use GPT-4-turbo to evaluate outputs from all LLMs.  
  Includes prompt structure and scoring logic.

- `8.LLM_as_a_judge_score.ipynb`  
  Purpose: Analyze scores produced by the LLM-as-a-Judge framework.  
  Includes summary statistics and model performance ranking.

- `9.Structured_analysis_results.ipynb`  
  Purpose: Evaluate how well models detect and classify labeled topics.  
  Includes F1-score and accuracy comparisons.

### Scripts

> **Note:** These scripts are used to run Llama3 and Qwen2 locally due to model size and inference cost.

- `llama3_analysis.py`  
  Purpose: Run structured topic and sentiment analysis using Llama3 (7B).  
  Loads model locally and processes all transcripts for theme detection and polarity classification.

- `qwen2_analysis.py`  
  Purpose: Perform the same analysis using Qwen2 (8B).  
  Useful for comparing lightweight open-source models with GPT-4-turbo.

## Evaluation Framework: LLM-as-a-Judge

To score and compare LLM outputs, we implement a novel evaluation method where each LLM (GPT-4, Llama3, Qwen2) scores the others on:

- Clarity  
- Adherence to prompt format  
- Depth of analysis  

This framework helps reduce human bias and enables scalable, reproducible evaluation.

## Key Results

- GPT-4-turbo consistently outperformed other models across all tasks, especially in contextual richness and accuracy.
- Traditional models (LDA, BERTopic) were more interpretable but underperformed on semantic precision and topic variety.
- Qwen2 showed promise as a lightweight alternative, though less consistent.
- Llama3 performed reasonably well in structured settings but showed lower reliability in unstructured tasks.
- The LLM-as-a-Judge framework proved effective in automating model evaluation and capturing nuances traditional metrics miss.

## Mapping Thesis Sections to Notebooks

| Thesis Section                         | Notebook/Script                        |
|----------------------------------------|----------------------------------------|
| 4. EDA                                 | `1.EDA.ipynb`                          |
| 5.1 Traditional NLP                    | `2.Traditional_NLP_Workflow.ipynb`    |
| 5.1.3 GPT-4 vs LDA/BERT                | `4.ChatGPT_Workflow.ipynb`, `5.Topic_Modeling_QnA.ipynb` |
| 5.2 and 5.3 LLM Comparison             | `6.TopicGPT_QnA.ipynb`, `7.LLM_as_a_judge_chatgpt.ipynb`, `8.LLM_as_a_judge_score.ipynb` |
| 5.4 Structured + Sentiment Evaluation  | `9.Structured_analysis_results.ipynb`, `llama3_analysis.py`, `qwen2_analysis.py` |
| Annexes (Prompts and Evaluation)       | Embedded in code or notebooks         |

## Tools and Technologies

- Traditional NLP: LDA (Gensim), BERTopic (with BERTimbau embeddings)
- LLMs: GPT-4-turbo (via API), Llama3 (7B, local), Qwen2 (8B, local)
- Evaluation: Coherence, UMass, F1-score, accuracy, LLM-as-a-Judge
- Stack: Python, Jupyter, OpenAI API, HuggingFace, UMAP, HDBScan

## Citation

If you reference this project in academic work, please cite the dissertation:

Barros, Arthur. *NLP in Brazilian Banking Results: Comparison Between Traditional Topic Modeling Techniques and LLMs*. Fundação Getúlio Vargas, 2025.
