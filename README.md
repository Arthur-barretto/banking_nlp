# Topic Modeling and Sentiment Analysis in Brazilian Banks' Quarterly Transcripts

This project aims to compare the performance of traditional NLP topic modeling techniques with the capabilities of Large Language Models (LLMs) in analyzing financial data. The dataset consists of quarterly earnings call transcripts from Brazilian banks, spanning from 2006 to 2023.

## Project Overview

### Motivation

Initially, the project sought to compare traditional NLP approaches like Latent Dirichlet Allocation (LDA) and BERTopic against LLMs (e.g., ChatGPT) for topic modeling. However, preliminary results showed that ChatGPT significantly outperformed traditional methods in extracting meaningful topics. As a result, the project's focus shifted to leveraging LLMs for both topic modeling and sentiment analysis, specifically using ChatGPT and Llama3.

### Objectives

- **Data Preparation and Exploration**: Clean and understand the dataset to ensure it is suitable for analysis.
- **Comparison of Topic Modeling Techniques**: Evaluate the performance of traditional NLP methods (e.g., LDA, BERTopic) against LLM-based approaches.
- **Sentiment Analysis**: Apply LLMs to analyze sentiment trends in the transcripts.
- **LLM Performance Analysis**: Compare ChatGPT and Llama3 in topic extraction and sentiment analysis.

## Dataset

The dataset comprises quarterly earnings call transcripts from Brazilian banks from 2006 to 2023. These transcripts contain rich textual data suitable for both topic modeling and sentiment analysis.

## Files and Structure

### 1. [EDA.ipynb](./EDA.ipynb)

**Purpose**: This notebook is dedicated to data cleaning and exploratory data analysis (EDA).

**Contents**:

- **Data Cleaning**: Removal of unnecessary characters, handling missing values, and standardizing the dataset.
- **Exploratory Analysis**: Initial exploration of word frequencies, distribution of transcripts over time, and other descriptive statistics to understand the dataset's structure and characteristics.

### 2. [Traditional\_NLP\_Workflow.ipynb](./Traditional_NLP_Workflow.ipynb)

**Purpose**: This notebook applies traditional NLP models, such as LDA and BERTopic, to analyze the dataset.

**Contents**:

- **Document-Level Topic Modeling**: Implementation of LDA and BERTopic on entire transcripts.
- **Chunking for Analysis**: Division of transcripts into smaller chunks to assess the impact on model performance.
- **Multilingual Embedding**: Use of BERTopic with multilingual embeddings for better contextual understanding.

### 3. [Separate\_Q&A\_Workflow.ipynb](./Separate_Q\&A_Workflow.ipynb)

**Purpose**: This notebook separates the transcripts into presentation and Q&A sections to improve topic modeling performance.

**Contents**:

- **Separation Process**: Utilizes ChatGPT to identify and split the presentation and Q&A sections in each transcript.
- **Rationale**: Presentation sections have consistent structure and content across documents, which limits their utility for topic modeling. The Q&A sections provide a better outline of distinct topics.

### 4. [ChatGPT\_Workflow.ipynb](./ChatGPT_Workflow.ipynb)

**Purpose**: This notebook applies topic modeling using ChatGPT at the document level.

**Contents**:

- **Document-Level Analysis**: Implementation of ChatGPT-based topic modeling on entire transcripts.
- **Insights into LLM Performance**: Focused evaluation of ChatGPT's ability to extract meaningful topics from full documents.

### 5. [Topic\_Modeling\_QnA.ipynb](./Topic_Modeling_QnA.ipynb)

**Purpose**: This notebook applies LDA and BERTopic to the Q&A sections of the transcripts and compares their results with ChatGPT.

**Contents**:

- **Q&A Topic Modeling**: Implementation of LDA and BERTopic on Q&A sections to extract topics.
- **Comparison with ChatGPT**: Evaluation of results using coherence scores and UMass indicators.
- **Qualitative Insights**: While LDA and BERTopic scored better in quantitative metrics, ChatGPT provided more meaningful and diverse topics upon manual inspection, especially given the dataset's uniform structure and small size.

### 6. [LLM\_llama3.py](./LLM_llama3.py)

**Purpose**: This script implements topic modeling using Llama 3.1.

**Contents**:

- **Local LLM Analysis**: Uses the Llama 3.1 model for analyzing Q&A sections of the transcripts.
- **RAG Pipeline**: Employs a Retrieval-Augmented Generation (RAG) pipeline to structure outputs based on predefined themes.
- **Theme-Based Evaluation**: Extracts insights for specific predefined themes and evaluates sentiment polarity.

### 7. [TopicGPT\_QnA.ipynb](./TopicGPT_QnA.ipynb)

**Purpose**: This notebook consolidates and analyzes the results of topic modeling from ChatGPT 4-turbo and Llama 3.1.

**Contents**:

- **Standardization and Cleaning**: Prepares the topic modeling results for analysis.
- **Exploratory Data Analysis**: Examines the distribution of topic mentions, average sentiment, and related trends.
- **Comparison with Stock Prices**: Analyzes the relationship between Banco do Brasil's stock price and sentiment of selected topics.
- **Model Comparison**: Evaluates the outputs of ChatGPT and Llama 3.1 for specific topics, highlighting differences in their performances.

## How to Navigate

1. Start with the `EDA.ipynb` notebook to understand the dataset and preprocessing steps.
2. Explore `Traditional_NLP_Workflow.ipynb` for a detailed implementation of traditional NLP methods.
3. Dive into `Separate_Q&A_Workflow.ipynb` to review the separation process and its impact on topic modeling.
4. Examine `ChatGPT_Workflow.ipynb` to analyze ChatGPT's topic modeling capabilities at the document level.
5. Review `Topic_Modeling_QnA.ipynb` for a comparative analysis of topic modeling approaches on the Q&A sections.
6. Run `LLM_llama3.py` to explore Llama 3.1's performance in topic modeling and sentiment evaluation.
7. Study `TopicGPT_QnA.ipynb` to understand the combined results and comparative analysis of ChatGPT and Llama 3.1.

## Tools and Frameworks

- **Traditional NLP**: LDA, BERTopic
- **LLMs**: ChatGPT, Llama3
- **Libraries**: Python (pandas, NumPy, matplotlib, seaborn, scikit-learn, etc.)
- **Environment**: Jupyter Notebooks, Python Scripts

