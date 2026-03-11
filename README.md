# SEC 8-K RAG Pipeline — MLOps Final Project

An end-to-end Machine Learning Operations pipeline for semantic search and analysis of SEC Form 8-K filings using Retrieval-Augmented Generation (RAG).

---

##  Project Overview

This project builds a production-ready RAG pipeline over 2024 SEC Form 8-K filings for S&P 500 companies. It covers the full MLOps lifecycle: data ingestion, versioning, vector storage, drift detection, and an interactive chatbot interface — all deployed on cloud infrastructure.

**Dataset:** 292 filings · 10 companies · 6 sectors · Full year 2024 (Q1–Q4)  
**Source:** [Kaggle — SEC 8-K Raw Text Filings 2024](https://www.kaggle.com/datasets/datavadar/sec-8k-raw-text-filings-2024)

---

##  Pipeline Architecture

```
Raw SEC Filings (Kaggle)
        │
        ▼
┌─────────────────────┐
│  1. Data Pipeline   │  EDA · S&P 500 filter · Text cleaning · Chunking
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  2. Classification  │  Yahoo Finance industry labels · DVC versioning
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  3. Vector Database │  Azure PostgreSQL · pgvector · OpenAI Embeddings
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  4. Drift Detection │  Cosine similarity · Evidently Cloud monitoring
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  5. RAG Chatbot     │  Gradio UI · LangChain · Metadata filters
└─────────────────────┘
```

---

##  Repository Structure

```
MLOps_Final_Project_SEC_Filings/
├── app.py                  # Gradio RAG chatbot application
├── Dockerfile              # Container definition for deployment
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore
└── MLOps_Final_Project.ipynb  # Full pipeline notebook (Colab)
```

---

##  Tech Stack

| Component | Technology |
|-----------|-----------|
| Data Versioning | DVC + Google Drive |
| Industry Classification | Yahoo Finance (`yfinance`) |
| Vector Database | Azure PostgreSQL + pgvector |
| Embeddings | OpenAI `text-embedding-3-small` (1,536 dims) |
| Vector Search | HNSW index (m=16, ef_construction=64) |
| Drift Detection | Evidently Cloud |
| RAG Framework | LangChain |
| Chatbot UI | Gradio |
| Deployment | Docker + Azure VM |

---

##  Dataset Summary

| Metric | Value |
|--------|-------|
| Raw filings | 58,126 |
| After S&P 500 filter | 503 companies |
| Final selection (all 4 quarters) | 10 companies |
| Total filings used | 292 |
| Text chunks | 5,967 |
| Embedding dimensions | 1,536 |
| Sectors covered | 6 |

**Companies:** JPMorgan Chase · Goldman Sachs · Capital One · ONEOK · EQT Corp · Kroger · Philip Morris · Super Micro · American Water Works · Cencora

---

##  Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/akgupta-droid/MLOps_Final_Project_SEC_Filings.git
cd MLOps_Final_Project_SEC_Filings
```

### 2. Set up environment variables
```bash
cp .env.example .env
# Edit .env and fill in your credentials
```

Required variables:
```
OPENAI_API_KEY=
DB_HOST=
DB_NAME=
DB_USER=
DB_PASSWORD=
DB_PORT=5432
DB_SSLMODE=require
EMBEDDING_MODEL=text-embedding-3-small
GENERATION_MODEL=gpt-5-mini
TOP_K=5
```

### 3. Run with Docker
```bash
docker build -t sec8k-rag .
docker run --env-file .env -p 7860:7860 sec8k-rag
```

### 4. Run locally
```bash
pip install -r requirements.txt
python app.py
```

The Gradio interface will be available at `http://localhost:7860`

---

##  Notebook

The full pipeline is documented in `MLOps_Final_Project.ipynb`, designed to run in Google Colab. It covers all 5 pipeline components end-to-end, from raw data download through chatbot deployment.

---

##  Drift Detection Results

Semantic drift was monitored by splitting data into reference (Q1+Q2: 2,484 chunks) and production (Q3+Q4: 3,483 chunks) sets using Evidently Cloud.

- **Result:** No significant semantic drift detected in embedding features
- **Drift detected in:** `symbol` and `sector` columns (structural metadata, not semantic content)
- **Conclusion:** Model embeddings remain stable — no retraining required

---

##  License

This project was developed as part of an MLOps course final project.
