# GenAI RAG Knowledge Assistant

A source-grounded document question-answering demo. It ingests local text/Markdown, creates overlapping chunks, retrieves relevant passages using TF-IDF cosine similarity, and returns evidence with source citations. The included extractive baseline is intentionally runnable without paid API credentials.

## Features
- Text and Markdown document ingestion
- Configurable chunk size and overlap
- Ranked retrieval with similarity scores and source/chunk citations
- Abstains when retrieval confidence is weak
- FastAPI endpoints: /health, /ingest, /ask
- Streamlit demo interface and pytest coverage

## Architecture
Documents → chunking → TF-IDF vectorization → cosine retrieval → cited evidence response.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```
API docs: http://127.0.0.1:8000/docs

In another terminal: `streamlit run streamlit_app.py`

## API example
```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question":"What is retrieval augmented generation?","top_k":3}'
```

Ingest documents through POST /ingest with JSON shaped as {"documents":[{"name":"guide.txt","text":"..."}]}.

## Evaluation and extensions
Evaluate Recall@k, MRR and answer faithfulness on a labelled question/context set before reporting benchmark results. The lexical TF-IDF retriever can be replaced with dense embeddings and an LLM generation adapter. Production use also requires authentication, document-level access controls, tenant isolation and monitoring.

## Stack
Python · scikit-learn · FastAPI · Streamlit · pytest

Only ingest documents you are authorized to process. This demo is not a secure multi-tenant document store.
