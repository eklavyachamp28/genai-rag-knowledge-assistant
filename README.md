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

Index a sample document, then ask a question about it:

```bash
curl --fail-with-body -sS http://127.0.0.1:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"documents":[{"name":"guide.txt","text":"Retrieval augmented generation retrieves relevant evidence before answering."}]}'

curl --fail-with-body -sS http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is retrieval augmented generation?","top_k":3}'
```

The second response includes an answer and a citation to `guide.txt`. If the index is empty, `/ask` returns HTTP 400 until you ingest documents. The index lives in memory. After restarting, re-ingest uploaded documents or place text and Markdown files in `data/` so they are loaded at startup.

## Evaluation and extensions
Evaluate Recall@k, MRR and answer faithfulness on a labelled question/context set before reporting benchmark results. The lexical TF-IDF retriever can be replaced with dense embeddings and an LLM generation adapter. Production use also requires authentication, document-level access controls, tenant isolation and monitoring.

## Stack
Python · scikit-learn · FastAPI · Streamlit · pytest

Only ingest documents you are authorized to process. This demo is not a secure multi-tenant document store.

## Advanced RAG extension (optional)
Install `pip install -r requirements-advanced.txt`. The `advanced_rag.py` module implements **LangChain Documents**, **Hugging Face sentence-transformer embeddings**, **Chroma vector database**, and a two-node **LangGraph** retrieve → answer workflow. It returns source-labeled extractive evidence rather than claiming to generate LLM answers. First execution downloads a model. Example:

```python
from advanced_rag import DenseRAG
rag = DenseRAG.from_texts(["RAG retrieves evidence before answering."], ["guide.md"])
print(rag.ask("What does RAG do?"))
```

`rag_evaluation.py` reports **Recall@k**, **MRR@k**, **Precision@k**, and **nDCG@k**. Pass ranked source IDs and labeled relevant source IDs to `evaluate_retrieval`. For example, `evaluate_retrieval([{"retrieved": ["guide.md", "faq.md"], "relevant": ["guide.md"]}], k=2)` returns recall, MRR, and nDCG of `1.0` and precision of `0.5`.

`lexical_groundedness` is a separate lexical-overlap proxy. The proxy is *not* a validated LLM-as-judge or faithfulness metric. Benchmark results have not been established. Existing TF-IDF API remains the default lightweight baseline.
