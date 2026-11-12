from pathlib import Path
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.knowledge_assistant import KnowledgeBase

app = FastAPI(title="GenAI RAG Knowledge Assistant", version="1.0.0")
kb = KnowledgeBase()

class Document(BaseModel):
    name: str = "document.txt"
    text: str

class IngestRequest(BaseModel):
    documents: list[Document]

class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=10)

@app.on_event("startup")
def load_documents():
    folder = Path(os.getenv("RAG_DOCS_DIR", "data"))
    docs = [{"name": p.name, "text": p.read_text(encoding="utf-8", errors="ignore")}
            for p in folder.glob("**/*") if p.is_file() and p.suffix.lower() in {".txt", ".md"}]
    if docs:
        kb.add_documents(docs)

@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": len(kb.chunks)}

@app.post("/ingest")
def ingest(request: IngestRequest):
    try:
        count = kb.add_documents([d.model_dump() for d in request.documents])
        return {"added_chunks": count, "total_chunks": len(kb.chunks)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/ask")
def ask(request: AskRequest):
    try:
        return kb.answer(request.question, request.top_k)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
