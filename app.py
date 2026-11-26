from contextlib import asynccontextmanager
from pathlib import Path
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.knowledge_assistant import KnowledgeBase

kb = KnowledgeBase()

class Document(BaseModel):
    name: str = "document.txt"
    text: str

class IngestRequest(BaseModel):
    documents: list[Document]

class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=10)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    folder = Path(os.getenv("RAG_DOCS_DIR", "data"))
    docs = [
        {
            "name": path.relative_to(folder).as_posix(),
            "text": path.read_text(encoding="utf-8", errors="ignore"),
        }
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}
    ]
    if docs:
        kb.add_documents(docs)
    yield

app = FastAPI(title="GenAI RAG Knowledge Assistant", version="1.0.0", lifespan=lifespan)

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
