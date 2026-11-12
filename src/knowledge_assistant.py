"""Explainable lexical retrieval baseline with source citations."""
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Chunk:
    source: str
    index: int
    text: str

def chunk_text(text, source, chunk_size=900, overlap=120):
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    chunks, start, index = [], 0, 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(source, index, piece))
            index += 1
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks

class KnowledgeBase:
    def __init__(self):
        self.chunks, self.vectorizer, self.matrix = [], TfidfVectorizer(stop_words="english", ngram_range=(1, 2)), None

    def add_documents(self, documents, chunk_size=900, overlap=120):
        added = []
        for doc in documents:
            if str(doc.get("text", "")).strip():
                added.extend(chunk_text(str(doc["text"]), str(doc.get("name", "untitled")), chunk_size, overlap))
        if not added and not self.chunks:
            raise ValueError("No non-empty documents supplied")
        self.chunks.extend(added)
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])
        return len(added)

    def search(self, question, top_k=3):
        if self.matrix is None:
            raise RuntimeError("Ingest documents first")
        if not question.strip():
            raise ValueError("Question cannot be empty")
        scores = cosine_similarity(self.vectorizer.transform([question]), self.matrix).ravel()
        order = np.argsort(scores)[::-1][:max(1, min(top_k, len(scores)))]
        return [{"source": self.chunks[i].source, "chunk": self.chunks[i].index,
                 "text": self.chunks[i].text, "score": float(scores[i])} for i in order]

    def answer(self, question, top_k=3, min_score=0.02):
        hits = self.search(question, top_k)
        useful = [h for h in hits if h["score"] >= min_score]
        if not useful:
            return {"answer": "No relevant information was found in indexed documents.", "citations": [], "retrieval": hits}
        answer = "\n\n".join(f"[{i+1}] {h['text']}" for i, h in enumerate(useful))
        citations = [{"source": h["source"], "chunk": h["chunk"], "score": h["score"]} for h in useful]
        return {"answer": answer, "citations": citations, "retrieval": hits}
