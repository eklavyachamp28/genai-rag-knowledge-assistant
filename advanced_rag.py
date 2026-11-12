"""Optional dense RAG pipeline: LangChain + Chroma + Hugging Face + LangGraph.

Install requirements-advanced.txt. Model downloads require network access.
No external LLM API is needed: the answer is extractive and source-grounded.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TypedDict
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END


class RAGState(TypedDict):
    question: str
    documents: list[Document]
    answer: str


@dataclass
class DenseRAG:
    store: Chroma
    min_score: float = 0.30

    @classmethod
    def from_texts(cls, texts: list[str], sources: list[str], model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not texts or len(texts) != len(sources):
            raise ValueError("texts and sources must be nonempty and equal-length")
        docs = [Document(page_content=t, metadata={"source": s}) for t, s in zip(texts, sources)]
        embeddings = HuggingFaceEmbeddings(model_name=model)
        store = Chroma.from_documents(docs, embedding=embeddings, collection_name="knowledge_assistant")
        return cls(store=store)

    def ask(self, question: str, k: int = 3) -> dict:
        if not question.strip() or k < 1:
            raise ValueError("Provide a nonempty question and k >= 1")

        def retrieve(state: RAGState):
            # Chroma relevance scores are normalized similarity values, not calibrated probabilities.
            hits = self.store.similarity_search_with_relevance_scores(state["question"], k=k)
            return {"documents": [d for d, score in hits if score >= self.min_score]}

        def answer(state: RAGState):
            if not state["documents"]:
                return {"answer": "Insufficient evidence in the indexed documents."}
            snippets = [
                f"[{d.metadata.get('source', 'unknown')}] {d.page_content[:400]}"
                for d in state["documents"]
            ]
            return {"answer": "\n\n".join(snippets)}

        graph = StateGraph(RAGState)
        graph.add_node("retrieve", retrieve)
        graph.add_node("answer", answer)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "answer")
        graph.add_edge("answer", END)
        result = graph.compile().invoke({"question": question, "documents": [], "answer": ""})
        return {"answer": result["answer"], "sources": [d.metadata.get("source", "unknown") for d in result["documents"]]}
