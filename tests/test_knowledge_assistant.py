from src.knowledge_assistant import KnowledgeBase, chunk_text

def test_chunk_text_preserves_source_and_splits_long_text():
    chunks = chunk_text("alpha beta gamma delta epsilon", "notes.txt", chunk_size=12, overlap=3)
    assert len(chunks) > 1
    assert all(chunk.source == "notes.txt" for chunk in chunks)
    assert chunks[0].index == 0

def test_knowledge_base_returns_source_citations():
    kb = KnowledgeBase()
    kb.add_documents([{"name": "guide.txt", "text": "Retrieval augmented generation combines search with language models."}])
    result = kb.answer("What combines search with language models?")
    assert result["citations"]
    assert result["citations"][0]["source"] == "guide.txt"

def test_knowledge_base_abstains_when_no_match():
    kb = KnowledgeBase()
    kb.add_documents([{"name": "guide.txt", "text": "The invoice is due on Friday."}])
    result = kb.answer("How do I bake sourdough bread?", min_score=0.9)
    assert result["citations"] == []
