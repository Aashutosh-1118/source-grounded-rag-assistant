import pytest
from langchain_core.documents import Document

import prompt as prompts
from rag import parse_citations

from .conftest import DOCS, load_docs, make_engine

RATE_Q = "What was the 30-year fixed mortgage rate?"
OFF_TOPIC_Q = "quantum chromodynamics lagrangian symmetry"


def test_ingest_and_list(engine):
    docs = engine.list_documents()
    assert {d["source"] for d in docs} == set(DOCS)
    assert all(d["chunks"] >= 1 for d in docs)


def test_reingest_replaces_instead_of_duplicating(engine):
    before = {d["document_id"]: d["chunks"] for d in engine.list_documents()}
    engine.ingest([Document(page_content=DOCS["home.md"])], "doc0", "home.md", "file")
    after = {d["document_id"]: d["chunks"] for d in engine.list_documents()}
    assert before == after


def test_delete_document(engine):
    assert engine.delete_document("doc0") is True
    assert engine.delete_document("doc0") is False
    res = engine.retrieve(RATE_Q, strategy="hybrid")
    assert all(c.doc.metadata["source"] != "home.md" for c in res.chunks)


def test_vector_retrieval_finds_right_document(engine):
    res = engine.retrieve(RATE_Q, strategy="vector", top_k=3)
    assert res.chunks[0].doc.metadata["source"] == "home.md"
    assert res.top_vector_score > 0.3


def test_hybrid_includes_keyword_match(engine):
    res = engine.retrieve("Riverton", strategy="hybrid", top_k=3)
    assert res.chunks[0].doc.metadata["source"] == "rent.md"


def test_rerank_strategy_orders_by_reranker_score(engine):
    class FakeReranker:
        def predict(self, pairs):
            return [1.0 if "vacancy" in text.lower() else 0.0 for _, text in pairs]

    engine._reranker = FakeReranker()
    res = engine.retrieve("mortgage rate", strategy="hybrid_rerank", top_k=3)
    assert res.chunks[0].doc.metadata["source"] == "rent.md"
    assert res.chunks[0].rerank_score == 1.0


def test_strict_mode_refuses_without_calling_llm(tmp_path):
    eng = make_engine(tmp_path, responses=["LLM WAS CALLED"])
    load_docs(eng)
    final = eng.answer(OFF_TOPIC_Q, mode="strict")
    assert final["answer"] == prompts.NOT_FOUND_MESSAGE
    assert final["grounded"] is False
    assert final["sources"] == []


def test_grounded_answer_has_cited_sources(engine):
    final = engine.answer(RATE_Q)
    assert final["grounded"] is True
    assert "6.8" in final["answer"]
    first = final["sources"][0]
    assert first["source"] == "home.md" and first["cited"] is True
    assert all("snippet" in s for s in final["sources"])


def test_flexible_mode_labels_general_knowledge(tmp_path):
    eng = make_engine(tmp_path, responses=["Paris is the capital of France."])
    load_docs(eng)
    final = eng.answer(OFF_TOPIC_Q, mode="flexible")
    assert final["answer"].startswith(prompts.GENERAL_KNOWLEDGE_NOTICE)
    assert final["grounded"] is False
    assert final["sources"] == []


def test_model_saying_not_found_is_not_grounded(tmp_path):
    eng = make_engine(tmp_path, responses=[prompts.NOT_FOUND_MESSAGE])
    load_docs(eng)
    final = eng.answer(RATE_Q)
    assert final["grounded"] is False
    assert final["sources"] == []


def test_followup_is_rewritten_using_history(tmp_path):
    eng = make_engine(tmp_path, responses=["What was the 15-year fixed mortgage rate?", "It was 6.1 percent [1]."])
    load_docs(eng)
    history = [
        {"role": "user", "content": RATE_Q},
        {"role": "assistant", "content": "It was 6.8 percent [1]."},
    ]
    final = eng.answer("and for 15 years?", history=history)
    assert final["standalone_question"] == "What was the 15-year fixed mortgage rate?"
    assert final["sources"][0]["source"] == "home.md"


def test_no_history_skips_rewrite(engine):
    final = engine.answer(RATE_Q)
    assert final["standalone_question"] == RATE_Q


def test_stream_event_order(engine):
    events = list(engine.answer_stream(RATE_Q))
    types = [e["type"] for e in events]
    assert types[0] == "meta" and types[-1] == "final"
    assert "token" in types
    streamed = "".join(e["text"] for e in events if e["type"] == "token")
    assert streamed.strip() == events[-1]["answer"]


def test_bad_domain_and_mode(engine):
    with pytest.raises(KeyError):
        list(engine.answer_stream(RATE_Q, domain="nope"))
    with pytest.raises(ValueError):
        list(engine.answer_stream(RATE_Q, mode="loose"))


def test_parse_citations():
    assert parse_citations("A [1][3] and B [2, 4] and C [9].", max_n=4) == {1, 2, 3, 4}
    assert parse_citations("no citations here", max_n=4) == set()
