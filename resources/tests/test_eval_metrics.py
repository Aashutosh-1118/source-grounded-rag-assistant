import json
from pathlib import Path

from langchain_core.documents import Document

from evaluation import run_eval as ev
from rag import RetrievedChunk

from .conftest import make_engine


def chunk(source, text):
    return RetrievedChunk(Document(page_content=text, metadata={"source": source}))


Q = {"source": "a.md", "evidence": ["6.8 percent"]}


def test_hit_requires_right_source_and_evidence():
    assert ev.is_hit(chunk("a.md", "rate was 6.8 percent"), Q)
    assert not ev.is_hit(chunk("b.md", "rate was 6.8 percent"), Q)
    assert not ev.is_hit(chunk("a.md", "something else"), Q)


def test_first_hit_rank():
    chunks = [chunk("a.md", "nope"), chunk("a.md", "6.8 percent")]
    assert ev.first_hit_rank(chunks, Q) == 2
    assert ev.first_hit_rank(chunks[:1], Q) is None


def test_keywords_support_alternatives():
    assert ev.keywords_ok("Down payment is 5%.", ["5 percent|5%"])
    assert not ev.keywords_ok("Down payment is 7%.", ["5 percent|5%"])
    assert ev.keywords_ok("between 0.5 and 1.5", ["0.5", "1.5"])


def test_retrieval_summary():
    rows = [
        {"answerable": True, "rank": 1, "top_score": 0.8},
        {"answerable": True, "rank": 2, "top_score": 0.7},
        {"answerable": True, "rank": None, "top_score": 0.6},
        {"answerable": False, "rank": None, "top_score": 0.1},
    ]
    s = ev.summarize_retrieval(rows)
    assert round(s["hit_rate"], 3) == round(2 / 3, 3)
    assert round(s["mrr"], 3) == round((1 + 0.5) / 3, 3)


def test_threshold_sweep_picks_separating_value():
    rows = [
        {"answerable": True, "top_score": 0.7},
        {"answerable": True, "top_score": 0.5},
        {"answerable": False, "top_score": 0.2},
        {"answerable": False, "top_score": 0.3},
    ]
    sweep = ev.sweep_thresholds(rows, [0.1, 0.25, 0.4, 0.6, 0.8])
    best = ev.best_threshold(sweep)
    assert best["balanced_accuracy"] == 1.0
    assert best["threshold"] in (0.4,)


def test_percentile():
    assert ev.percentile([100, 200, 300, 400, 500], 95) == 500
    assert ev.percentile([], 95) == 0.0


def test_eval_set_file_is_well_formed():
    path = Path(__file__).resolve().parent.parent / "evaluation" / "eval_set.json"
    questions = json.loads(path.read_text(encoding="utf-8"))
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))
    for q in questions:
        if q["answerable"]:
            assert q["source"] and q["evidence"] and q["answer_keywords"]


def test_generation_run_smoke(tmp_path):
    """Run the answer loop end to end with a fake LLM, judge off."""
    eng = make_engine(tmp_path, responses=["The rate was 6.8 percent [1]."])
    eng.ingest(
        [Document(page_content="The average 30-year fixed mortgage rate was 6.8 percent in March.")],
        "d1", "home.md", "file",
    )
    questions = [
        {"id": "a", "question": "What was the 30-year fixed mortgage rate?", "answerable": True,
         "source": "home.md", "evidence": ["6.8 percent"], "answer_keywords": ["6.8"]},
        {"id": "b", "question": "quantum chromodynamics lagrangian symmetry", "answerable": False},
    ]
    rows = ev.run_generation(eng, questions, "hybrid", 3, judge=False, sleep=0)
    summary = ev.summarize_generation(rows)
    assert summary["answer_accuracy"] == 1.0
    assert summary["correct_abstain_rate"] == 1.0
    assert summary["false_abstain_rate"] == 0.0
