"""Shared test helpers. Tests use fake embeddings and a fake LLM, so they need no API key or downloads."""

import hashlib
import math

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from bm25 import tokenize
from config import Settings
from rag import RagEngine

DIM = 1024


class FakeEmbeddings(Embeddings):
    """Hashes words into a fixed-size vector. Texts that share words get a high cosine similarity."""

    def embed_query(self, text):
        vec = [0.0] * DIM
        for token in tokenize(text):
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            vec[0], norm = 1.0, 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


DOCS = {
    "home.md": "The average 30-year fixed mortgage rate was 6.8 percent in March. The 15-year fixed rate was 6.1 percent.",
    "rent.md": "The median rent for a one-bedroom apartment in Riverton was 1,450 dollars. Vacancy was 3.1 percent.",
    "plants.md": "Photosynthesis converts sunlight into chemical energy inside plant leaves.",
}


def make_engine(tmp_path, responses=None, **overrides):
    overrides.setdefault("min_relevance", 0.3)
    settings = Settings(vectorstore_dir=tmp_path / "vs", collection_name="test_docs", **overrides)
    llm = FakeListChatModel(responses=responses or ["The rate was 6.8 percent [1]."])
    return RagEngine(settings, llm=llm, embeddings=FakeEmbeddings())


def load_docs(engine, docs=None):
    for i, (name, text) in enumerate((docs or DOCS).items()):
        engine.ingest([Document(page_content=text)], document_id=f"doc{i}", source=name, source_type="file")


@pytest.fixture
def engine(tmp_path):
    eng = make_engine(tmp_path)
    load_docs(eng)
    return eng
