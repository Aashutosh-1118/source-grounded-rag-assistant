"""The RAG engine: ingestion, hybrid retrieval, grounding gate, streaming answers.

One class (RagEngine) holds all state. The LLM and embeddings can be injected,
which keeps the engine easy to test without API keys or model downloads.
"""

import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

import loaders
import prompt as prompts
from bm25 import BM25Index
from config import VALID_MODES, VALID_STRATEGIES, Settings

RRF_K = 60  # standard constant for reciprocal rank fusion
SNIPPET_CHARS = 300


@dataclass
class RetrievedChunk:
    doc: Document
    vector_score: Optional[float] = None  # cosine similarity, None if found by keywords only
    rerank_score: Optional[float] = None


@dataclass
class RetrievalResult:
    chunks: list
    top_vector_score: float
    strategy: str


def _clean_metadata(meta: dict) -> dict:
    """Chroma only accepts str, int, float and bool values."""
    return {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}


def format_context(chunks: list) -> str:
    parts = []
    for n, c in enumerate(chunks, 1):
        m = c.doc.metadata
        label = m.get("source", "unknown")
        if m.get("page"):
            label += f", page {m['page']}"
        parts.append(f"[{n}] (source: {label})\n{c.doc.page_content.strip()}")
    return "\n\n".join(parts)


def parse_citations(answer: str, max_n: int) -> set:
    """Find passage numbers like [1], [2][3] or [1, 2] in an answer."""
    found = set()
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer):
        for num in group.split(","):
            n = int(num)
            if 1 <= n <= max_n:
                found.add(n)
    return found


class RagEngine:
    def __init__(self, settings: Optional[Settings] = None, llm=None, embeddings=None):
        self.settings = settings or Settings.from_env()
        self._llm = llm
        self._embeddings = embeddings
        self._store = None
        self._bm25 = None
        self._reranker = None
        self._splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    # ------------------------------------------------------------------ components

    @property
    def llm(self):
        if self._llm is None:
            from langchain_groq import ChatGroq

            s = self.settings
            self._llm = ChatGroq(model=s.groq_model, temperature=s.groq_temperature, max_tokens=s.groq_max_tokens)
        return self._llm

    @property
    def embeddings(self):
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.settings.embedding_model,
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    @property
    def store(self) -> Chroma:
        if self._store is None:
            s = self.settings
            self._store = Chroma(
                collection_name=s.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(s.vectorstore_dir),
                # Cosine distance, so 1 - distance is a similarity between 0 and 1.
                collection_metadata={"hnsw:space": "cosine"},
            )
        return self._store

    # ------------------------------------------------------------------ ingestion

    def ingest(self, docs: list, document_id: str, source: str, source_type: str) -> dict:
        chunks = [c for c in self._splitter.split_documents(docs) if c.page_content.strip()]
        if not chunks:
            raise ValueError("No text to index.")
        ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_{i}"
            chunk.metadata = {
                **_clean_metadata(chunk.metadata),
                "document_id": document_id,
                "chunk_index": i,
                "chunk_id": chunk_id,
                "source": source,
                "source_type": source_type,
            }
            ids.append(chunk_id)
        # Replace any older version of the same document.
        self.store.delete(where={"document_id": document_id})
        self.store.add_documents(chunks, ids=ids)
        self._bm25 = None
        return {"document_id": document_id, "source": source, "chunks": len(chunks)}

    def ingest_url(self, url: str) -> dict:
        docs = loaders.load_url(url, allow_private=self.settings.allow_private_urls)
        return self.ingest(docs, loaders.document_id_for_url(url), loaders.normalize_url(url), "web")

    def ingest_file(self, path: Path, display_name: str) -> dict:
        docs = loaders.load_file(path, display_name)
        return self.ingest(docs, loaders.document_id_for_file(display_name), display_name, "file")

    def list_documents(self) -> list:
        data = self.store.get(include=["metadatas"])
        agg = {}
        for m in data.get("metadatas") or []:
            entry = agg.setdefault(
                m["document_id"],
                {
                    "document_id": m["document_id"],
                    "source": m.get("source"),
                    "title": m.get("title") or m.get("source"),
                    "source_type": m.get("source_type"),
                    "chunks": 0,
                },
            )
            entry["chunks"] += 1
        return sorted(agg.values(), key=lambda d: d["source"] or "")

    def delete_document(self, document_id: str) -> bool:
        exists = any(d["document_id"] == document_id for d in self.list_documents())
        if exists:
            self.store.delete(where={"document_id": document_id})
            self._bm25 = None
        return exists

    # ------------------------------------------------------------------ retrieval

    def _keyword_index(self) -> BM25Index:
        if self._bm25 is None:
            data = self.store.get(include=["documents", "metadatas"])
            docs = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(data.get("documents") or [], data.get("metadatas") or [])
            ]
            self._bm25 = BM25Index(docs)
        return self._bm25

    def _vector_search(self, query: str, k: int) -> list:
        results = self.store.similarity_search_with_score(query, k=k)
        return [(doc, 1.0 - float(dist)) for doc, dist in results]

    @staticmethod
    def _fuse(vector_hits: list, keyword_hits: list) -> list:
        """Reciprocal rank fusion: reward chunks that rank well in either list."""
        scores = defaultdict(float)
        docs, vscores = {}, {}
        for rank, (doc, sim) in enumerate(vector_hits):
            cid = doc.metadata["chunk_id"]
            scores[cid] += 1.0 / (RRF_K + rank + 1)
            docs[cid], vscores[cid] = doc, sim
        for rank, (doc, _) in enumerate(keyword_hits):
            cid = doc.metadata["chunk_id"]
            scores[cid] += 1.0 / (RRF_K + rank + 1)
            docs.setdefault(cid, doc)
        order = sorted(scores, key=scores.get, reverse=True)
        return [RetrievedChunk(docs[c], vscores.get(c)) for c in order]

    def _rerank(self, query: str, chunks: list) -> list:
        if not chunks:
            return chunks
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.settings.reranker_model)
        scores = self._reranker.predict([(query, c.doc.page_content) for c in chunks])
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)

    def retrieve(self, query: str, top_k: Optional[int] = None, strategy: Optional[str] = None) -> RetrievalResult:
        s = self.settings
        top_k = top_k or s.top_k
        strategy = strategy or s.retrieval_strategy
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {VALID_STRATEGIES}")
        candidate_k = max(s.candidate_k, top_k)

        vector_hits = self._vector_search(query, candidate_k)
        if not vector_hits:
            return RetrievalResult([], 0.0, strategy)
        top_score = max(sim for _, sim in vector_hits)

        if strategy == "vector":
            ranked = [RetrievedChunk(doc, sim) for doc, sim in vector_hits]
        else:
            keyword_hits = self._keyword_index().search(query, candidate_k)
            ranked = self._fuse(vector_hits, keyword_hits)[:candidate_k]
            if strategy == "hybrid_rerank":
                ranked = self._rerank(query, ranked)
        return RetrievalResult(ranked[:top_k], top_score, strategy)

    # ------------------------------------------------------------------ answering

    def condense_question(self, question: str, history: Optional[list]) -> str:
        """Turn a follow-up like "and for 15 years?" into a standalone question."""
        recent = (history or [])[-2 * self.settings.history_turns :]
        if not recent:
            return question
        lines = []
        for m in recent:
            who = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{who}: {m['content'][:600]}")
        chain = prompts.CONDENSE_PROMPT | self.llm | StrOutputParser()
        rewritten = chain.invoke({"history": "\n".join(lines), "question": question}).strip()
        return rewritten or question

    @staticmethod
    def _source_payload(n: int, chunk: RetrievedChunk, cited: bool) -> dict:
        m = chunk.doc.metadata
        text = " ".join(chunk.doc.page_content.split())
        return {
            "n": n,
            "source": m.get("source"),
            "title": m.get("title") or m.get("source"),
            "page": m.get("page"),
            "snippet": text[:SNIPPET_CHARS] + ("..." if len(text) > SNIPPET_CHARS else ""),
            "score": round(chunk.vector_score, 3) if chunk.vector_score is not None else None,
            "cited": cited,
        }

    def answer_stream(
        self,
        question: str,
        history: Optional[list] = None,
        mode: Optional[str] = None,
        domain: Optional[str] = None,
        top_k: Optional[int] = None,
        strategy: Optional[str] = None,
    ) -> Iterator[dict]:
        """Yield events: one 'meta', many 'token', then one 'final'."""
        started = time.perf_counter()
        s = self.settings
        mode = mode or s.default_mode
        domain = domain or s.default_domain
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}")
        prompts.get_persona(domain)  # fail early on an unknown domain

        standalone = self.condense_question(question, history)
        result = self.retrieve(standalone, top_k, strategy)
        covered = bool(result.chunks) and result.top_vector_score >= s.min_relevance

        yield {
            "type": "meta",
            "standalone_question": standalone,
            "covered_by_sources": covered,
            "top_score": round(result.top_vector_score, 3),
            "strategy": result.strategy,
            "mode": mode,
        }

        def final(answer: str, grounded: bool, sources: list) -> dict:
            return {
                "type": "final",
                "answer": answer,
                "grounded": grounded,
                "sources": sources,
                "standalone_question": standalone,
                "top_score": round(result.top_vector_score, 3),
                "strategy": result.strategy,
                "mode": mode,
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }

        # Not covered, strict: refuse without calling the LLM.
        if not covered and mode == "strict":
            yield {"type": "token", "text": prompts.NOT_FOUND_MESSAGE}
            yield final(prompts.NOT_FOUND_MESSAGE, False, [])
            return

        if covered:
            chain = prompts.build_answer_prompt(domain) | self.llm | StrOutputParser()
            inputs = {"context": format_context(result.chunks), "question": standalone}
            prefix = ""
        else:
            # Not covered, flexible: answer from general knowledge with a fixed label.
            chain = prompts.build_general_prompt(domain) | self.llm | StrOutputParser()
            inputs = {"question": standalone}
            prefix = prompts.GENERAL_KNOWLEDGE_NOTICE
            yield {"type": "token", "text": prefix}

        pieces = []
        for piece in chain.stream(inputs):
            if piece:
                pieces.append(piece)
                yield {"type": "token", "text": piece}
        answer = prefix + "".join(pieces).strip()

        if not covered:
            yield final(answer, False, [])
            return

        # The model may still say the passages are not enough.
        if prompts.NOT_FOUND_MESSAGE in answer:
            yield final(answer, False, [])
            return

        cited = parse_citations(answer, len(result.chunks))
        sources = [self._source_payload(n, c, n in cited) for n, c in enumerate(result.chunks, 1)]
        yield final(answer, True, sources)

    def answer(self, question: str, **kwargs) -> dict:
        """Non-streaming version. Returns the 'final' event."""
        last = None
        for event in self.answer_stream(question, **kwargs):
            if event["type"] == "final":
                last = event
        return last
