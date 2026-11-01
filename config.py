"""Central settings. Everything tunable lives here and can be set from a .env file."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("USER_AGENT", "source-grounded-rag/1.0")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

BASE_DIR = Path(__file__).parent

VALID_STRATEGIES = ("vector", "hybrid", "hybrid_rerank")
VALID_MODES = ("strict", "flexible")


def _get(name, default, cast=str):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return cast(raw.strip())


def _as_bool(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # LLM (Groq). Model is an env var so a model deprecation needs no code change.
    groq_model: str = "openai/gpt-oss-120b"
    groq_temperature: float = 0.2
    groq_max_tokens: int = 5000

    # Embeddings and storage
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vectorstore_dir: Path = BASE_DIR / "resources" / "vectorstore"
    collection_name: str = "documents"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    top_k: int = 5  # passages sent to the LLM
    candidate_k: int = 20  # candidates pulled before fusion and reranking
    retrieval_strategy: str = "hybrid"  # vector | hybrid | hybrid_rerank
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Grounding. Cosine similarity of the best vector match must reach this value,
    # otherwise the question counts as "not covered by the sources".
    min_relevance: float = 0.25
    default_mode: str = "strict"  # strict | flexible
    default_domain: str = "general"

    # Chat
    history_turns: int = 3

    # API
    max_upload_mb: int = 20
    allow_private_urls: bool = False
    cors_origins: tuple = field(default_factory=lambda: ("http://localhost:8501",))

    # UI
    api_url: str = "http://localhost:8000"

    def __post_init__(self):
        if self.retrieval_strategy not in VALID_STRATEGIES:
            raise ValueError(f"RETRIEVAL_STRATEGY must be one of {VALID_STRATEGIES}")
        if self.default_mode not in VALID_MODES:
            raise ValueError(f"DEFAULT_MODE must be one of {VALID_MODES}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()  # defaults
        origins = _get("CORS_ORIGINS", ",".join(d.cors_origins))
        return cls(
            groq_model=_get("GROQ_MODEL", d.groq_model),
            groq_temperature=_get("GROQ_TEMPERATURE", d.groq_temperature, float),
            groq_max_tokens=_get("GROQ_MAX_TOKENS", d.groq_max_tokens, int),
            embedding_model=_get("EMBEDDING_MODEL", d.embedding_model),
            vectorstore_dir=_get("VECTORSTORE_DIR", d.vectorstore_dir, Path),
            collection_name=_get("COLLECTION_NAME", d.collection_name),
            chunk_size=_get("CHUNK_SIZE", d.chunk_size, int),
            chunk_overlap=_get("CHUNK_OVERLAP", d.chunk_overlap, int),
            top_k=_get("TOP_K", d.top_k, int),
            candidate_k=_get("CANDIDATE_K", d.candidate_k, int),
            retrieval_strategy=_get("RETRIEVAL_STRATEGY", d.retrieval_strategy),
            reranker_model=_get("RERANKER_MODEL", d.reranker_model),
            min_relevance=_get("MIN_RELEVANCE", d.min_relevance, float),
            default_mode=_get("DEFAULT_MODE", d.default_mode),
            default_domain=_get("DEFAULT_DOMAIN", d.default_domain),
            history_turns=_get("HISTORY_TURNS", d.history_turns, int),
            max_upload_mb=_get("MAX_UPLOAD_MB", d.max_upload_mb, int),
            allow_private_urls=_get("ALLOW_PRIVATE_URLS", d.allow_private_urls, _as_bool),
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
            api_url=_get("API_URL", d.api_url).rstrip("/"),
        )
