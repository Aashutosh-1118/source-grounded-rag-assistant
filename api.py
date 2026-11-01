"""FastAPI backend. Run with: uvicorn api:app --reload"""

import json
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import loaders
import prompt as prompts
from config import Settings
from rag import RagEngine

app = FastAPI(title="Source-Grounded Research Assistant", version="1.0.0")

_settings = Settings.from_env()
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_settings.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_engine() -> RagEngine:
    return RagEngine(Settings.from_env())


# ---------------------------------------------------------------------- models


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list)
    mode: Optional[Literal["strict", "flexible"]] = None
    domain: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=15)
    strategy: Optional[Literal["vector", "hybrid", "hybrid_rerank"]] = None


class IngestUrlsRequest(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=10)


def _ask_kwargs(req: AskRequest) -> dict:
    if req.domain is not None and req.domain not in prompts.DOMAIN_PRESETS:
        raise HTTPException(400, f"Unknown domain '{req.domain}'.")
    return {
        "history": [m.model_dump() for m in req.history],
        "mode": req.mode,
        "domain": req.domain,
        "top_k": req.top_k,
        "strategy": req.strategy,
    }


# ---------------------------------------------------------------------- routes


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/domains")
def domains():
    return prompts.list_domains()


@app.get("/documents")
def documents(engine: RagEngine = Depends(get_engine)):
    return engine.list_documents()


@app.delete("/documents/{document_id}")
def delete_document(document_id: str, engine: RagEngine = Depends(get_engine)):
    if not engine.delete_document(document_id):
        raise HTTPException(404, "Document not found.")
    return {"deleted": document_id}


@app.post("/ingest/urls")
def ingest_urls(req: IngestUrlsRequest, engine: RagEngine = Depends(get_engine)):
    ingested, errors = [], []
    for url in req.urls:
        try:
            ingested.append(engine.ingest_url(url))
        except ValueError as exc:
            errors.append({"source": url, "error": str(exc)})
        except Exception as exc:  # network failures, parser errors
            errors.append({"source": url, "error": f"Could not load this page: {exc}"})
    return {"ingested": ingested, "errors": errors}


@app.post("/ingest/files")
def ingest_files(files: list[UploadFile] = File(...), engine: RagEngine = Depends(get_engine)):
    max_bytes = engine.settings.max_upload_mb * 1024 * 1024
    ingested, errors = [], []
    for upload in files:
        name = Path(upload.filename or "upload").name  # strip any directory parts
        if Path(name).suffix.lower() not in loaders.SUPPORTED_EXTENSIONS:
            errors.append({"source": name, "error": f"Unsupported type. Use {sorted(loaders.SUPPORTED_EXTENSIONS)}."})
            continue
        data = upload.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            errors.append({"source": name, "error": f"File is larger than {engine.settings.max_upload_mb} MB."})
            continue
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / name
                path.write_bytes(data)
                ingested.append(engine.ingest_file(path, name))
        except ValueError as exc:
            errors.append({"source": name, "error": str(exc)})
        except Exception as exc:
            errors.append({"source": name, "error": f"Could not read this file: {exc}"})
    return {"ingested": ingested, "errors": errors}


@app.post("/ask")
def ask(req: AskRequest, engine: RagEngine = Depends(get_engine)):
    kwargs = _ask_kwargs(req)
    try:
        return engine.answer(req.question, **kwargs)
    except Exception as exc:
        raise HTTPException(502, f"Answer generation failed: {exc}")


@app.post("/ask/stream")
def ask_stream(req: AskRequest, engine: RagEngine = Depends(get_engine)):
    """Server-sent events: 'meta', then 'token' events, then 'final' (or 'error')."""
    kwargs = _ask_kwargs(req)

    def events():
        try:
            for event in engine.answer_stream(req.question, **kwargs):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
