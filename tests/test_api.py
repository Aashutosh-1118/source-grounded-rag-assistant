import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import loaders
from api import app, get_engine

from .conftest import make_engine


@pytest.fixture
def client(tmp_path):
    eng = make_engine(tmp_path, responses=["The vault code is 4417 [1]."])
    app.dependency_overrides[get_engine] = lambda: eng
    yield TestClient(app)
    app.dependency_overrides.clear()


def upload_vault(client):
    files = [("files", ("vault.txt", b"The vault code is 4417. Keep it secret.", "text/plain"))]
    return client.post("/ingest/files", files=files)


def test_health_and_domains(client):
    assert client.get("/health").json() == {"status": "ok"}
    keys = {d["key"] for d in client.get("/domains").json()}
    assert {"general", "real_estate"} <= keys


def test_upload_list_ask_delete(client):
    res = upload_vault(client).json()
    assert len(res["ingested"]) == 1 and res["errors"] == []

    docs = client.get("/documents").json()
    assert [d["source"] for d in docs] == ["vault.txt"]

    answer = client.post("/ask", json={"question": "What is the vault code?"}).json()
    assert answer["grounded"] is True
    assert answer["sources"][0]["source"] == "vault.txt"

    assert client.delete(f"/documents/{docs[0]['document_id']}").status_code == 200
    assert client.delete(f"/documents/{docs[0]['document_id']}").status_code == 404


def test_stream_endpoint_emits_meta_tokens_final(client):
    upload_vault(client)
    with client.stream("POST", "/ask/stream", json={"question": "What is the vault code?"}) as resp:
        events = [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]
    types = [e["type"] for e in events]
    assert types[0] == "meta" and types[-1] == "final" and "token" in types


def test_unsupported_file_type_reported(client):
    res = client.post("/ingest/files", files=[("files", ("run.exe", b"x", "application/octet-stream"))]).json()
    assert res["ingested"] == [] and "Unsupported" in res["errors"][0]["error"]


def test_private_url_rejected(client):
    res = client.post("/ingest/urls", json={"urls": ["http://127.0.0.1/secret"]}).json()
    assert res["ingested"] == [] and len(res["errors"]) == 1


def test_url_ingest_uses_loader(client, monkeypatch):
    monkeypatch.setattr(
        loaders,
        "load_url",
        lambda url, allow_private=False: [
            Document(page_content="Page text about vault codes.", metadata={"source": url, "title": "T", "source_type": "web"})
        ],
    )
    res = client.post("/ingest/urls", json={"urls": ["https://example.com/a"]}).json()
    assert res["ingested"][0]["source"] == "https://example.com/a"


def test_unknown_domain_is_400(client):
    assert client.post("/ask", json={"question": "hi", "domain": "nope"}).status_code == 400


def test_empty_question_is_422(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
