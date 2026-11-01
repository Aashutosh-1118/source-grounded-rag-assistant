"""Turn URLs and uploaded files into LangChain Documents."""

import hashlib
import ipaddress
import socket
from pathlib import Path
from urllib.parse import urlparse

from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def normalize_url(url: str) -> str:
    return url.strip().rstrip("/")


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def document_id_for_url(url: str) -> str:
    """Same URL always maps to the same id, so re-adding it replaces the old chunks."""
    return _short_hash("url:" + normalize_url(url))


def document_id_for_file(name: str) -> str:
    """Same file name always maps to the same id, so re-uploading updates the document."""
    return _short_hash("file:" + name.strip().lower())


def validate_public_url(url: str) -> None:
    """Reject non-http URLs and addresses that point inside a private network.

    This blocks the obvious server-side request forgery cases for a hosted demo.
    It does not defend against a public page that redirects to a private address.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Only http and https URLs are supported.")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        raise ValueError(f"Could not resolve host: {parsed.hostname}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError("URLs that point to private or local addresses are not allowed.")


def load_url(url: str, allow_private: bool = False) -> list:
    from langchain_community.document_loaders import WebBaseLoader

    url = normalize_url(url)
    if not allow_private:
        validate_public_url(url)
    elif urlparse(url).scheme not in ("http", "https"):
        raise ValueError("Only http and https URLs are supported.")

    loader = WebBaseLoader(
        url,
        requests_kwargs={"timeout": 20},
        bs_get_text_kwargs={"separator": "\n", "strip": True},
    )
    docs = [d for d in loader.load() if d.page_content.strip()]
    if not docs:
        raise ValueError("No readable text found on that page.")
    for d in docs:
        title = (d.metadata.get("title") or "").strip()
        d.metadata = {"source": url, "title": title or url, "source_type": "web"}
    return docs


def load_file(path: Path, display_name: str) -> list:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{suffix}'. Use one of {sorted(SUPPORTED_EXTENSIONS)}.")

    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        raw = PyPDFLoader(str(path)).load()
        docs = [
            Document(
                page_content=d.page_content,
                metadata={"page": int(d.metadata.get("page", 0)) + 1},
            )
            for d in raw
        ]
    elif suffix == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        docs = Docx2txtLoader(str(path)).load()
    else:
        docs = [Document(page_content=path.read_text(encoding="utf-8", errors="replace"))]

    docs = [d for d in docs if d.page_content.strip()]
    if not docs:
        raise ValueError("No readable text found in this file. Scanned PDFs need OCR first.")
    for d in docs:
        extra = {"page": d.metadata["page"]} if "page" in d.metadata else {}
        d.metadata = {"source": display_name, "title": display_name, "source_type": "file", **extra}
    return docs
