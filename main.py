"""Streamlit UI. It only talks to the FastAPI backend over HTTP.

Run the backend first:  uvicorn api:app --reload
Then the UI:            streamlit run main.py
"""

import json
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Source-Grounded Research Assistant", page_icon="📚", layout="wide")


# ---------------------------------------------------------------------- API helpers


def api_get(path):
    resp = requests.get(f"{API_URL}{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_post(path, **kwargs):
    resp = requests.post(f"{API_URL}{path}", timeout=300, **kwargs)
    resp.raise_for_status()
    return resp.json()


def stream_answer(payload: dict, holder: dict):
    """Yield answer tokens. The meta and final events are stored in `holder`."""
    with requests.post(f"{API_URL}/ask/stream", json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        resp.encoding = "utf-8"  # event streams carry no charset, requests would guess latin-1
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event["type"] == "token":
                yield event["text"]
            elif event["type"] in ("meta", "final"):
                holder[event["type"]] = event
            elif event["type"] == "error":
                raise RuntimeError(event["message"])


def show_ingest_result(result: dict):
    for item in result.get("ingested", []):
        st.success(f"Added {item['source']} ({item['chunks']} chunks)")
    for item in result.get("errors", []):
        st.error(f"{item['source']}: {item['error']}")


# ---------------------------------------------------------------------- state

st.session_state.setdefault("messages", [])
st.session_state.setdefault("reset_counter", 0)

try:
    domains = api_get("/domains")
    documents = api_get("/documents")
    backend_ok = True
except requests.RequestException:
    domains, documents, backend_ok = [], [], False

# ---------------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Settings")
    domain_labels = {d["label"]: d["key"] for d in domains}
    domain_label = st.selectbox("Domain", list(domain_labels) or ["General research"])
    domain = domain_labels.get(domain_label, "general")

    mode = st.radio(
        "When sources do not cover the question",
        ["strict", "flexible"],
        format_func=lambda m: "Say it is not found (strict)" if m == "strict" else "Use general knowledge, labeled (flexible)",
    )
    with st.expander("Retrieval options"):
        strategy = st.selectbox("Strategy", ["hybrid", "vector", "hybrid_rerank"])
        top_k = st.slider("Passages sent to the model", 3, 10, 5)

    st.divider()
    st.header("Add sources")
    key = st.session_state.reset_counter
    urls_text = st.text_area("Article URLs, one per line", key=f"urls_{key}", placeholder="https://...")
    uploads = st.file_uploader(
        "Or upload files", type=["pdf", "txt", "md", "docx"], accept_multiple_files=True, key=f"files_{key}"
    )

    if st.button("Add sources", type="primary", use_container_width=True, disabled=not backend_ok):
        urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
        if not urls and not uploads:
            st.warning("Enter a URL or choose a file first.")
        else:
            try:
                with st.spinner("Reading and indexing..."):
                    if urls:
                        show_ingest_result(api_post("/ingest/urls", json={"urls": urls}))
                    if uploads:
                        files = [("files", (f.name, f.getvalue())) for f in uploads]
                        show_ingest_result(api_post("/ingest/files", files=files))
                st.session_state.reset_counter += 1
                st.session_state.messages = []
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")

    st.divider()
    st.header(f"Loaded sources ({len(documents)})")
    if not documents:
        st.caption("Nothing loaded yet.")
    for doc in documents:
        left, right = st.columns([5, 1])
        left.caption(f"{doc['source']}  \n{doc['chunks']} chunks")
        if right.button("✕", key=f"del_{doc['document_id']}", help="Remove this source"):
            requests.delete(f"{API_URL}/documents/{doc['document_id']}", timeout=30)
            st.session_state.messages = []
            st.rerun()

    if st.session_state.messages and st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------- main panel

st.title("Source-Grounded Research Assistant")
st.caption("Ask questions about your own documents. Every answer cites the passages it came from.")

if not backend_ok:
    st.error(f"Cannot reach the backend at {API_URL}. Start it with: uvicorn api:app --reload")
    st.stop()

if not documents:
    st.info("Add one or more URLs or files in the sidebar to begin.", icon="👈")


def render_extras(msg: dict):
    final = msg.get("final") or {}
    if not final:
        return
    if final.get("standalone_question") and final["standalone_question"] != msg.get("question"):
        st.caption(f"Searched for: {final['standalone_question']}")
    if final.get("grounded"):
        cited = [s for s in final["sources"] if s["cited"]]
        others = [s for s in final["sources"] if not s["cited"]]
        with st.expander(f"Sources ({len(cited)} cited, {len(others)} other retrieved)"):
            for group, title in ((cited, "Cited"), (others, "Also retrieved")):
                if not group:
                    continue
                st.markdown(f"**{title}**")
                for s in group:
                    where = f", page {s['page']}" if s.get("page") else ""
                    name = f"[{s['source']}]({s['source']})" if str(s["source"]).startswith("http") else s["source"]
                    score = f" (similarity {s['score']})" if s.get("score") is not None else ""
                    st.markdown(f"[{s['n']}] {name}{where}{score}")
                    st.caption(s["snippet"])
    st.caption(f"{final.get('latency_ms', 0)} ms, {final.get('strategy')} retrieval, best similarity {final.get('top_score')}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_extras(msg)

query = st.chat_input("Ask about your sources...", disabled=not documents)
if query:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages][-6:]
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    payload = {
        "question": query,
        "history": history,
        "mode": mode,
        "domain": domain,
        "top_k": top_k,
        "strategy": strategy,
    }
    holder = {}
    with st.chat_message("assistant"):
        try:
            text = st.write_stream(stream_answer(payload, holder))
        except Exception as exc:
            text = f"Something went wrong: {exc}"
            st.error(text)
    st.session_state.messages.append(
        {"role": "assistant", "content": text, "question": query, "final": holder.get("final")}
    )
    st.rerun()
