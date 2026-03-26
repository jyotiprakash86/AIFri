"""
Single-page Streamlit RAG Evaluation Engine
Context: AI Hackathon submission evaluation with ChromaDB + recursive chunking + OpenAI-compatible chat completions.
"""

# [STEP 0] Imports
# Annotation: Import standard library modules, Streamlit UI, ChromaDB, PDF/DOCX readers, and the GenAI Lab-compatible chat/embedding clients.
import io
import json
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import httpx
import streamlit as st
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from notifications import send_sms_via_twilio, send_whatsapp_via_twilio
from pypdf import PdfReader
from docx import Document
from user_registry import authenticate_user, find_user, load_users, register_user


# [STEP 1] App constants
# Annotation: Centralize model IDs, API configuration, prompt defaults, storage paths, and basic moderation configuration.
APP_TITLE = "Hackathon RAG Evaluation Engine"
CHROMA_DIR = "./chroma_eval_db"
COLLECTION_NAME = "hackathon_submissions"
GENAILAB_BASE_URL = "https://genailab.tcs.in"
PRIMARY_CHAT_MODEL = "azure_ai/genailab-maas-DeepSeek-V3-0324"
EMBEDDING_MODEL_NAME = "azure/genailab-maas-text-embedding-3-large"
TOP_K = 6
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
MAX_CONTEXT_CHARS = 14000

DEFAULT_RUBRIC = {
    "Design": {
        "weight": 25,
        "criteria": [
            "problem framing and clarity",
            "architecture quality",
            "user experience and usability",
        ],
    },
    "Functionality": {
        "weight": 40,
        "criteria": [
            "core feature completeness",
            "technical correctness",
            "stability and error handling",
        ],
    },
    "Innovation": {
        "weight": 20,
        "criteria": [
            "novelty of idea",
            "smart use of AI",
            "differentiation from basic demos",
        ],
    },
    "Documentation": {
        "weight": 15,
        "criteria": [
            "submission completeness",
            "readme quality",
            "explainability for jurors",
        ],
    },
}

GUARDRAIL_PATTERNS = {
    "abusive_language": [
        r"\bidiot\b",
        r"\bstupid\b",
        r"\bmoron\b",
        r"\bhate you\b",
    ],
    "sexual_content": [
        r"\bexplicit sexual\b",
        r"\bsexual act\b",
        r"\bporn\b",
    ],
    "child_abuse": [
        r"\bchild abuse\b",
        r"\bminor sexual\b",
        r"\bexploit a child\b",
    ],
    "self_harm": [
        r"\bsuicide\b",
        r"\bkill myself\b",
        r"\bself harm\b",
    ],
}

GENERIC_MODEL_ERROR = "Model is not able to identify the response right now."


# [STEP 2] Data classes
# Annotation: Use compact typed structures to keep chunk metadata and chat results clean.
@dataclass
class ChunkRecord:
    doc_id: str
    source_name: str
    chunk_id: str
    text: str
    metadata: dict[str, Any]


# [STEP 3] Streamlit page setup
# Annotation: Configure a single-page app with a clean jury-friendly UI.
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(
    """
    <style>
      :root {
        --bg1: #edf6ff;
        --bg2: #fff5ea;
        --ink: #0f172a;
        --muted: #475569;
        --card: rgba(255,255,255,0.82);
        --line: rgba(15,23,42,0.08);
        --shadow: 0 18px 44px rgba(15,23,42,0.10);
        --accent: #0f766e;
        --accent2: #2563eb;
        --accent3: #f97316;
      }
      .stApp {
        background:
          radial-gradient(900px 500px at 10% 10%, rgba(37,99,235,0.16), transparent 60%),
          radial-gradient(800px 500px at 90% 10%, rgba(249,115,22,0.16), transparent 55%),
          radial-gradient(1000px 700px at 50% 100%, rgba(15,118,110,0.10), transparent 65%),
          linear-gradient(180deg, var(--bg1), var(--bg2));
        background-size: 150% 150%;
        animation: bgFlow 18s ease-in-out infinite;
      }
      .block-container {
        padding-top: 2rem;
        animation: riseIn 500ms ease-out both;
      }
      @keyframes bgFlow {
        0% { background-position: 0% 40%; }
        50% { background-position: 100% 60%; }
        100% { background-position: 0% 40%; }
      }
      @keyframes riseIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
      }
      .hero {
        position: relative;
        overflow: hidden;
        padding: 1.2rem 1.4rem;
        border: 1px solid var(--line);
        border-radius: 24px;
        background: var(--card);
        backdrop-filter: blur(10px);
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
      }
      .hero::before {
        content: "";
        position: absolute;
        width: 220px;
        height: 220px;
        top: -80px;
        right: -50px;
        background: radial-gradient(circle, rgba(37,99,235,0.18), transparent 65%);
        animation: floatGlow 9s ease-in-out infinite;
      }
      .hero::after {
        content: "";
        position: absolute;
        width: 180px;
        height: 180px;
        bottom: -70px;
        left: -30px;
        background: radial-gradient(circle, rgba(249,115,22,0.18), transparent 65%);
        animation: floatGlow 11s ease-in-out infinite reverse;
      }
      @keyframes floatGlow {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-14px); }
        100% { transform: translateY(0px); }
      }
      .hero h2 {
        margin: 0 0 0.4rem 0;
        color: var(--ink);
      }
      .hero p {
        margin: 0;
        color: var(--muted);
      }
      [data-testid="stSidebar"] > div:first-child {
        background: rgba(255,255,255,0.76);
      }
      [data-testid="stFileUploader"], [data-testid="stExpander"], [data-testid="stForm"] {
        border-radius: 20px;
      }
      [data-testid="stFileUploader"] {
        border: 1px dashed rgba(15,118,110,0.30);
        background: linear-gradient(135deg, rgba(255,255,255,0.82), rgba(237,246,255,0.70));
        transition: transform 140ms ease, box-shadow 140ms ease;
      }
      [data-testid="stFileUploader"]:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow);
      }
      .stButton > button {
        border-radius: 14px;
        border: 1px solid rgba(37,99,235,0.16);
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 55%, var(--accent3) 100%);
        color: white;
        font-weight: 600;
        box-shadow: 0 10px 22px rgba(15,23,42,0.10);
        transition: transform 140ms ease, box-shadow 140ms ease;
      }
      .stButton > button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: var(--shadow);
      }
      div[data-baseweb="input"] > div, div[data-baseweb="textarea"] > div, div[data-baseweb="select"] > div {
        border-radius: 14px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title(APP_TITLE)
st.markdown(
    """
    <div class="hero">
      <h2>RAG Solution Architect View</h2>
      <p>Upload hackathon submissions, index them in ChromaDB, ask questions in a chat window, and generate rubric-based evaluation reports with built-in guardrails.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# [STEP 4] Secret loading
# Annotation: Read the GenAI Lab-compatible API key from backend-only sources.
def get_secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value.strip()
    try:
        maybe_secret = st.secrets.get(name)
    except Exception:
        maybe_secret = None
    return str(maybe_secret).strip() if maybe_secret else None


def load_api_key() -> str | None:
    for env_name in ("GENAILAB_API_KEY", "OPENAI_API_KEY"):
        value = get_secret(env_name)
        if value:
            return value
    key_file = Path("API") / "API Key.txt"
    if key_file.exists():
        for line in key_file.read_text(encoding="utf-8").splitlines():
            cleaned = line.strip()
            if cleaned and not cleaned.startswith("#"):
                return cleaned.split()[0]
    return None


api_key = load_api_key()
if not api_key:
    st.error("Backend API key is missing. Set `GENAILAB_API_KEY` or add a key to `API/API Key.txt`.")
    st.stop()


# [STEP 4B] Notification setup
# Annotation: Load the account registry path and Twilio settings for registration and evaluation updates.
users_path = "registered_users.json"
twilio_account_sid = get_secret("TWILIO_ACCOUNT_SID")
twilio_auth_token = get_secret("TWILIO_AUTH_TOKEN")
twilio_from_phone = get_secret("TWILIO_FROM_NUMBER")
twilio_whatsapp_from = get_secret("TWILIO_WHATSAPP_FROM")


# [STEP 5] Client initialization
# Annotation: Initialize the GenAI Lab-compatible HTTP client, chat model, and embedding model.
http_trust_env = str(get_secret("HTTP_TRUST_ENV") or "").strip().lower() in {"1", "true", "yes", "on"}
http_verify = str(get_secret("HTTP_VERIFY") or "").strip().lower() in {"1", "true", "yes", "on"}
http_timeout = httpx.Timeout(60.0, connect=20.0)
shared_http_client = httpx.Client(verify=http_verify, trust_env=http_trust_env, timeout=http_timeout)

chat_model = ChatOpenAI(
    base_url=GENAILAB_BASE_URL,
    model=PRIMARY_CHAT_MODEL,
    api_key=api_key,
    http_client=shared_http_client,
    temperature=0.2,
)
embedding_model = OpenAIEmbeddings(
    base_url=GENAILAB_BASE_URL,
    model=EMBEDDING_MODEL_NAME,
    api_key=api_key,
    http_client=shared_http_client,
    tiktoken_enabled=False,
    check_embedding_ctx_length=False,
)


# [STEP 6] Chroma setup
# Annotation: Persist vectors locally in ChromaDB so the index survives app reloads.
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


# [STEP 7A] Account helpers
# Annotation: Reuse the existing registry and notification helpers to support login, registration, and confirmations.
if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None


def _toast(message: str) -> None:
    try:
        toast = getattr(st, "toast", None)
        if callable(toast):
            toast(message)
    except Exception:
        return


def _get_view() -> str:
    try:
        view = st.query_params.get("view")
    except Exception:
        view = None
    if isinstance(view, list):
        view = view[0] if view else None
    return str(view or "login").strip().lower()


def _set_view(view: str) -> None:
    try:
        st.query_params["view"] = view
    except Exception:
        pass


def _send_registration_confirmation(*, username: str, phone_e164: str, channel: str) -> None:
    if not (twilio_account_sid and twilio_auth_token):
        st.warning("Registered, but Twilio credentials are not set so no confirmation was sent.")
        return

    body = f"Hi {username}, your hackathon evaluator account is registered successfully."
    normalized_channel = str(channel or "").strip().lower()

    if normalized_channel == "sms":
        if not twilio_from_phone:
            st.warning("Registered, but `TWILIO_FROM_NUMBER` is not set so no SMS confirmation was sent.")
            return
        result = send_sms_via_twilio(
            to_phone_e164=phone_e164,
            from_phone_e164=twilio_from_phone,
            body=body,
            account_sid=twilio_account_sid,
            auth_token=twilio_auth_token,
            http_client=shared_http_client,
        )
        if result.ok:
            _toast("SMS confirmation sent.")
        else:
            st.warning(result.message)
        return

    if normalized_channel == "whatsapp":
        if not twilio_whatsapp_from:
            st.warning("Registered, but `TWILIO_WHATSAPP_FROM` is not set so no WhatsApp confirmation was sent.")
            return
        result = send_whatsapp_via_twilio(
            to_phone_e164=phone_e164,
            from_whatsapp=twilio_whatsapp_from,
            body=body,
            account_sid=twilio_account_sid,
            auth_token=twilio_auth_token,
            http_client=shared_http_client,
        )
        if result.ok:
            _toast("WhatsApp confirmation sent.")
        else:
            st.warning(result.message)


def _send_report_notification(*, username: str, phone_e164: str, channel: str) -> None:
    if not (twilio_account_sid and twilio_auth_token):
        return

    body = f"Hi {username}, your hackathon evaluation report is ready in the app."
    normalized_channel = str(channel or "").strip().lower()

    if normalized_channel == "sms" and twilio_from_phone:
        send_sms_via_twilio(
            to_phone_e164=phone_e164,
            from_phone_e164=twilio_from_phone,
            body=body,
            account_sid=twilio_account_sid,
            auth_token=twilio_auth_token,
            http_client=shared_http_client,
        )
    elif normalized_channel == "whatsapp" and twilio_whatsapp_from:
        send_whatsapp_via_twilio(
            to_phone_e164=phone_e164,
            from_whatsapp=twilio_whatsapp_from,
            body=body,
            account_sid=twilio_account_sid,
            auth_token=twilio_auth_token,
            http_client=shared_http_client,
        )


# [STEP 7] Guardrail helpers
# Annotation: Apply lightweight input/output controls for harmful content categories and PII masking.
def mask_pii(text: str) -> str:
    text = re.sub(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[EMAIL_MASKED]", text, flags=re.I)
    text = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4})\b", "[PHONE_MASKED]", text)
    text = re.sub(r"\b\d{12}\b", "[ID_MASKED]", text)
    text = re.sub(r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b", "[CARD_MASKED]", text)
    return text


def detect_guardrail_violation(text: str) -> str | None:
    probe = text.lower()
    for category, patterns in GUARDRAIL_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, probe, flags=re.I):
                return category
    return None


def apply_guardrails(text: str) -> tuple[bool, str]:
    violation = detect_guardrail_violation(text)
    if violation:
        return False, f"Request blocked by guardrails: `{violation}`."
    return True, mask_pii(text)


# [STEP 8] File readers
# Annotation: Support common hackathon submission formats and ZIP bundles in a single uploader.
def read_pdf_bytes(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def read_docx_bytes(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)


def read_text_bytes(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")


def extract_zip_bytes(data: bytes) -> list[tuple[str, str]]:
    extracted: list[tuple[str, str]] = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            suffix = Path(member.filename).suffix.lower()
            raw = archive.read(member.filename)
            content = parse_uploaded_file(member.filename, raw)
            if content.strip():
                extracted.append((member.filename, content))
    return extracted


def parse_uploaded_file(file_name: str, data: bytes) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".pdf":
        return read_pdf_bytes(data)
    if suffix == ".docx":
        return read_docx_bytes(data)
    if suffix in {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv"}:
        return read_text_bytes(data)
    return ""


# [STEP 9] Chunking helper
# Annotation: Use recursive chunking so mixed technical documents are split semantically before indexing.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""],
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    return embedding_model.embed_documents(texts)


def build_chunk_records(source_name: str, text: str, doc_id: str) -> list[ChunkRecord]:
    masked_text = mask_pii(text)
    chunks = splitter.split_text(masked_text)
    records: list[ChunkRecord] = []
    for idx, chunk in enumerate(chunks):
        records.append(
            ChunkRecord(
                doc_id=doc_id,
                source_name=source_name,
                chunk_id=f"{doc_id}-chunk-{idx}",
                text=chunk,
                metadata={"doc_id": doc_id, "source_name": source_name, "chunk_index": idx},
            )
        )
    return records


# [STEP 10] Indexing helper
# Annotation: Upsert chunk embeddings into ChromaDB so multiple team artifacts can be searched together.
def index_submission_documents(uploaded_files: list[Any]) -> dict[str, Any]:
    all_records: list[ChunkRecord] = []
    indexed_sources: list[str] = []

    for uploaded in uploaded_files:
        file_bytes = uploaded.getvalue()
        suffix = Path(uploaded.name).suffix.lower()

        if suffix == ".zip":
            for inner_name, inner_text in extract_zip_bytes(file_bytes):
                doc_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", inner_name).strip("-").lower() or "document"
                all_records.extend(build_chunk_records(inner_name, inner_text, doc_id))
                indexed_sources.append(inner_name)
        else:
            parsed = parse_uploaded_file(uploaded.name, file_bytes)
            if parsed.strip():
                doc_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", uploaded.name).strip("-").lower() or "document"
                all_records.extend(build_chunk_records(uploaded.name, parsed, doc_id))
                indexed_sources.append(uploaded.name)

    if not all_records:
        raise ValueError("No readable content found in the uploaded files.")

    ids = [record.chunk_id for record in all_records]
    docs = [record.text for record in all_records]
    metas = [record.metadata for record in all_records]
    embeds = embed_texts(docs)

    existing = collection.get(where={"doc_id": {"$ne": ""}}, include=[])
    if existing.get("ids"):
        collection.delete(ids=existing["ids"])

    collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)

    return {
        "indexed_sources": indexed_sources,
        "chunk_count": len(all_records),
    }


# [STEP 11] Retrieval helper
# Annotation: Retrieve the most relevant chunks using the same embedding model used at indexing time.
def retrieve_context(question: str, top_k: int = TOP_K) -> dict[str, Any]:
    query_embedding = embedding_model.embed_query(mask_pii(question))
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    documents = result["documents"][0] if result.get("documents") else []
    metadatas = result["metadatas"][0] if result.get("metadatas") else []
    return {"documents": documents, "metadatas": metadatas}


def build_context_block(retrieval: dict[str, Any]) -> str:
    sections: list[str] = []
    for idx, (doc, meta) in enumerate(zip(retrieval["documents"], retrieval["metadatas"]), start=1):
        source = meta.get("source_name", f"source-{idx}") if meta else f"source-{idx}"
        sections.append(f"[Source {idx}: {source}]\n{doc}")
    return "\n\n".join(sections)[:MAX_CONTEXT_CHARS]


# [STEP 12] Chat completion helper
# Annotation: Use the GenAI Lab-compatible chat model over retrieved context for Q&A and evaluation.
def run_chat_completion(system_prompt: str, user_prompt: str) -> str:
    response = chat_model.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return getattr(response, "content", None) or GENERIC_MODEL_ERROR


# [STEP 13] Evaluation prompt helper
# Annotation: Build a rubric-aware scoring instruction that turns retrieved evidence into a consistent report.
def build_evaluation_prompt(context_block: str, rubric_text: str, team_name: str) -> tuple[str, str]:
    system_prompt = (
        "You are an AI hackathon evaluation engine. "
        "Score submissions fairly, stay evidence-based, avoid hallucinations, and cite only the provided context."
    )
    user_prompt = f"""
Team name: {team_name}

Rubric:
{rubric_text}

Retrieved submission context:
{context_block}

Tasks:
1. Score each rubric category out of its weight.
2. Explain the evidence for each score.
3. Highlight strengths, gaps, and risk flags.
4. Provide an overall score out of 100.
5. Return a final recommendation: Strong Accept / Accept / Borderline / Needs Improvement.
6. Mention if documentation is missing for any score.

Return your answer in markdown with these sections:
- Executive Summary
- Rubric Scores
- Evidence Notes
- Risks and Missing Items
- Final Recommendation
""".strip()
    return system_prompt, user_prompt


# [STEP 14] Session state
# Annotation: Keep chat history and index status stable across Streamlit reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "indexed_sources" not in st.session_state:
    st.session_state.indexed_sources = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# [STEP 15] Authentication flow
# Annotation: Require login or registration before the evaluator workspace is shown.
view = _get_view()
if st.session_state.logged_in_user is None and view != "register":
    st.subheader("Login")
    login_username = st.text_input("Username")
    login_password = st.text_input("Password", type="password")
    if st.button("Login", use_container_width=True):
        user = authenticate_user(path=users_path, username=login_username, password=login_password)
        if user is None:
            existing = find_user(load_users(users_path), login_username)
            if existing and not existing.has_password:
                st.session_state.logged_in_user = existing
                _set_view("app")
                st.rerun()
            else:
                st.error("Invalid username/password.")
        else:
            st.session_state.logged_in_user = user
            _set_view("app")
            st.rerun()
    st.markdown("Not registered? [Create an account](?view=register)")
    st.stop()

if st.session_state.logged_in_user is None and view == "register":
    st.subheader("Register")
    reg_username = st.text_input("Username")
    reg_phone = st.text_input("Phone (E.164)", placeholder="+91XXXXXXXXXX")
    reg_password = st.text_input("Password", type="password")
    reg_password_confirm = st.text_input("Confirm password", type="password")
    reg_channel = st.selectbox("Send confirmation via", options=["SMS", "WhatsApp"])
    if st.button("Register", use_container_width=True):
        if not reg_username.strip() or not reg_phone.strip() or not reg_password:
            st.error("Enter username, phone, and password.")
        elif reg_password != reg_password_confirm:
            st.error("Passwords do not match.")
        else:
            try:
                user = register_user(
                    path=users_path,
                    username=reg_username,
                    password=reg_password,
                    phone_e164=reg_phone,
                    notify_channel=reg_channel.lower(),
                )
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
            else:
                _send_registration_confirmation(
                    username=user.name,
                    phone_e164=user.phone_e164,
                    channel=user.notify_channel or reg_channel.lower(),
                )
                st.session_state.logged_in_user = user
                _set_view("app")
                st.rerun()
    st.markdown("Already registered? [Go to login](?view=login)")
    st.stop()


# [STEP 16] Sidebar controls
# Annotation: Show account details, notification settings, and evaluator controls for the logged-in user.
st.sidebar.header("Account")
st.sidebar.write(f"Logged in as: {st.session_state.logged_in_user.name}")
if st.sidebar.button("Logout"):
    st.session_state.logged_in_user = None
    _set_view("login")
    st.rerun()
notify_on_report = st.sidebar.checkbox("Notify me when report is ready", value=True)
user_channel = (st.session_state.logged_in_user.notify_channel or "sms").strip().lower()
st.sidebar.caption(f"Notification channel: {user_channel}")

st.sidebar.header("Evaluation Controls")
team_name = st.sidebar.text_input("Team Name", value="Team AI Friday")
rubric_text = st.sidebar.text_area(
    "Scoring Rubric (JSON or notes)",
    value=json.dumps(DEFAULT_RUBRIC, indent=2),
    height=320,
)
st.sidebar.caption(
    "Current stack: ChromaDB + Recursive Chunking + GenAI Lab embeddings + GenAI Lab chat completions."
)


# [STEP 17] Upload section
# Annotation: Accept mixed submission artifacts such as README files, Python files, PDFs, DOCX, and ZIP bundles.
uploaded_files = st.file_uploader(
    "Upload team submission files",
    type=["pdf", "txt", "md", "py", "json", "docx", "zip", "yaml", "yml", "csv"],
    accept_multiple_files=True,
)

col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("Index Submission", use_container_width=True):
        if not uploaded_files:
            st.error("Upload at least one file before indexing.")
        else:
            try:
                stats = index_submission_documents(uploaded_files)
                st.session_state.index_ready = True
                st.session_state.indexed_sources = stats["indexed_sources"]
                st.session_state.chunk_count = stats["chunk_count"]
                st.success(f"Indexed {stats['chunk_count']} chunks from {len(stats['indexed_sources'])} file(s).")
            except Exception:
                st.error(GENERIC_MODEL_ERROR)

with col_b:
    if st.button("Generate Evaluation Report", use_container_width=True):
        if not st.session_state.index_ready:
            st.error("Index the submission first.")
        else:
            try:
                retrieval = retrieve_context(
                    "Evaluate this hackathon submission against the rubric, architecture, implementation quality, and innovation."
                )
                context_block = build_context_block(retrieval)
                system_prompt, user_prompt = build_evaluation_prompt(context_block, rubric_text, team_name)
                report = run_chat_completion(system_prompt, user_prompt)
                allowed, safe_report = apply_guardrails(report)
                if not allowed:
                    st.error(safe_report)
                else:
                    st.subheader("Evaluation Report")
                    st.markdown(mask_pii(safe_report))
                    if notify_on_report:
                        _send_report_notification(
                            username=st.session_state.logged_in_user.name,
                            phone_e164=st.session_state.logged_in_user.phone_e164,
                            channel=user_channel,
                        )
            except Exception:
                st.error(GENERIC_MODEL_ERROR)


# [STEP 18] Index status section
# Annotation: Show which files were indexed so jurors know what evidence base the engine used.
if st.session_state.index_ready:
    st.info(
        f"Indexed sources: {len(st.session_state.indexed_sources)} | Total chunks: {st.session_state.chunk_count}"
    )
    with st.expander("Indexed Files", expanded=False):
        for name in st.session_state.indexed_sources:
            st.write(f"- {name}")


# [STEP 19] Chat window
# Annotation: Build a lightweight Streamlit chat UI to query indexed submission documents.
st.subheader("Ask Questions About the Submission")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask about architecture, functionality, innovation, risks, missing docs, or rubric fit")
if question:
    if not st.session_state.index_ready:
        st.error("Index the submission before asking questions.")
    else:
        allowed, safe_question = apply_guardrails(question)
        if not allowed:
            st.error(safe_question)
        else:
            st.session_state.messages.append({"role": "user", "content": safe_question})
            with st.chat_message("user"):
                st.markdown(safe_question)

            try:
                retrieval = retrieve_context(safe_question)
                context_block = build_context_block(retrieval)
                system_prompt = (
                    "You are a RAG assistant for hackathon jury support. "
                    "Answer only from the retrieved submission context. "
                    "If the context is insufficient, say so clearly."
                )
                user_prompt = f"""
Question:
{safe_question}

Retrieved context:
{context_block}

Answer in markdown and include a short section called `Evidence Used`.
""".strip()
                answer = run_chat_completion(system_prompt, user_prompt)
                allowed, safe_answer = apply_guardrails(answer)
                final_answer = safe_answer if allowed else "Response blocked by guardrails."
            except Exception:
                final_answer = GENERIC_MODEL_ERROR

            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            with st.chat_message("assistant"):
                st.markdown(final_answer)


# [STEP 20] Packages section
# Annotation: Print the required Python packages so the project can be installed quickly during the hackathon.
with st.expander("Required Packages", expanded=False):
    st.code(
        "\n".join(
            [
                "streamlit",
                "chromadb",
                "httpx",
                "langchain-openai",
                "langchain-text-splitters",
                "pypdf",
                "python-docx",
            ]
        ),
        language="text",
    )


# [STEP 21] Run instructions
# Annotation: Add practical startup notes so the app is easy to demo.
with st.expander("Run Instructions", expanded=False):
    st.markdown(
        """
1. Set `GENAILAB_API_KEY` in your environment or place it in `API/API Key.txt`.
2. Install the packages listed above.
3. Run: `streamlit run hackathon_rag_evaluator.py`
4. Upload team files, click `Index Submission`, then use `Generate Evaluation Report` or the chat box.
        """
    )
