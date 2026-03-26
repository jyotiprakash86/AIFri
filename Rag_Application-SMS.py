# [STEP 0] Imports
# Annotation: Load standard libs, Streamlit UI, LangChain components, and local helpers.
import os
import tempfile
import hashlib

import httpx
import streamlit as st
from openai import PermissionDeniedError
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from notifications import send_sms_via_twilio, send_whatsapp_via_twilio
from pdf_text import extract_text
from user_registry import authenticate_user, find_user, load_users, register_user


PRIMARY_MODEL = "azure_ai/genailab-maas-DeepSeek-V3-0324"
DEFAULT_VALIDATOR_MODEL = "azure_ai/genailab-maas-Meta-Llama-3.1-70B-Instruct"
CONTENT_FILTER_REPLACEMENTS = {
    "attack": "security incident",
    "attacks": "security incidents",
    "attacked": "affected",
    "attacking": "targeting",
}
GENERIC_MODEL_ERROR = "Model is not able to identify the response right now."


# [STEP 1] Secrets helper
# Annotation: Read values from environment variables or Streamlit secrets.
def _get_secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value.strip()

    try:
        from_streamlit = st.secrets.get(name)
    except Exception:
        from_streamlit = None

    if from_streamlit:
        return str(from_streamlit).strip()

    return None


# [STEP 1B] Bool secret helper
# Annotation: Parse booleans from environment variables or Streamlit secrets.
def _get_bool_secret(name: str, default: bool) -> bool:
    raw = _get_secret(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


# [STEP 2] API key loader
# Annotation: Resolve the LLM/embeddings API key from env/secrets/file (optional).
def _load_api_key_optional() -> str | None:
    for env_name in ("GENAILAB_API_KEY", "OPENAI_API_KEY"):
        value = _get_secret(env_name)
        if value:
            return value

    key_file = os.path.join("API", "API Key.txt")
    if os.path.exists(key_file):
        with open(key_file, "r", encoding="utf-8") as file_handle:
            raw = file_handle.read()
            # Support "key file" that contains comments/examples.
            for line in raw.splitlines():
                cleaned = line.strip()
                if not cleaned or cleaned.startswith("#"):
                    continue
                # If someone pasted "KEY # comment", keep just the key token.
                token = cleaned.split()[0].strip()
                if token and not token.startswith("#"):
                    return token
            return None

    return None


def _build_chat_model(*, api_key: str, http_client: httpx.Client, model_name: str, temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        base_url="https://genailab.tcs.in",
        model=model_name,
        api_key=api_key,
        http_client=http_client,
        temperature=temperature,
    )


def _prepare_validation_context(source_documents: list, limit: int = 3, snippet_chars: int = 800) -> str:
    context_parts = []
    for idx, doc in enumerate(source_documents[:limit], start=1):
        text = (getattr(doc, "page_content", "") or "").strip()
        if text:
            context_parts.append(f"Context chunk {idx}:\n{text[:snippet_chars]}")
    return "\n\n".join(context_parts) if context_parts else "No supporting context available."


def _collection_name_for_pdf(pdf_digest: str) -> str:
    return f"pdf_{pdf_digest[:16]}"


def _sanitize_text_for_llm(text: str) -> str:
    sanitized = text
    for original, replacement in CONTENT_FILTER_REPLACEMENTS.items():
        sanitized = sanitized.replace(original, replacement)
        sanitized = sanitized.replace(original.capitalize(), replacement.capitalize())
        sanitized = sanitized.replace(original.upper(), replacement.upper())
    return sanitized


def _safe_rag_invoke(rag_chain: RetrievalQA, query: str) -> dict | None:
    try:
        return rag_chain.invoke({"query": _sanitize_text_for_llm(query)})
    except PermissionDeniedError:
        st.error(GENERIC_MODEL_ERROR)
        return None
    except Exception:  # noqa: BLE001
        st.error(GENERIC_MODEL_ERROR)
        return None


def _validate_with_second_model(
    *,
    validator: ChatOpenAI,
    task_label: str,
    prompt_or_question: str,
    answer: str,
    source_documents: list,
) -> str:
    validation_context = _prepare_validation_context(source_documents)
    validation_prompt = f"""
You are validating a {task_label} generated from a PDF question-answering system.

User request:
{prompt_or_question}

Generated answer:
{answer}

Supporting document context:
{validation_context}

Review the answer using only the supporting document context.
Respond in this exact structure:
Verdict: Correct / Partially Correct / Not Supported
Reason: Short explanation
Improved Answer: A corrected or improved answer grounded in the context
""".strip()
    try:
        return validator.invoke(_sanitize_text_for_llm(validation_prompt)).content
    except PermissionDeniedError:
        return (
            "Verdict: Not Supported\n"
            f"Reason: {GENERIC_MODEL_ERROR}\n"
            "Improved Answer: Validation could not be completed for this document."
        )
    except Exception:  # noqa: BLE001
        return (
            "Verdict: Not Supported\n"
            f"Reason: {GENERIC_MODEL_ERROR}\n"
            "Improved Answer: Validation could not be completed for this document."
        )


# [STEP 3] Streamlit page setup
# Annotation: Configure the page and main title.
st.set_page_config(page_title="RAG PDF Summarizer")
st.markdown(
    """
    <style>
      :root {
        --bg: #eef4ff;
        --bg2: #fff8ef;
        --text: #0f172a;
        --muted: rgba(15, 23, 42, 0.60);
        --card: rgba(255, 255, 255, 0.72);
        --card-solid: #ffffff;
        --card-brd: rgba(15, 23, 42, 0.10);
        --shadow: 0 22px 56px rgba(15, 23, 42, 0.12);
        --shadow-sm: 0 12px 28px rgba(15, 23, 42, 0.10);
        --accent: #0f766e;
        --accent2: #f97316;
        --accent3: #1d4ed8;
      }

      html, body, [class*="css"] {
        font-family: "Trebuchet MS", "Segoe UI", sans-serif;
        color: var(--text);
      }

      .stApp {
        background:
          radial-gradient(850px 650px at 12% 18%, rgba(29, 78, 216, 0.18), transparent 58%),
          radial-gradient(780px 560px at 88% 12%, rgba(249, 115, 22, 0.16), transparent 55%),
          radial-gradient(920px 720px at 45% 90%, rgba(15, 118, 110, 0.14), transparent 62%),
          linear-gradient(180deg, var(--bg2) 0%, var(--bg) 58%, #f4f7fb 100%);
        background-size: 155% 155%;
        animation: bgDrift 20s ease-in-out infinite;
      }

      @keyframes bgDrift {
        0% { background-position: 0% 35%; }
        50% { background-position: 100% 65%; }
        100% { background-position: 0% 35%; }
      }

      .block-container {
        padding-top: 2.0rem;
        animation: fadeInUp 520ms ease-out both;
      }

      @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
      }

      h1, h2, h3 {
        letter-spacing: -0.02em;
      }

      h1 {
        background: linear-gradient(90deg, #0f172a 0%, #0f766e 42%, #f97316 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin-bottom: 0.4rem;
      }

      .hero-shell {
        position: relative;
        overflow: hidden;
        padding: 1.35rem 1.45rem;
        margin: 0.4rem 0 1.2rem 0;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.82), rgba(255, 250, 245, 0.64));
        border: 1px solid rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(12px);
        box-shadow: var(--shadow);
      }

      .hero-shell::before {
        content: "";
        position: absolute;
        inset: -40% auto auto -10%;
        width: 240px;
        height: 240px;
        background: radial-gradient(circle, rgba(29, 78, 216, 0.18), transparent 66%);
        animation: floatOrb 9s ease-in-out infinite;
      }

      .hero-shell::after {
        content: "";
        position: absolute;
        right: -30px;
        bottom: -60px;
        width: 210px;
        height: 210px;
        background: radial-gradient(circle, rgba(249, 115, 22, 0.18), transparent 64%);
        animation: floatOrb 11s ease-in-out infinite reverse;
      }

      @keyframes floatOrb {
        0% { transform: translateY(0px) scale(1); }
        50% { transform: translateY(-18px) scale(1.05); }
        100% { transform: translateY(0px) scale(1); }
      }

      .hero-title {
        position: relative;
        font-size: 1.08rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
      }

      .hero-copy {
        position: relative;
        margin: 0;
        max-width: 760px;
        color: var(--muted);
        line-height: 1.55;
      }

      p, li, label, .stMarkdown {
        color: var(--text);
      }

      /* Premium card look for key containers */
      [data-testid="stForm"], [data-testid="stExpander"], [data-testid="stFileUploader"], section.main > div {
        border-radius: 18px;
      }

      [data-testid="stAlert"] {
        border-radius: 16px;
        border: 1px solid var(--card-brd);
        background: rgba(255, 255, 255, 0.72);
        backdrop-filter: blur(12px);
        box-shadow: var(--shadow-sm);
        animation: fadeInUp 380ms ease-out both;
      }

      /* Sidebar: clean, premium, slightly glassy */
      [data-testid="stSidebar"] > div:first-child {
        background: rgba(255, 255, 255, 0.70);
        backdrop-filter: blur(14px);
        border-right: 1px solid rgba(15, 23, 42, 0.10);
      }

      [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
      [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: var(--text);
      }

      /* Inputs: softer radius + subtle focus ring */
      input, textarea {
        border-radius: 14px !important;
      }

      div[data-baseweb="input"] > div, div[data-baseweb="textarea"] > div, div[data-baseweb="select"] > div {
        border-radius: 14px !important;
        border: 1px solid rgba(15, 23, 42, 0.12);
        background: rgba(255, 255, 255, 0.85);
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        transition: box-shadow 120ms ease, border-color 120ms ease;
      }

      div[data-baseweb="input"] > div:focus-within,
      div[data-baseweb="textarea"] > div:focus-within,
      div[data-baseweb="select"] > div:focus-within {
        border-color: rgba(15, 118, 110, 0.40);
        box-shadow: 0 10px 24px rgba(15, 118, 110, 0.14);
      }

      /* Buttons: premium gradient */
      .stButton > button {
        border-radius: 14px;
        border: 1px solid rgba(15, 118, 110, 0.20);
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.96) 0%, rgba(29, 78, 216, 0.92) 55%, rgba(249, 115, 22, 0.92) 100%);
        color: #ffffff;
        font-weight: 600;
        letter-spacing: 0.01em;
        box-shadow: var(--shadow-sm);
        transition: transform 150ms ease, box-shadow 150ms ease, filter 150ms ease;
      }

      .stButton > button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: var(--shadow);
        filter: saturate(1.05);
      }

      .stButton > button:active {
        transform: translateY(0) scale(0.99);
      }

      [data-testid="stFileUploader"] {
        border: 1px dashed rgba(15, 118, 110, 0.30);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.82), rgba(238, 244, 255, 0.70));
        box-shadow: var(--shadow-sm);
        transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
      }

      [data-testid="stFileUploader"]:hover {
        transform: translateY(-2px);
        border-color: rgba(249, 115, 22, 0.45);
        box-shadow: var(--shadow);
      }

      [data-testid="stRadio"] > div {
        gap: 0.6rem;
      }

      [data-testid="stRadio"] label {
        border-radius: 999px;
        padding: 0.55rem 0.95rem;
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: var(--shadow-sm);
        transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
      }

      [data-testid="stRadio"] label:hover {
        transform: translateY(-1px);
        border-color: rgba(15, 118, 110, 0.28);
        box-shadow: var(--shadow);
      }

      [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.68);
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: var(--shadow-sm);
      }

      /* Links */
      a, a:visited {
        color: var(--accent);
      }

      /* Make captions muted */
      .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("RAG-powered PDF Summarizer")
st.markdown(
    """
    <div class="hero-shell">
      <div class="hero-title">Upload once, then choose your flow</div>
      <p class="hero-copy">
        Switch between document summarization and question answering, then validate the generated response with a second model for extra confidence.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# [STEP 4] API key gating
# Annotation: Read the key only from backend sources so it is never requested in the UI.
api_key = _load_api_key_optional()
if not api_key:
    st.error(
        "API key not configured in backend. Set `GENAILAB_API_KEY` or `OPENAI_API_KEY`, "
        "or add the key to `API/API Key.txt`."
    )
    st.stop()

# [STEP 5] HTTP client
# Annotation: Shared HTTPX client used by the LLM and (optionally) Twilio requests.
# In many enterprise networks, reaching internal gateways requires proxy env vars.
# `trust_env=True` allows HTTP(S)_PROXY / NO_PROXY to work.
http_trust_env = _get_bool_secret("HTTP_TRUST_ENV", default=False)
http_verify = _get_bool_secret("HTTP_VERIFY", default=False)
http_timeout = httpx.Timeout(60.0, connect=20.0)
client = httpx.Client(verify=http_verify, trust_env=http_trust_env, timeout=http_timeout)

# [STEP 6] Chat model
# Annotation: Configure the chat LLM used by RetrievalQA.
llm = _build_chat_model(api_key=api_key, http_client=client, model_name=PRIMARY_MODEL, temperature=0)
validator_model_name = (
    _get_secret("VALIDATOR_MODEL")
    or st.sidebar.text_input("Validator model", value=DEFAULT_VALIDATOR_MODEL)
).strip()
validator_llm = _build_chat_model(
    api_key=api_key,
    http_client=client,
    model_name=validator_model_name or DEFAULT_VALIDATOR_MODEL,
    temperature=0,
)

# [STEP 7] Embeddings model
# Annotation: Configure embeddings used to build the vector database.
embedding_model = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key=api_key,
    http_client=client,
    tiktoken_enabled=False,
    check_embedding_ctx_length=False,
)

# [STEP 8] Twilio configuration
# Annotation: Read Twilio credentials required to send SMS/WhatsApp messages.
users_path = "registered_users.json"
twilio_account_sid = _get_secret("TWILIO_ACCOUNT_SID")
twilio_auth_token = _get_secret("TWILIO_AUTH_TOKEN")
twilio_from_phone = _get_secret("TWILIO_FROM_NUMBER")
twilio_whatsapp_from = _get_secret("TWILIO_WHATSAPP_FROM")

# [STEP 9] Simple login/register flow
# Annotation: Show a login page first; link to registration if the user isn't registered.
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

    view = str(view or "").strip().lower()
    return view or "login"


def _set_view(view: str) -> None:
    try:
        st.query_params["view"] = view
    except Exception:
        pass


def _send_registration_confirmation(*, username: str, phone_e164: str, channel: str) -> None:
    if not (twilio_account_sid and twilio_auth_token):
        st.warning("Registered, but Twilio credentials are not set so no confirmation was sent.")
        return

    body = f"Hi {username}, your account is registered successfully."
    channel = str(channel or "").strip().lower()

    if channel == "sms":
        if not twilio_from_phone:
            st.warning("Registered, but `TWILIO_FROM_NUMBER` is not set so no SMS confirmation was sent.")
            return
        result = send_sms_via_twilio(
            to_phone_e164=phone_e164,
            from_phone_e164=twilio_from_phone,
            body=body,
            account_sid=twilio_account_sid,
            auth_token=twilio_auth_token,
            http_client=client,
        )
        if not result.ok:
            st.warning(result.message)
        else:
            _toast("SMS confirmation sent.")
        return

    if channel == "whatsapp":
        if not twilio_whatsapp_from:
            st.warning("Registered, but `TWILIO_WHATSAPP_FROM` is not set so no WhatsApp confirmation was sent.")
            return
        result = send_whatsapp_via_twilio(
            to_phone_e164=phone_e164,
            from_whatsapp=twilio_whatsapp_from,
            body=body,
            account_sid=twilio_account_sid,
            auth_token=twilio_auth_token,
            http_client=client,
        )
        if not result.ok:
            st.warning(result.message)
        else:
            _toast("WhatsApp confirmation sent.")
        return

    st.warning("Registered, but notification channel is not set (choose SMS or WhatsApp).")


view = _get_view()
if st.session_state.logged_in_user is None and view != "register":
    st.subheader("Login")
    login_username = st.text_input("Username")
    login_password = st.text_input("Password", type="password")
    if st.button("Login"):
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

    if st.button("Register"):
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

# [STEP 10] Logged-in sidebar controls
st.sidebar.header("Account")
st.sidebar.write(f"Logged in as: {st.session_state.logged_in_user.name}")
if st.sidebar.button("Logout"):
    st.session_state.logged_in_user = None
    _set_view("login")
    st.rerun()

# [STEP 11] Notification preference
notify_on_done = st.sidebar.checkbox("Notify me when summary is ready", value=True)
user_channel = (st.session_state.logged_in_user.notify_channel or "sms").strip().lower()
st.sidebar.caption(f"Confirmation channel: {user_channel}")

# [STEP 14] PDF upload
# Annotation: User uploads a PDF to summarize.
upload_file = st.file_uploader("Upload a PDF", type="pdf")

if upload_file:
    if "qna_history" not in st.session_state:
        st.session_state.qna_history = []
    if "validation_cache" not in st.session_state:
        st.session_state.validation_cache = {}

    # [STEP 15] Save uploaded PDF
    # Annotation: Persist the uploaded PDF to a temporary file for parsing.
    pdf_bytes = upload_file.getvalue()
    pdf_digest = hashlib.sha256(pdf_bytes).hexdigest()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_bytes)
        temp_file_path = temp_file.name

    needs_reindex = st.session_state.get("pdf_digest") != pdf_digest
    if needs_reindex:
        st.session_state.pdf_digest = pdf_digest
        st.session_state.summary_result = None
        st.session_state.summary_validation = None
        st.session_state.qna_history = []
        st.session_state.validation_cache = {}

        # [STEP 16] Extract PDF text
        # Annotation: Convert PDF contents into raw text for chunking.
        raw_text = extract_text(temp_file_path)
        if not raw_text.strip():
            st.error("No readable text was found in this PDF, so the app cannot summarize it.")
            st.stop()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        # [STEP 17] Split into chunks
        # Annotation: Break long text into retrieval-friendly chunks.
        chunks = text_splitter.split_text(_sanitize_text_for_llm(raw_text))
        if not chunks:
            st.error("The uploaded PDF did not produce text chunks for summarization.")
            st.stop()

        # [STEP 18] Build vector index
        # Annotation: Embed chunks and store them in Chroma for similarity search.
        with st.spinner("Indexing document..."):
            try:
                collection_name = _collection_name_for_pdf(pdf_digest)
                vectordb = Chroma.from_texts(
                    chunks,
                    embedding_model,
                    collection_name=collection_name,
                    persist_directory="./chroma_index",
                )
                vectordb.persist()
            except Exception:  # noqa: BLE001
                st.error(GENERIC_MODEL_ERROR)
                st.stop()

        # [STEP 19] Create retriever
        # Annotation: Configure similarity search over the vector index.
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # [STEP 20] Create RAG chain
        # Annotation: RetrievalQA combines retriever + LLM to answer based on document context.
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
        )

        st.session_state.vectordb = vectordb
        st.session_state.retriever = retriever
        st.session_state.rag_chain = rag_chain
        st.session_state.current_upload_name = upload_file.name
        st.session_state.current_chunk_count = len(chunks)
    else:
        rag_chain = st.session_state.get("rag_chain")

    st.subheader("Choose what you want to do")
    st.caption(
        f"Current file: {st.session_state.get('current_upload_name', upload_file.name)} "
        f"| Chunks indexed: {st.session_state.get('current_chunk_count', 0)}"
    )
    app_mode = st.radio(
        "After uploading the PDF, select one option",
        options=["Summarizer", "Q&A"],
        horizontal=True,
    )

    if app_mode == "Summarizer":
        # [STEP 21] Run summarization prompt
        # Annotation: Ask the LLM to summarize using retrieved chunks.
        if st.session_state.get("summary_result") is None:
            summary_prompt = (
                f"Summarize the uploaded PDF named '{st.session_state.get('current_upload_name', upload_file.name)}'. "
                "Include the main topics, important points, and a concise conclusion based only on this document. "
                "Use neutral, factual wording."
            )
            with st.spinner("Running RAG summarization..."):
                st.session_state.summary_result = _safe_rag_invoke(rag_chain, summary_prompt)

        # [STEP 22] Display results
        # Annotation: Show the generated summary in the app.
        st.subheader("Summary")
        if st.session_state.get("summary_result"):
            st.write(st.session_state.summary_result["result"])
        if st.button("Validate summary with second model"):
            with st.spinner("Validating summary..."):
                try:
                    if st.session_state.get("summary_result"):
                        st.session_state.summary_validation = _validate_with_second_model(
                            validator=validator_llm,
                            task_label="summary",
                            prompt_or_question="Summarize the uploaded PDF.",
                            answer=st.session_state.summary_result.get("result", ""),
                            source_documents=st.session_state.summary_result.get("source_documents") or [],
                        )
                except Exception:  # noqa: BLE001
                    st.error(GENERIC_MODEL_ERROR)

        if st.session_state.get("summary_validation"):
            st.markdown("**Validation by second model**")
            st.write(st.session_state.summary_validation)

    # [STEP 23] Send confirmation (optional)
    # Annotation: Notify the logged-in user that summarization is complete via their chosen channel.
    if notify_on_done and st.session_state.get("summary_notified_digest") != pdf_digest:
        target_phone = st.session_state.logged_in_user.phone_e164
        if user_channel == "sms":
            if not (twilio_account_sid and twilio_auth_token and twilio_from_phone):
                st.error(
                    "SMS not configured. Set `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, and `TWILIO_FROM_NUMBER`."
                )
            else:
                notification = send_sms_via_twilio(
                    to_phone_e164=target_phone,
                    from_phone_e164=twilio_from_phone,
                    body="Your PDF summarization is complete. Open the app to view the summary.",
                    account_sid=twilio_account_sid,
                    auth_token=twilio_auth_token,
                    http_client=client,
                )
                if notification.ok:
                    st.success("SMS notification sent.")
                    _toast("SMS notification sent.")
                    st.session_state.summary_notified_digest = pdf_digest
                else:
                    st.error(notification.message)
        elif user_channel == "whatsapp":
            if not (twilio_account_sid and twilio_auth_token and twilio_whatsapp_from):
                st.error(
                    "WhatsApp not configured. Set `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, and `TWILIO_WHATSAPP_FROM`."
                )
            else:
                notification = send_whatsapp_via_twilio(
                    to_phone_e164=target_phone,
                    from_whatsapp=twilio_whatsapp_from,
                    body="Your PDF summarization is complete. Open the app to view the summary.",
                    account_sid=twilio_account_sid,
                    auth_token=twilio_auth_token,
                    http_client=client,
                )
                if notification.ok:
                    st.success("WhatsApp notification sent.")
                    _toast("WhatsApp notification sent.")
                    st.session_state.summary_notified_digest = pdf_digest
                else:
                    st.error(notification.message)
        else:
            st.warning(f"Unsupported confirmation channel: {user_channel}")

    if app_mode == "Q&A":
        st.subheader("Ask questions about this PDF")
        st.caption("Answers are generated using retrieval from your uploaded document.")

        show_sources = st.checkbox("Show sources", value=True)
        with st.form("qna_form", clear_on_submit=True):
            question = st.text_input("Your question", placeholder="e.g., What is the main objective of the document?")
            submitted = st.form_submit_button("Ask")

        if submitted:
            q = (question or "").strip()
            if not q:
                st.error("Please enter a question.")
            else:
                with st.spinner("Answering..."):
                    qna_result = _safe_rag_invoke(rag_chain, q)
                if qna_result:
                    st.session_state.qna_history.append(
                        {
                            "question": q,
                            "answer": qna_result.get("result", ""),
                            "sources": qna_result.get("source_documents") or [],
                            "validation": None,
                        }
                    )

        if st.session_state.qna_history:
            col_left, col_right = st.columns([1, 1])
            with col_left:
                if st.button("Clear Q&A history"):
                    st.session_state.qna_history = []
                    st.rerun()
            with col_right:
                st.caption(f"Q&A turns: {len(st.session_state.qna_history)}")

            for idx, turn in enumerate(reversed(st.session_state.qna_history), start=1):
                history_index = len(st.session_state.qna_history) - idx
                st.markdown(f"**Q{history_index + 1}.** {turn['question']}")
                st.write(turn["answer"])
                validate_key = f"validate_qna_{history_index}"
                if st.button("Validate this answer", key=validate_key):
                    with st.spinner("Validating answer..."):
                        try:
                            st.session_state.qna_history[history_index]["validation"] = _validate_with_second_model(
                                validator=validator_llm,
                                task_label="Q&A answer",
                                prompt_or_question=turn["question"],
                                answer=turn["answer"],
                                source_documents=turn.get("sources") or [],
                            )
                        except Exception:  # noqa: BLE001
                            st.error(GENERIC_MODEL_ERROR)

                if turn.get("validation"):
                    st.markdown("**Validation by second model**")
                    st.write(turn["validation"])
                if show_sources:
                    sources = turn.get("sources") or []
                    if sources:
                        with st.expander("Sources", expanded=False):
                            for s_idx, doc in enumerate(sources[:5], start=1):
                                text = getattr(doc, "page_content", "") or ""
                                snippet = text[:400].strip()
                                st.markdown(f"**Chunk {s_idx}:** {snippet}{'…' if len(text) > 400 else ''}")
