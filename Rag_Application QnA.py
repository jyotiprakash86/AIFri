import tempfile

import httpx
import streamlit as st
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdfminer.high_level import extract_text

# Step 1: Create an HTTP client that the LLM and embedding model will use.
client = httpx.Client(verify=False, trust_env=False)

# Step 2: Configure the chat model that will generate summaries and answers.
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="sk-tstRcevZRhRYRGeK-GkbnA",
    http_client=client,
    temperature=0,
)

# Step 3: Configure the embedding model used to convert PDF text into vectors.
embedding_model = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key="sk-tstRcevZRhRYRGeK-GkbnA",
    http_client=client,
    tiktoken_enabled=False,
    check_embedding_ctx_length=False,
)

# Step 4: Set the Streamlit page title and heading for the app UI.
st.set_page_config(page_title="RAG PDF Summarizer and Q&A")
st.title("RAG-powered PDF Summarizer and Q&A")

# Step 5: Initialize session state to remember the currently uploaded file.
if "active_file_name" not in st.session_state:
    st.session_state.active_file_name = None
# Step 6: Initialize session state to store the reusable RAG chain.
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
# Step 7: Initialize session state to store the generated PDF summary.
if "summary" not in st.session_state:
    st.session_state.summary = None
# Step 8: Initialize session state to store the chatbot conversation history.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Step 9: Let the user upload a PDF document from the Streamlit interface.
upload_file = st.file_uploader("Upload a PDF", type="pdf")

# Step 10: Start processing only after a PDF has been uploaded.
if upload_file:
    # Step 11: Check whether the uploaded file is different from the previously processed file.
    is_new_file = st.session_state.active_file_name != upload_file.name

    # Step 12: Rebuild the document index only when a new file is uploaded.
    if is_new_file:
        # Step 13: Save the current file name and clear old chat history for the new document.
        st.session_state.active_file_name = upload_file.name
        st.session_state.chat_history = []

        # Step 14: Write the uploaded PDF into a temporary local file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(upload_file.read())
            temp_file_path = temp_file.name

        # Step 15: Extract all readable text content from the temporary PDF file.
        raw_text = extract_text(temp_file_path)
        # Step 16: Split the PDF text into smaller chunks for embedding and retrieval.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(raw_text)

        # Step 17: Create and persist a Chroma vector database from the text chunks.
        with st.spinner("Indexing document..."):
            vectordb = Chroma.from_texts(
                chunks,
                embedding_model,
                persist_directory="./chroma_index",
            )
            vectordb.persist()

        # Step 18: Create a retriever so the app can fetch the most relevant chunks.
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )
        # Step 19: Build the RetrievalQA chain that combines retrieval with the chat model.
        st.session_state.rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
        )

        # Step 20: Prepare a fixed prompt that asks for a document summary.
        summary_prompt = "Please summarize this document based on the key topics:"
        # Step 21: Run the summary query through the RAG pipeline.
        with st.spinner("Running RAG summarization..."):
            result = st.session_state.rag_chain.invoke({"query": summary_prompt})

        # Step 22: Store the generated summary so it remains visible across reruns.
        st.session_state.summary = result["result"]

    # Step 23: Show the generated summary for the uploaded PDF.
    st.subheader("Summary")
    st.write(st.session_state.summary)

    # Step 24: Display the Q&A section title for document-based chat.
    st.subheader("Ask Questions About the PDF")

    # Step 25: Render the previous chat messages from session state.
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Step 26: Provide an input box for the user's next PDF-related question.
    user_question = st.chat_input("Ask a question about the uploaded PDF")

    # Step 27: Answer the question only when the user submits one.
    if user_question:
        # Step 28: Save the user's question in the chat history.
        st.session_state.chat_history.append(
            {"role": "user", "content": user_question}
        )
        # Step 29: Display the user's message in the chat interface.
        with st.chat_message("user"):
            st.write(user_question)

        # Step 30: Query the RAG chain and display the assistant's answer.
        with st.chat_message("assistant"):
            with st.spinner("Searching the document..."):
                answer = st.session_state.rag_chain.invoke({"query": user_question})
            st.write(answer["result"])

        # Step 31: Save the assistant's answer so it remains in the conversation history.
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer["result"]}
        )
