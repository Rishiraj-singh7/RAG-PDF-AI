
import os

import streamlit as st

import rag_engine


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PDF AI Chatbot", page_icon="📄", layout="wide")
st.title("📄 PDF AI Chatbot")
st.caption(
    "Upload PDF(s), then ask questions. Answers are grounded in your documents "
    "and always cite the source page(s)."
)

@st.cache_resource(show_spinner="Loading the embedding model (first time only)…")
def load_embeddings():
    """
    Load the Sentence-Transformers embedding model once and reuse it.

    @st.cache_resource keeps the same object alive across reruns and users, so
    we don't reload the ~80 MB model on every interaction.
    """
    return rag_engine.get_embeddings()


def get_api_key():
    """
    Fetch the Groq API key.

    We look in two places (in order):
      1. Streamlit secrets  -> used on Streamlit Community Cloud (st.secrets)
      2. Environment var    -> used for local development (GROQ_API_KEY)

    Returns:
        str | None: the key, or None if it has not been configured.
    """

    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")



if "messages" not in st.session_state:
    # Chat history: a list of dicts {role, content, sources}.
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    # The searchable index built from the uploaded PDFs.
    st.session_state.vectorstore = None
if "processed_files" not in st.session_state:
    # Names of the PDFs currently indexed (shown in the sidebar).
    st.session_state.processed_files = []


with st.sidebar:
    st.header("1 · Upload your PDFs")

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files (max 50 MB each)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    # The 50 MB limit is also enforced globally in .streamlit/config.toml,
    # but we double-check here to give a friendly message.
    MAX_BYTES = 50 * 1024 * 1024
    oversized = [f.name for f in (uploaded_files or []) if f.size > MAX_BYTES]
    if oversized:
        st.error(f"These files exceed 50 MB and were skipped: {', '.join(oversized)}")

    if st.button("Process documents", type="primary", disabled=not uploaded_files):
        valid_files = [f for f in uploaded_files if f.size <= MAX_BYTES]
        if not valid_files:
            st.warning("No valid files to process.")
        else:
            with st.spinner("Reading, chunking and indexing your PDFs…"):
                # Run the first three RAG steps: extract -> split -> embed/store.
                pages = rag_engine.extract_documents_from_pdfs(valid_files)

                if not pages:
                    st.error(
                        "No extractable text found. The PDF may be scanned "
                        "images without a text layer."
                    )
                else:
                    chunks = rag_engine.split_documents(pages)
                    embeddings = load_embeddings()
                    st.session_state.vectorstore = rag_engine.build_vectorstore(
                        chunks, embeddings
                    )
                    st.session_state.processed_files = [f.name for f in valid_files]
                    # Start a fresh conversation for the new document set.
                    st.session_state.messages = []
                    st.success(
                        f"Indexed {len(chunks)} chunks from "
                        f"{len(valid_files)} file(s). Ask away! 👉"
                    )

    # Show what is currently indexed.
    if st.session_state.processed_files:
        st.markdown("**Indexed documents:**")
        for name in st.session_state.processed_files:
            st.markdown(f"- {name}")

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()



# Replay the existing conversation so it persists across reruns.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # For assistant messages we also show the source excerpts in an
        # expander (Source Attribution requirement).
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📑 Sources & excerpts used"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['source']} — page {src['page']}**")
                    st.caption(src["excerpt"])
                    st.divider()



question = st.chat_input("Ask a question about your PDFs…")

if question:
    # Guard rails: make sure the app is ready before answering.
    api_key = get_api_key()
    if st.session_state.vectorstore is None:
        st.warning("Please upload and process at least one PDF first.")
        st.stop()
    if not api_key:
        st.error(
            "No Groq API key found. Set GROQ_API_KEY as an environment variable "
            "(local) or in Streamlit secrets (cloud)."
        )
        st.stop()

    # 1. Show the user's message immediately.
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 2. Retrieve relevant chunks and generate the answer.
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents and thinking…"):
            chunks = rag_engine.retrieve_chunks(st.session_state.vectorstore, question)

            # Convert prior messages into the (role, content) form the engine wants.
            history = [
                (m["role"], m["content"]) for m in st.session_state.messages[:-1]
            ]

            answer = rag_engine.answer_question(question, chunks, history, api_key)

            # Build a tidy list of sources/excerpts to display and to store.
            sources = [
                {
                    "source": c.metadata.get("source", "unknown"),
                    "page": c.metadata.get("page", "?"),
                    # A short preview of the chunk text.
                    "excerpt": c.page_content[:400].strip() + "…",
                }
                for c in chunks
            ]

        st.markdown(answer)
        with st.expander("📑 Sources & excerpts used"):
            for src in sources:
                st.markdown(f"**{src['source']} — page {src['page']}**")
                st.caption(src["excerpt"])
                st.divider()

    # 3. Persist the assistant turn (with its sources) into chat history.
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
