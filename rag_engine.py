

import os
import tempfile

# Silence ChromaDB's anonymous telemetry (prints harmless errors otherwise).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


LLM_MODEL_NAME = "llama-3.3-70b-versatile"


CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 150    # characters shared between neighbouring chunks


TOP_K = 4

def extract_documents_from_pdfs(uploaded_files):
    """
    Read one or more uploaded PDF files and return a list of LangChain
    `Document` objects — one per page.

    Each Document carries:
        * page_content : the text of that page
        * metadata     : {"source": <filename>, "page": <1-based page number>}

    Keeping the page number here is what later lets us show "Source: report.pdf,
    page 7" in the answer (Source Attribution requirement).

    Args:
        uploaded_files: list of Streamlit UploadedFile objects.

    Returns:
        list[Document]: every page of every PDF, with metadata attached.
    """
    all_pages = []

    for uploaded_file in uploaded_files:
        # PyPDFLoader needs a real file path, but Streamlit gives us the file
        # in memory. So we write the bytes to a temporary file on disk first.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            # PyPDFLoader returns one Document per page and sets metadata["page"]
            # (0-indexed) automatically.
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            for page in pages:
                # Overwrite metadata with friendly values:
                #  - keep the ORIGINAL filename (the temp path is meaningless)
                #  - convert the 0-based page index to a human 1-based page number
                page.metadata["source"] = uploaded_file.name
                page.metadata["page"] = page.metadata.get("page", 0) + 1
                # Skip empty pages (e.g. scanned image-only pages with no text).
                if page.page_content and page.page_content.strip():
                    all_pages.append(page)
        finally:
            # Always clean up the temporary file, even if loading failed.
            os.remove(tmp_path)

    return all_pages


def split_documents(documents):
    """
    Split full pages into smaller, overlapping chunks.

    Why split? LLMs and embedding models work best on short, focused passages.
    A whole page is often too big and mixes several topics, which hurts search
    quality. Overlap makes sure a sentence sitting on a chunk boundary is not
    lost.

    We use RecursiveCharacterTextSplitter, which tries to split on natural
    boundaries first (paragraphs, then lines, then words) so chunks stay
    readable.

    Args:
        documents: list[Document] (typically the per-page output of step 1).

    Returns:
        list[Document]: smaller chunks. Each chunk inherits the page metadata
        of the page it came from, so source attribution still works.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Order matters: try paragraph breaks first, then lines, then spaces.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# 3. Chunks -> embeddings -> ChromaDB vector store
# ---------------------------------------------------------------------------
def get_embeddings():
    """
    Create the embedding model (Sentence-Transformers via LangChain).

    This is separated into its own function so the Streamlit app can cache it
    (st.cache_resource) — loading the model takes a few seconds and we only
    want to do it once.

    Returns:
        HuggingFaceEmbeddings: object that converts text <-> vectors.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        # Normalised vectors make cosine similarity behave nicely.
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks, embeddings):
    """
    Embed every chunk and store the vectors in a ChromaDB collection.

    The collection is persisted to a unique temporary directory on disk. We do
    NOT use a pure in-memory collection because Streamlit re-runs the whole
    script on every interaction, and ChromaDB's shared in-memory system database
    gets reset between runs — which makes the collection "disappear" and raises
    `no such table: collections` when you later ask a question. A small on-disk
    directory survives across those reruns for the life of the session.

    Args:
        chunks:     list[Document] from split_documents().
        embeddings: the model returned by get_embeddings().

    Returns:
        Chroma: a vector store you can search with retrieve_chunks().
    """
    # A fresh, unique folder per upload so different document sets never clash.
    persist_dir = tempfile.mkdtemp(prefix="chroma_")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="pdf_chunks",
        persist_directory=persist_dir,
    )
    return vectorstore


# ---------------------------------------------------------------------------
# 4. Question -> most relevant chunks (semantic search)
# ---------------------------------------------------------------------------
def retrieve_chunks(vectorstore, question, k=TOP_K):
    """
    Find the `k` chunks whose meaning is closest to the user's question.

    ChromaDB compares the question's embedding against every stored chunk
    embedding and returns the nearest ones (semantic similarity, not keyword
    matching). The matching score is returned too so we can show how confident
    a match is.

    Args:
        vectorstore: the Chroma store from build_vectorstore().
        question:    the user's question (plain text).
        k:           how many chunks to return.

    Returns:
        list[Document]: the top-k most relevant chunks (with page metadata).
    """
    # similarity_search returns the chunks ordered from most to least relevant.
    return vectorstore.similarity_search(question, k=k)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly \
about the user's uploaded PDF documents.

Rules you must always follow:
1. Use ONLY the information in the "Context" provided below. Do not use outside \
knowledge.
2. If the answer is not in the context, reply exactly: \
"I could not find the answer to that in the uploaded document(s)."
3. Be clear and concise. Quote the document where helpful.
4. At the end of your answer, add a "Sources:" line listing the document name \
and page number(s) you used, e.g. "Sources: report.pdf (page 3, page 7)".
"""


def format_context(chunks):
    """
    Turn the retrieved chunks into a single text block for the LLM prompt.

    We label each chunk with its source and page so the model can cite them
    accurately.

    Args:
        chunks: list[Document] from retrieve_chunks().

    Returns:
        str: a numbered, labelled context block.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        blocks.append(
            f"[Chunk {i} | {source} | page {page}]\n{chunk.page_content}"
        )
    return "\n\n".join(blocks)


def answer_question(question, chunks, chat_history, api_key):
    """
    Ask the LLM to answer the question using ONLY the retrieved chunks.

    Args:
        question:     the user's current question.
        chunks:       relevant chunks from retrieve_chunks().
        chat_history: list of (role, content) tuples from earlier in the chat,
                      so the model understands follow-up questions.
        api_key:      Groq API key.

    Returns:
        str: the model's answer (already contains a "Sources:" line).
    """
    # Create the LLM client. temperature=0 keeps answers factual/deterministic.
    llm = ChatGroq(
        model=LLM_MODEL_NAME,
        temperature=0,
        api_key=api_key,
    )

    context = format_context(chunks)

    # Build the message list: system rules -> past turns -> current question.
    messages = [("system", SYSTEM_PROMPT)]

    # Include a short window of previous turns so follow-ups make sense.
    for role, content in chat_history[-6:]:
        messages.append((role, content))

    # The final user message bundles the retrieved context with the question.
    messages.append(
        (
            "user",
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer using only the context above and cite your sources.",
        )
    )

    response = llm.invoke(messages)
    return response.content
