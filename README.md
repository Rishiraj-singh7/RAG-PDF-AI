# 📄 PDF AI Chatbot (RAG)

A web application that lets you **upload one or more PDFs and chat with them**.
Ask a question in plain English and get an answer that is grounded in your
documents — every answer cites the **source page(s)** and shows the **exact
excerpt** it used.

Built with a classic **Retrieval-Augmented Generation (RAG)** pipeline:

```
PDF → extract text (per page) 
→ split into chunks 
→ embed 
→ store in ChromaDB
                                                              │
question → 
embed → 
semantic search (top-k chunks) 
→ LLM (Groq/Llama 3.3) 
→ answer
```

---

##  Features (mapped to the requirements)

| Requirement | How it's met |
|---|---|
| **PDF upload up to 50 MB** | `st.file_uploader` (multi-file) + `maxUploadSize=50` in `.streamlit/config.toml`, with a friendly size check |
| **Text extraction** | `PyPDFLoader` extracts text **per page**, preserving page numbers |
| **Chunking** | `RecursiveCharacterTextSplitter` (1000 chars, 150 overlap) |
| **Embeddings + vector DB** | `sentence-transformers/all-MiniLM-L6-v2` → **ChromaDB** |
| **Retrieval** | Cosine similarity search, top-k = 4 |
| **Chat interface + history** | Streamlit `chat_message` / `chat_input`, history kept in `st.session_state` |
| **Source attribution** | Each answer lists `document (page N)` and shows the excerpt in an expander |
| **Deployment** | Streamlit Community Cloud (free, public URL) |

---

## 🗺️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          app.py  (Streamlit UI)                    │
│   • file upload (≤50 MB)   • chat box   • renders answers+sources  │
│   • st.session_state: chat history + vector store                  │
└───────────────┬───────────────────────────────┬──────────────────┘
                │ uploads PDFs                    │ question
                ▼                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                       rag_engine.py  (RAG logic)                   │
│                                                                    │
│  INGEST (once per upload):                                         │
│    1. extract_documents_from_pdfs  → PyPDFLoader (1 Doc / page)    │
│    2. split_documents              → RecursiveCharacterTextSplitter│
│    3. build_vectorstore            → MiniLM embeddings → ChromaDB  │
│                                                                    │
│  QUERY (per question):                                             │
│    4. retrieve_chunks              → ChromaDB similarity search    │
│    5. answer_question              → ChatGroq (Llama 3.3 70B)      │
└───────────────┬───────────────────────────────┬──────────────────┘
                ▼                                 ▼
        ┌───────────────┐                ┌────────────────────┐
        │   ChromaDB     │                │   Groq LLM API     │
        │ (in-memory     │                │ llama-3.3-70b-     │
        │  vector store) │                │ versatile          │
        └───────────────┘                └────────────────────┘
```

**Two files, clear separation of concerns:**
- [`app.py`](app.py) — everything the user sees (UI + session state).
- [`rag_engine.py`](rag_engine.py) — the RAG pipeline (no UI code), so the logic
  is easy to read and reuse.

---

##  Design decisions (required explanations)

### 1. Chunking strategy
- **Splitter:** `RecursiveCharacterTextSplitter`.
- **Size:** `1000` characters, **overlap:** `150` characters.
- **Why:** We split **page-by-page Documents**, so every chunk keeps the page
  number it came from — this is what makes accurate page citations possible.
  `1000` chars (~150–200 words) is large enough to hold a complete idea but
  small enough to give the embedding model a focused passage, which improves
  retrieval precision. The `150`-char overlap (15%) prevents a sentence that
  straddles a chunk boundary from being lost. The *recursive* splitter prefers
  natural break points (paragraph → line → sentence → word), so chunks stay
  readable instead of being cut mid-sentence.

### 2. Embedding model choice
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim vectors).
- **Why:** It is **open-source, free, fast, and CPU-friendly** (~80 MB), which
  is essential for a free cloud deployment with no GPU. Despite its small size
  it is one of the most popular general-purpose semantic-search models and
  offers an excellent quality/speed trade-off. Embeddings are **normalised** so
  similarity is plain cosine similarity. (For higher accuracy you could swap in
  `BAAI/bge-small-en-v1.5` — a one-line change in `rag_engine.py`.)

### 3. Prompt design
The system prompt (in `rag_engine.py`) enforces four rules:
1. **Use only the provided context** — no outside knowledge → reduces
   hallucination (helps "Correctness of Answers").
2. **Admit when the answer isn't there** with a fixed sentence, instead of
   guessing.
3. **Be concise and quote the document** where helpful.
4. **Always end with a `Sources:` line** citing document + page numbers.

Each retrieved chunk is injected into the prompt **labelled with its source and
page** (`[Chunk 1 | report.pdf | page 3]`) so the model can cite accurately.
A short window of previous turns is included so **follow-up questions** work.
`temperature=0` keeps answers factual and reproducible.

### 4. Retrieval approach
- Build a **ChromaDB** vector store from the chunk embeddings.
- At query time, embed the question and run **cosine-similarity search**,
  returning the **top-k = 4** most relevant chunks.
- Those chunks are passed to the LLM as context **and** surfaced to the user as
  source excerpts — so retrieval is both the grounding for the answer and the
  basis for attribution.
- The vector store is **in-memory per session**, which keeps the app stateless
  and privacy-friendly (one user's PDFs are never mixed with another's).

---

## 🚀 Setup & run locally

**Prerequisites:** Python 3.9+ and a free [Groq API key](https://console.groq.com/keys).

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd rag

# 2. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Provide your Groq API key (either option works)
export GROQ_API_KEY="your_key_here"           # option A: env var
# option B: cp .streamlit/secrets.toml.example .streamlit/secrets.toml  (then edit)

# 5. Run
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501), upload a PDF,
click **Process documents**, and start asking questions.

---

## ☁️ Deploy (Streamlit Community Cloud — free public URL)

1. Push this repo to **GitHub** (public).
2. Go to **https://share.streamlit.io** → **New app** → pick your repo.
3. Set **Main file path** to `app.py`.
4. Under **Advanced settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
5. Click **Deploy**. You'll get a public `https://<your-app>.streamlit.app` URL.

> 🔴 **Live URL:** _add your deployed URL here after deploying._

---

## 📁 Project structure

```
rag/
├── app.py                          # Streamlit UI + chat + session state
├── rag_engine.py                   # RAG pipeline (extract→split→embed→retrieve→answer)
├── requirements.txt                # pinned dependencies
├── README.md                       # this file
├── .gitignore                      # keeps secrets/venv out of git
├── .env.example                    # sample env var for local dev
└── .streamlit/
    ├── config.toml                 # 50 MB upload limit + theme
    └── secrets.toml.example        # sample secret for cloud/local
```

---

## 🔧 Tech stack
Python · Streamlit · LangChain · ChromaDB · Sentence-Transformers (MiniLM) ·
Groq (Llama 3.3 70B, open-source model).
