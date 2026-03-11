

# 📄 Financial RAG Assistant

A **Retrieval-Augmented Generation (RAG)** application that allows users to ask questions about PDF documents. The system indexes document content using embeddings and a FAISS vector database, retrieves relevant chunks, and generates grounded answers using an LLM.

This project demonstrates how modern AI systems combine **vector search + large language models** to build document assistants.

---

# 🚀 Features

- 📄 PDF document ingestion
- ✂️ Smart text chunking
- 🧠 Semantic embeddings
- 🔎 Vector similarity search with FAISS
- 🤖 LLM-based grounded answer generation
- 💾 Persistent FAISS index storage
- 🐳 Dockerized environment
- 🌐 Streamlit web interface

---

# 🧠 Architecture

The system follows a standard **RAG pipeline**:

```
PDF
 ↓
Text extraction
 ↓
Chunking
 ↓
Embedding model
 ↓
FAISS vector index
 ↓
Similarity search
 ↓
Relevant chunks
 ↓
LLM API
 ↓
Final answer
```

The LLM answers questions **only using retrieved document context**, reducing hallucinations and enabling answers grounded in real data.

---

# 📂 Project Structure

```
financial-rag-assistant/
│
├── app/
│   └── streamlit_app.py        # Web UI
│
├── src/
│   ├── config.py
│   │
│   ├── ingestion/
│   │   ├── pdf_loader.py
│   │   ├── text_splitter.py
│   │   └── embedder.py
│   │
│   ├── retrieval/
│   │   └── vector_store.py
│   │
│   ├── llm/
│   │   └── generator.py
│   │
│   └── pipeline/
│       └── rag_pipeline.py
│
├── data/                       # Input PDFs
├── storage/                    # Saved FAISS index
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# ⚙️ Setup

## 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

## 2️⃣ Add environment variables

Create a `.env` file:

```
OPENAI_API_KEY=your_key_here
```

---

# ▶️ Run the App

```bash
streamlit run app/streamlit_app.py
```

Then open:

```
http://localhost:8501
```

---

# 🐳 Run with Docker

Build the image:

```bash
docker build -t rag-assistant .
```

Run the container:

```bash
docker run \
  --env-file .env \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/storage:/app/storage \
  rag-assistant
```

### Volume mounts

| Folder | Purpose |
|------|------|
| `data/` | Input PDF documents (read-only) |
| `storage/` | Persistent FAISS index |

---

# 📊 How It Works

### 1️⃣ Document Processing

PDF documents are:

- loaded
- cleaned
- split into overlapping chunks

### 2️⃣ Embedding Generation

Each chunk is converted into a **vector representation** using an embedding model.

### 3️⃣ Vector Search

Embeddings are stored in a **FAISS index**, enabling fast semantic search.

### 4️⃣ Retrieval

When a user asks a question:

- the question is embedded
- similar chunks are retrieved

### 5️⃣ Answer Generation

The retrieved chunks are provided to an LLM, which generates a grounded answer.

---

# 🔒 Persistent Index

The FAISS index and metadata are saved to disk:

```
storage/
 ├── faiss.index
 └── chunks.json
```

This allows the system to **load the index on startup instead of rebuilding it**.

---

# 🧪 Example Questions

```
What risks does the company mention?

What are the key financial results?

What strategy does the company describe?
```

---

# 🛠 Future Improvements

Possible extensions:

- Multiple document support
- Chat-style interaction
- Upload new PDFs through the UI
- Hybrid search (keyword + vector)
- Streaming responses
- Evaluation metrics for retrieval quality

---

# 📚 Technologies Used

- Python
- FAISS
- OpenAI API
- Streamlit
- Docker

---

# 🎯 Purpose

This project was built as a **learning and demonstration project** to understand how modern AI applications combine:

- embeddings
- vector search
- large language models

into practical document assistants.

---