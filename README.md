# ai-assistant-backend

A production-ready, beginner-friendly Retrieval-Augmented Generation (RAG) backend built with **FastAPI**, **PostgreSQL**, **pgvector**, and **LangChain**.

---

## Features

- **Multi-Format Document Ingestion**: Upload `.pdf`, `.docx`, or `.txt` files.
- **Smart Text Chunking**: Recursive character splitting with configurable chunk size and overlap, preserving page and source metadata.
- **Vector Storage in PostgreSQL**: Non-blocking async vector search using `pgvector` and `JSONB` metadata indexing.
- **Tenant / Document Isolation**: Vector retrieval strictly filters by `document_id` so conversations never leak cross-document context.
- **Grounded RAG Pipeline**: LangChain LCEL pipeline (`retriever | prompt | llm | parser`) instructing the model to answer only using document facts.
- **General Fallback Answering**: When questions cannot be answered by the document, safely routes to a general AI assistant.
- **Conversation Logging**: Stores user questions and assistant answers in PostgreSQL for conversation history.
- **Automated Test Suite**: 19 unit and integration tests covering loaders, splitters, embeddings, vectorstore, retriever, RAG chains, and endpoints.

---

## Architecture Overview

```text
Upload Pipeline:
documents/api.py (POST /upload)
       ↓
documents/service.py
       ↓
rag/loaders.py (PDF, DOCX, TXT)
       ↓
rag/splitter.py (Recursive Character Splitter)
       ↓
rag/embeddings.py (1536-dim Embeddings)
       ↓
rag/vectorstore.py (PGVector)
       ↓
PostgreSQL + pgvector

Chat Pipeline:
chat_bot/api.py (POST /chat)
       ↓
chat_bot/service.py
       ↓
rag/chain.py
       ↓
rag/retriever.py (Top-K with document_id filter)
       ↓
rag/prompts.py (Factual grounding prompt)
       ↓
rag/llm.py (ChatOpenAI / gpt-4o-mini)
       ↓
Answer + Source Citations
```

---

## Quick Start Guide

### 1. Prerequisites
* **Python**: `3.14` (or `>=3.10`)
* **Docker Desktop**: For running PostgreSQL with pgvector
* **Poetry**: Package and dependency manager

### 2. Configure Environment (.env)
Copy the example environment file:
```powershell
cp .env.example .env
```
Ensure your `.env` contains:
```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/rag_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=4
LLM_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

### 3. Start PostgreSQL with pgvector
Start the Docker container:
```powershell
docker compose up -d
```
Verify the container is healthy:
```powershell
docker ps
```

### 4. Install Dependencies
```powershell
poetry install
```

### 5. Start the FastAPI Server
```powershell
poetry run uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```
Interactive API docs are available at:
👉 **[http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)**

### 6. Run the Automated Test Suite
```powershell
poetry run pytest -v
```

---

## API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check endpoint |
| `POST` | `/api/v1/documents/upload` | Upload `.pdf`, `.docx`, or `.txt` file and index in pgvector |
| `GET` | `/api/v1/documents` | List all uploaded documents |
| `GET` | `/api/v1/documents/{document_id}` | Get metadata for a specific document |
| `DELETE`| `/api/v1/documents/{document_id}` | Delete document from disk and database |
| `POST` | `/api/v1/chat` | Ask a question grounded in an uploaded document |
| `GET` | `/api/v1/chat/{document_id}/history` | Retrieve conversation history for a document |
