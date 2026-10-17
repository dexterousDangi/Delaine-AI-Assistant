# Delaine RAG Bot 🤖

A production-ready, AI-powered Telegram RAG (Retrieval-Augmented Generation) chatbot that answers questions from custom knowledge base using semantic search + LLMs.

Built for high performance, secure deployment, and 24/7 availability.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## 🌟 Live Demo

- **Telegram Bot:** [@YourBotName](https://t.me/YourBotName) 
- **Web Service:** `https://delaine-rag-bot-xxxx.onrender.com`
- **Health Check:** `/health`

## 🏗️ Architecture: 
User Query (Telegram)
       ↓
FastAPI Webhook (/telegram-webhook)
       ↓
Jina AI Embeddings (jina-embeddings-v2-base-en) → Query Vector
       ↓
FAISS Index (529 Chunks) → Top-K Semantic Search (k=5)
       ↓
Context + Query → Groq LLM (Llama-3.3-70b) → Grounded Answer
       ↓
Telegram Reply


## ✨ Key Features

- **Semantic Retrieval:** Uses Jina AI embeddings + FAISS for sub-100ms similarity search
- **529 Knowledge Chunks:** Custom-built vector index from domain documents
- **No Hallucination:** Strict RAG prompting - answers only from retrieved context
- **Secure by Design:** No secrets in repo, GitHub Push Protection compliant, .gitignore enforced
- **Dockerized:** One-command deployment anywhere (Render, Railway, AWS)
- **Telegram Webhook:** Real-time, serverless webhook (not polling)
- **Production Ready:** Health checks, error handling, logging

## 📁 Project Structure

delaine-rag-bot/
├── main.py              # FastAPI app + Telegram webhook + RAG pipeline
├── build_index.py       # Script to chunk docs and build FAISS index
├── requirements.txt     # Python dependencies
├── Dockerfile           # Production Docker build
├── .gitignore           # Excludes .env, venv, pycache
├── .env                 # LOCAL ONLY - API keys (never commit)
├── faiss_index/
│   ├── index.faiss      # FAISS vector store (binary)
│   └── index.pkl        # Metadata: chunks + sources
└── README.md            

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Uvicorn |
| Vector DB | FAISS (Facebook AI Similarity Search) |
| Embeddings | Jina AI `jina-embeddings-v2-base-en` (768 dim) |
| LLM | Groq - Llama 3.3 70B (ultra-fast inference) |
| Bot Framework | python-telegram-bot |
| Deployment | Docker + Render |
| Language | Python 3.11 |

## 🚀 Local Setup

### 1. Clone & Venv
```bash
git clone https://github.com/dexterousDangi/delaine-rag-bot.git
cd delaine-rag-bot
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

2. Install Dependenciesbashpip install -r requirements.txt3. Create .env file (LOCAL ONLY)envJINA_API_KEY=jina_xxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxx
TELEGRAM_BOT_TOKEN=123456:ABC-YourToken4.

 Build Index (if needed)bashpython build_index.py
# This will create faiss_index/index.faiss + index.pkl with 529 chunks5.
 Run Locallybashuvicorn main.py:app --reload --port 8000
# Test: http://localhost:8000/health