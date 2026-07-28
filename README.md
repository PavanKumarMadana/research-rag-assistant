# 🤖 AI Research & Knowledge Assistant

> **Enterprise-grade Retrieval-Augmented Generation (RAG) system** for intelligent document understanding, semantic search, AI-powered Q&A, and TensorFlow-based document classification.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00.svg)](https://tensorflow.org)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.3-yellow.svg)](https://chromadb.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Usage Examples](#-usage-examples)
- [TensorFlow ML Pipeline](#-tensorflow-ml-pipeline)
- [Deployment](#-deployment)
- [Testing](#-testing)
- [Assumptions & Design Decisions](#-assumptions--design-decisions)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)

---

## 🎯 Overview

Modern organizations deal with thousands of research papers, technical documents, and knowledge bases. The **AI Research & Knowledge Assistant** solves the challenge of finding accurate information from these documents by combining:

- **Retrieval-Augmented Generation (RAG)** for grounded AI responses
- **Semantic Search** using vector embeddings for context-aware retrieval
- **TensorFlow ML Pipeline** for automatic document classification
- **Conversation Memory** for contextual follow-up interactions
- **Multi-document Comparison & Summarization**

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Applications                      │
│  (React Frontend / Postman / cURL / Python SDK)              │
└──────────────────────┬──────────────────────────────────────┘
                       │ REST APIs (FastAPI)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
├──────────────────────┬──────────────────────────────────────┤
│   Document Routes    │         RAG Routes                   │
│   /api/documents     │    /api/rag                          │
├──────────────────────┼──────────────────────────────────────┤
│   ML Routes          │      Analytics Routes                │
│   /api/ml            │    /api/analytics                    │
└──────────┬───────────┴──────────────┬───────────────────────┘
           │                          │
           ▼                          ▼
┌──────────────────┐    ┌─────────────────────────────┐
│   SQLAlchemy     │    │       ChromaDB              │
│   SQLite DB      │    │   (Vector Store)            │
│   Documents      │    │   - Embeddings              │
│   Conversations  │    │   - Semantic Search         │
│   Analytics      │    │   - Keyword Search          │
└──────────────────┘    └─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                             │
├──────────────────────┬──────────────────────────────────────┤
│  DocumentProcessor   │        LLMService                    │
│  - PDF Extraction    │    - Gemini / OpenAI                 │
│  - Text Chunking     │    - Response Generation             │
│  - Embedding Gen     │                                      │
├──────────────────────┼──────────────────────────────────────┤
│  RAGService          │    ClassificationService             │
│  - Q&A               │    - TF Model / Keyword Fallback     │
│  - Summarization     │    - Training Pipeline               │
│  - Comparison        │    - Auto-Classification             │
│  - Chat Memory       │                                      │
└──────────────────────┴──────────────────────────────────────┘
```

---

## ✨ Features

### ✅ Document Management
- Upload multiple PDF documents simultaneously
- Automatic text extraction and preprocessing
- Metadata tracking (pages, chunks, status, category)
- Delete and reprocess documents

### ✅ Document Processing Pipeline
- **Text Extraction**: PyMuPDF for high-quality PDF parsing
- **Data Cleaning**: Normalization, noise removal, URL stripping
- **Intelligent Chunking**: RecursiveCharacterTextSplitter with configurable chunk size (1000) and overlap (200)
- **Embedding Generation**: sentence-transformers (all-MiniLM-L6-v2, 384-dim)
- **Vector Indexing**: ChromaDB persistent storage

### ✅ Semantic Search
- **Keyword Search**: Direct text matching
- **Semantic Search**: Embedding-based similarity (cosine)
- **Hybrid Search**: Weighted combination of semantic + keyword (70/30)
- **Document Filtering**: Search within specific documents
- **Result Ranking**: By similarity score

### ✅ AI Question Answering (RAG)
- Grounded answers **only** from retrieved document context
- Source citations with document name and page number
- Confidence scoring
- "Cannot determine answer" for insufficient context
- Multiple search modes support

### ✅ Document Summarization
- **Executive Summary**: Key findings and conclusions
- **Technical Summary**: Methodology and technical details
- **Bullet Point Summary**: Concise key points
- **Key Takeaways**: Most important insights

### ✅ Document Comparison
- Compare methodologies, findings, advantages, disadvantages
- Structured comparison table generation
- Per-aspect breakdown per document
- Up to 5 documents simultaneously

### ✅ TensorFlow Document Classification
- 8 predefined categories (AI, ML, CV, NLP, Robotics, Security, Cloud, Uncategorized)
- Keyword-based fallback when TF model unavailable
- Training pipeline with synthetic data generation
- Model evaluation metrics
- Auto-classification on upload
- Batch classification for existing documents

### ✅ Conversation Memory
- Session-based conversation tracking
- Pronoun resolution ("its", "it" references)
- Follow-up question detection
- Full conversation history retrieval

### ✅ Analytics Dashboard
- Total documents, chunks, embeddings
- Query statistics and most queried documents
- Category distribution
- Recent activity log
- Response time tracking

### ✅ REST APIs
- Full CRUD for documents
- Swagger/OpenAPI documentation at `/docs`
- Pagination, filtering, sorting
- Proper error handling
- Request validation

---

## 🛠 Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **Python 3.11** | Core programming language |
| **FastAPI** | Web framework with async support |
| **SQLAlchemy 2.0** | Async ORM for database operations |
| **SQLite** | Lightweight embedded database |
| **Pydantic** | Data validation and settings |

### AI & ML
| Technology | Purpose |
|------------|---------|
| **LangChain** | RAG orchestration and text splitting |
| **ChromaDB** | Vector database for embeddings |
| **sentence-transformers** | Embedding model (all-MiniLM-L6-v2) |
| **Google Gemini 2.0 Flash** | Primary LLM |
| **OpenAI GPT-4o-mini** | Alternative LLM provider |
| **TensorFlow 2.x** | Document classification model |
| **Scikit-learn** | ML preprocessing (TF-IDF, LabelEncoder) |
| **PyMuPDF** | PDF text extraction |

### Deployment
| Technology | Purpose |
|------------|---------|
| **Docker** | Containerization |
| **Render** | Backend deployment |
| **GitHub Actions** | CI/CD pipeline |

---

## 📁 Project Structure

```
research-rag-assistant/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── config.py          # Configuration management
│   │   │   ├── database.py        # Database setup and sessions
│   │   │   ├── logging.py         # Structured logging setup
│   │   │   └── security.py        # Auth, API keys, JWT
│   │   ├── models/
│   │   │   ├── document.py        # Document SQLAlchemy model
│   │   │   ├── conversation.py    # Conversation & Message models
│   │   │   └── analytics.py       # Analytics event model
│   │   ├── schemas/
│   │   │   ├── document.py        # Document Pydantic schemas
│   │   │   ├── rag.py            # RAG request/response schemas
│   │   │   └── analytics.py      # Analytics schemas
│   │   ├── repositories/
│   │   │   ├── document_repository.py
│   │   │   ├── conversation_repository.py
│   │   │   └── analytics_repository.py
│   │   ├── services/
│   │   │   ├── document_processor.py  # PDF pipeline
│   │   │   └── llm_service.py         # LLM abstraction
│   │   ├── vector_store/
│   │   │   ├── embedding_service.py   # Embedding generation
│   │   │   └── vector_store_service.py # ChromaDB operations
│   │   ├── rag/
│   │   │   └── rag_service.py    # RAG: Q&A, summary, compare
│   │   ├── ml/
│   │   │   └── classification_service.py # TF + keyword classifier
│   │   ├── analytics/
│   │   │   └── analytics_service.py # Analytics business logic
│   │   ├── routes/
│   │   │   ├── document_routes.py   # Document management APIs
│   │   │   ├── rag_routes.py       # RAG APIs
│   │   │   ├── ml_routes.py        # ML APIs
│   │   │   └── analytics_routes.py # Analytics APIs
│   │   └── main.py              # FastAPI application entry
│   └── requirements.txt
├── models/                      # Trained TF models
├── data/                        # SQLite DB, uploads, vector store
├── tests/
│   ├── test_document_repository.py
├── scripts/
│   └── train_model.py          # TF model training script
├── .github/workflows/
│   └── ci.yml                  # GitHub Actions CI/CD
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── render.yaml
├── vercel.json
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/research-rag-assistant.git
cd research-rag-assistant

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY or OPENAI_API_KEY

# 5. Run the application
uvicorn backend.app.main:app --reload
```

### Using Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t research-rag-assistant .
docker run -p 8000:8000 --env-file .env research-rag-assistant
```

### Access the Application

- **API**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/analytics/health

---

## 🔧 Configuration

All configuration is managed through environment variables. See `.env.example` for all options.

### Required Configuration

```bash
# Choose at least one LLM provider
GEMINI_API_KEY=your_gemini_key
# or
OPENAI_API_KEY=your_openai_key

# LLM Provider selection
LLM_PROVIDER=gemini  # or "openai"
```

### Optional Configuration

```bash
# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Vector Store
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Search
TOP_K_RESULTS=5

# Model
MODEL_PATH=./models/document_classifier
```

---

## 📚 API Documentation

### Document Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/documents/upload` | Upload PDF files |
| `GET` | `/api/documents/` | List all documents |
| `GET` | `/api/documents/{id}` | Get document details |
| `DELETE` | `/api/documents/{id}` | Delete a document |
| `POST` | `/api/documents/reprocess/{id}` | Reprocess a document |
| `GET` | `/api/documents/search/{query}` | Search across documents |

### RAG & AI Assistant

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/rag/ask` | Ask a question with RAG |
| `POST` | `/api/rag/summarize` | Summarize a document |
| `POST` | `/api/rag/compare` | Compare documents |
| `POST` | `/api/rag/chat` | Chat with AI assistant |
| `GET` | `/api/rag/conversations/{session_id}` | Get conversation history |

### Machine Learning

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/ml/classify/{document_id}` | Classify a document |
| `POST` | `/api/ml/classify-text` | Classify arbitrary text |
| `GET` | `/api/ml/model-info` | Get model information |
| `POST` | `/api/ml/train` | Train the TF model |
| `GET` | `/api/ml/evaluate` | Evaluate model |
| `POST` | `/api/ml/classify-all` | Batch classify documents |

### Analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/analytics/overview` | Get overview statistics |
| `GET` | `/api/analytics/full` | Get full analytics |
| `GET` | `/api/analytics/health` | Health check |

---

## 💡 Usage Examples

### Upload a Document

```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "files=@research_paper.pdf"
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/api/rag/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main contribution of this paper?",
    "session_id": "session-123"
  }'
```

### Semantic Search

```bash
curl -X GET "http://localhost:8000/api/documents/search/transformer%20architecture?search_mode=semantic&top_k=5"
```

### Summarize Document

```bash
curl -X POST http://localhost:8000/api/rag/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc-uuid-here",
    "summary_type": "executive",
    "max_length": 500
  }'
```

### Compare Documents

```bash
curl -X POST http://localhost:8000/api/rag/compare \
  -H "Content-Type: application/json" \
  -d '{
    "document_ids": ["doc1-uuid", "doc2-uuid"],
    "comparison_aspects": ["Methodology", "Conclusions"],
    "session_id": "session-123"
  }'
```

### Classify Document

```bash
curl -X POST http://localhost:8000/api/ml/classify/doc-uuid-here
```

### Train TensorFlow Model

```bash
python scripts/train_model.py
```

---

## 🧠 TensorFlow ML Pipeline

### Pipeline Stages

1. **Data Preparation**: Synthetic training data from keyword definitions
2. **Feature Engineering**: TF-IDF vectorization with n-grams (1-2)
3. **Model Architecture**: 
   - Input → Dense(256, ReLU) → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.2) → Dense(64, ReLU) → Dense(8, Softmax)
4. **Training**: 50 epochs, Adam optimizer, categorical crossentropy
5. **Evaluation**: Accuracy, classification report, confusion matrix
6. **Persistence**: Saved as TensorFlow SavedModel + Pickle vectorizer/encoder
7. **Prediction**: Real-time classification via API

### Categories

1. Artificial Intelligence
2. Machine Learning
3. Computer Vision
4. Natural Language Processing
5. Robotics
6. Cyber Security
7. Cloud Computing
8. Uncategorized

---

## 🚢 Deployment

### Deploy to Render (Backend)

1. Fork the repository to your GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Connect your GitHub repository
4. Use the following settings:
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables**: Add `GEMINI_API_KEY`, `OPENAI_API_KEY`
5. Deploy!

Or use the included `render.yaml` blueprint:
1. Push to GitHub
2. Connect Render to your repository
3. Render auto-detects `render.yaml` and deploys

### Deploy to Vercel (Frontend)

1. Push the repository to GitHub
2. Import the project in [Vercel](https://vercel.com)
3. Use the included `vercel.json`
4. Set `VITE_API_BASE_URL` to the deployed Render backend URL
5. Vercel builds `frontend/` with `npm run build` and serves `frontend/dist`

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=backend --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_document_repository.py -v
```

---

## 🤔 Assumptions & Design Decisions

### Search Strategies
| Strategy | When to Use |
|----------|-------------|
| **Keyword** | Exact phrase matching, known terminology |
| **Semantic** | Conceptual questions, different wording than source |
| **Hybrid** | Best balance of precision and recall |

### Chunking Strategy
- **RecursiveCharacterTextSplitter**: Maintains semantic boundaries
- **Chunk Size: 1000 characters**: Balances context length with precision
- **Overlap: 200 characters**: Ensures no context is lost at boundaries
- **Separators**: Prioritize paragraph breaks, then sentences

### Classification Approach
- **TensorFlow model** for production use after training
- **Keyword fallback** when TF is unavailable (lighter, no GPU needed)
- **Automated classification** triggers on document upload

### Conversation Memory
- **Session-based** (not user-based) for simplicity
- **Last 10 messages** kept for context window management
- **Pronoun detection** for follow-up question handling

### Database Choice
- **SQLite**: Zero configuration, sufficient for single-server deployment
- **Async SQLAlchemy**: Non-blocking database operations
- **Easily swappable** to PostgreSQL via connection string

---

## ⚠️ Limitations

1. **PDF-only**: Currently supports only PDF format
2. **Single-user**: No multi-user authentication out of the box
3. **SQLite**: Not suitable for high-concurrency production use
4. **Synthetic training data**: Model accuracy depends on keyword coverage
5. **No OCR**: Scanned PDFs without text layer cannot be processed
6. **Memory-bound**: Large documents (>100 pages) may consume significant RAM
7. **English-centric**: Optimized for English language documents

---

## 🔮 Future Improvements

### Security & Access
- [ ] JWT-based authentication and authorization
- [ ] Multi-user support with roles
- [ ] API rate limiting

### Retrieval & AI
- [ ] Streaming LLM responses
- [ ] BM25 + Vector hybrid retrieval (Elasticsearch)
- [ ] Cross-encoder reranking models
- [ ] Agent-based architecture for complex workflows

### Document Handling
- [ ] Multi-format support (DOCX, TXT, HTML, Markdown)
- [ ] OCR for scanned PDFs using Tesseract
- [ ] Image and table extraction
- [ ] Batch document upload with progress tracking

### Engineering
- [ ] Redis caching for embeddings
- [ ] PostgreSQL migration for production
- [ ] Kubernetes deployment manifests
- [ ] End-to-end integration tests
- [ ] Load testing and performance optimization
- [ ] Monitoring with Prometheus/Grafana

### Frontend
- [ ] React dashboard with real-time updates
- [ ] Drag-and-drop file upload
- [ ] Document preview in browser
- [ ] Real-time streaming responses

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com) for RAG framework
- [ChromaDB](https://chromadb.com) for vector storage
- [Google Gemini](https://deepmind.google/technologies/gemini/) for LLM capabilities
- [Sentence-Transformers](https://sbert.net) for embeddings
- [FastAPI](https://fastapi.tiangolo.com) for the web framework

---

<div align="center">
  <sub>Built with ❤️ for the AI Research & Knowledge Assistant Assignment</sub>
</div>
