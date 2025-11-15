# RAG-based FAQ Support Chatbot

A Retrieval-Augmented Generation system that answers employee questions about HR policies using vector search and language models. Supports both cloud APIs and local models for complete offline operation.

## Overview

This system chunks an FAQ document into semantic segments, generates embeddings, retrieves relevant context using cosine similarity, and produces grounded answers. You can run it using OpenRouter APIs or completely offline with local Ollama models.

**Key Features:**
- Chunks FAQ document into 28 semantic segments (~100 words each)
- Embeddings via OpenAI text-embedding-3-small or local all-minilm
- Top-5 chunk retrieval using k-NN cosine similarity
- Answer generation via GPT-3.5-turbo or local Llama2/Phi models
- JSON output with question, answer, and source chunks
- Evaluator agent for answer quality scoring (bonus)
- Full local mode support with Docker or native Ollama (bonus)

## Repository Structure

```
m2-mid-assignment/
├── data/
│   ├── faq_document.txt        # Source FAQ document (1000+ words)
│   ├── embeddings.npy          # Generated embeddings (created by build_index.py)
│   └── chunks.pkl              # Text chunks (created by build_index.py)
├── outputs/
│   └── sample_queries.json     # Example query-response pairs
├── src/
│   ├── build_index.py          # Data pipeline: chunking + embedding generation
│   ├── query.py                # Query pipeline: retrieval + answer generation
│   ├── evaluator.py            # Bonus: Answer quality evaluation agent
│   └── local_model.py          # Bonus: Local model setup & management (Ollama)
├── tests/
│   └── test_core.py            # Test suite (supports both API and local models)
├── .env.example                # Environment variable template
├── docker-compose.yml          # Ollama Docker setup (for local models)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Setup

### Requirements

- **Python 3.9+** (tested on Python 3.13.5)
- **Choose one:**
  - **API Mode**: OpenRouter API key (provides access to OpenAI models)
  - **Local Mode**: Docker or Ollama installed locally (no API key needed, runs offline)
- 8GB+ RAM recommended (16GB+ for local models)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd m2-mid-assignment
```

2. Create and activate virtual environment:
```bash
python3 -m venv AEM2-ASSIGNMENT
source AEM2-ASSIGNMENT/bin/activate  # macOS/Linux
# OR
AEM2-ASSIGNMENT\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
```

**Option A: Use API Models (OpenRouter)**
```bash
# Edit .env and set:
USE_LOCAL_MODEL=false
OPENROUTER_API_KEY=your-key-here
EMBEDDING_MODEL=openai/text-embedding-3-small
LLM_MODEL=gpt-3.5-turbo
```

To get an OpenRouter API key:
1. Visit https://openrouter.ai/
2. Sign up and create API key
3. Add credits ($5 minimum)

**Option B: Use Local Models (Ollama) - No API Key Needed**
```bash
# Edit .env and set:
USE_LOCAL_MODEL=true
LOCAL_EMBEDDING_MODEL=all-minilm
LOCAL_LLM_MODEL=phi  # Faster option (10-15s per query)
OLLAMA_BASE_URL=http://localhost:11434
```

Start Ollama server and download models:

**Using Docker (Recommended):**
```bash
docker-compose up -d
python src/local_model.py --setup
```

**Using Native Ollama:**
```bash
# Download from https://ollama.ai/download
ollama serve
python src/local_model.py --setup
```

**Model Options:**
- `phi` (1.6GB) - Fast, 10-15s per query, good quality (recommended for laptops)
- `llama2` (3.8GB) - Slower, 60-120s per query, better quality

See [LOCAL_MODELS.md](LOCAL_MODELS.md) for detailed setup guide.

## Usage

### Step 1: Build the Index

Generate embeddings and chunk the FAQ document:

```bash
python src/build_index.py
```

**What this does:**
- Loads `data/faq_document.txt`
- Chunks text into semantic segments (~100 words each)
- Generates embeddings using configured model (API or local)
- Saves `data/embeddings.npy` and `data/chunks.pkl`

Output shows whether you're using API or local models, number of chunks created, and embedding dimensions.

### Step 2: Query the System

Ask questions about the FAQ content:

```bash
python src/query.py "How many leave days do I get per year?"
```

**Output:** Structured JSON with `user_question`, `system_answer`, and `chunks_related` fields.

To save the response to `outputs/sample_queries.json`:

```bash
python src/query.py "How many leave days do I get per year?" --save
```

This will append the query-response pair to the existing `sample_queries.json` file, building up your collection of examples.

See [outputs/sample_queries.json](outputs/sample_queries.json) for full examples.

### Step 3: Run Tests

Verify the system works correctly:

```bash
pytest tests/test_core.py -v
```

OR run tests directly:

```bash
python tests/test_core.py
```

Tests verify embeddings, retrieval, and answer generation work correctly in both API and local modes.

### Bonus: Evaluate Answer Quality

Use the evaluator agent to score answer quality:

```bash
python src/evaluator.py
```

Or use programmatically:

```python
from src.evaluator import evaluate
from src.query import answer_query

# Get answer from system
result = answer_query("How can I reset my password?")

# Evaluate quality
evaluation = evaluate(
    result["user_question"],
    result["system_answer"],
    result["chunks_related"]
)

print(f"Score: {evaluation['score']}/10")
print(f"Reason: {evaluation['reason']}")
```

## Technical Choices

**Chunking:** Sentence-based chunking at ~100 words per chunk preserves semantic coherence. Sentences aren't split mid-way, keeping concepts intact.

**Embeddings:** API mode uses OpenAI text-embedding-3-small (1536 dims) for quality. Local mode uses all-minilm (384 dims) for offline operation. Both work well for semantic similarity.

**Retrieval:** k-NN cosine similarity with optional FAISS acceleration. Simple, interpretable, and effective for small-to-medium document collections. Retrieves top-5 chunks by default.

**Generation:** API mode uses GPT-3.5-turbo with grounded prompting (temp=0.3) for factual answers. Local mode uses Phi or Llama2 via Ollama, running completely offline with no API costs.

**Storage:** Numpy arrays for embeddings and pickle for chunks. Fast, memory-efficient, and easy to debug.

**Local Models:** Ollama provides a unified API for running LLMs locally via Docker or native install. Models are downloaded once and cached.

See [outputs/sample_queries.json](outputs/sample_queries.json) for 7 complete query-response examples.

## 📊 Metrics Summary

| Metric                  | Value                            |
| ----------------------- | -------------------------------- |
| Document Size           | 1,881 words                      |
| Chunks Generated        | 28 semantic chunks               |
| Embedding Dimensions    | 1536 (text-embedding-3-small)    |
| Average Query Time      | ~3-5 seconds                     |
| Top-k Retrieval         | 5 chunks (configurable via TOP_K)|

---

## Challenges and Lessons Learned

Finding the right chunk size took some experimentation. Too small and context is lost, too large and relevance drops. Settling on 100 words with sentence boundaries worked well. API rate limits needed retry logic and batch processing. Adding local model support meant abstracting the interface to work with both OpenRouter and Ollama APIs seamlessly.

## Notes

- Build the index first with `python src/build_index.py` before querying
- FAISS is optional, system falls back to numpy if unavailable
- Switch modes by changing `USE_LOCAL_MODEL` in `.env` and rebuilding the index
- Local models need ~4GB disk space (one-time download)
- Local queries take 10-30s depending on your hardware and model choice

## Conclusion

This system shows how vector search and language models combine to answer questions from private documents. The dual-mode architecture lets you choose between API convenience and local privacy. Works well for FAQ support, internal knowledge bases, or documentation Q&A.

---

## Author

Created as part of the AI Engineering Module 2 assignment on RAG systems.
