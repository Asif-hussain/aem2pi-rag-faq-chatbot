# Quick Start Guide

Get the RAG system running in 5 minutes.

## Setup (API Mode)

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Configure environment:**
```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

Get API key from https://openrouter.ai/keys

**3. Build index:**
```bash
python src/build_index.py
```

**4. Ask questions:**
```bash
python src/query.py "How many leave days do I get?"
```

**5. Run tests:**
```bash
python tests/test_core.py
```

## Local Mode (No API Key)

Want to run offline? See [LOCAL_MODELS.md](LOCAL_MODELS.md) for setup with Ollama.

## Sample Questions

```bash
python src/query.py "How many leave days do I get per year?"
python src/query.py "How can I reset my password?"
python src/query.py "What is the remote work policy?"
python src/query.py "When does payroll run?"
```

**Save responses to file:**
```bash
python src/query.py "How many leave days do I get per year?" --save
# Appends to outputs/sample_queries.json
```

## What You Get

Each query returns JSON with:
- `user_question` - the question you asked
- `system_answer` - grounded answer from the LLM
- `chunks_related` - top-5 retrieved context chunks

See example outputs in [outputs/sample_queries.json](outputs/sample_queries.json)

## Key Files

- `data/faq_document.txt` - Source FAQ (1,881 words)
- `src/build_index.py` - Data pipeline
- `src/query.py` - Query pipeline
- `src/evaluator.py` - Answer quality scoring
- `tests/test_core.py` - Test suite

Full docs in [README.md](README.md)
