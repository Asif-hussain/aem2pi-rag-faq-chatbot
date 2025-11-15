# Local Models Setup

Run the RAG system completely offline using Ollama. No API keys, no costs, complete privacy.

## Why Local Models?

- No API costs - run unlimited queries
- Complete privacy - data stays on your machine
- Works offline after initial download
- No network latency
- Great for learning how LLMs work

## Quick Start

### 1. Configure Environment

Edit `.env`:
```bash
USE_LOCAL_MODEL=true
LOCAL_EMBEDDING_MODEL=all-minilm
LOCAL_LLM_MODEL=phi  # Recommended for laptops
OLLAMA_BASE_URL=http://localhost:11434
```

### 2. Start Ollama

**Docker (Recommended):**
```bash
docker-compose up -d
curl http://localhost:11434/api/tags  # Verify running
python src/local_model.py --setup      # Download models (~4GB)
```

**Native Ollama:**
```bash
# Install from https://ollama.ai/download
ollama serve
python src/local_model.py --setup
```

### 3. Build Index

```bash
python src/build_index.py
```

### 4. Query

```bash
python src/query.py "How many leave days do I get?"
```

You'll see timing info for each step and which models are being used.

---

## Available Models

### Embedding Models

| Model | Dimensions | Size | Speed | Quality |
|-------|-----------|------|-------|---------|
| **all-minilm** (default) | 384 | 45MB | Fast | Good |
| nomic-embed-text | 768 | 274MB | Medium | Better |
| mxbai-embed-large | 1024 | 670MB | Slow | Best |

The default all-minilm works well for most cases. Change via `LOCAL_EMBEDDING_MODEL` in `.env`.

### LLM Models

| Model | Size | RAM Needed | Speed | Quality |
|-------|------|------------|-------|---------|
| **phi** (default) | 1.6GB | 4GB | Fast (10-15s) | Good |
| llama2 | 3.8GB | 8GB | Slow (60-120s) | Better |
| mistral | 4.1GB | 8GB | Slow (60-90s) | Better |
| llama2:13b | 7.4GB | 16GB | Very Slow | Best |

**Recommendation:** Use `phi` for laptops and testing. Use `llama2` or `mistral` if you have a powerful machine and want better quality.

Change model in `.env` via `LOCAL_LLM_MODEL`, then run `python src/local_model.py --setup` to download.

## Management Commands

**Setup models:**
```bash
python src/local_model.py --setup
```

**Test models:**
```bash
python src/local_model.py --test
```

**List downloaded models:**
```bash
python src/local_model.py --list
```

## Common Issues

**Ollama not running:** Start with `docker-compose up -d` or `ollama serve`

**Model download fails:** Check internet connection and disk space (~4GB needed). Try smaller model like `phi`.

**Out of memory:** Switch to `phi` model, close other apps, or upgrade RAM.

**Slow responses (>30s):** Use `phi` instead of `llama2`. Check Docker resource limits. First query is slower as the model loads into memory.

## Performance Comparison

**API Mode:**
- Setup: 2 minutes
- Query time: 3-5 seconds
- Cost: ~$0.01 per query
- Quality: Excellent

**Local Mode:**
- Setup: 15-20 minutes (model downloads)
- Query time: 10-30 seconds (depends on model)
- Cost: Free
- Quality: Good to better

## Switching Modes

Change `USE_LOCAL_MODEL` in `.env`, then rebuild the index:
```bash
python src/build_index.py
python tests/test_core.py  # Test it works
```

You must rebuild because embedding dimensions differ (1536 for API vs 384 for local).

## Docker Notes

The docker-compose file runs Ollama on port 11434, persists models in a volume, and auto-restarts.

**Useful commands:**
```bash
docker-compose up -d        # Start
docker-compose down         # Stop
docker-compose logs -f      # View logs
docker-compose down -v      # Remove everything including models
```

## Tips

- You can't run both API and local modes simultaneously. Pick one and rebuild the index when you switch.
- After downloading models once, the system works completely offline.
- Ollama will automatically use your GPU if you have NVIDIA, AMD, or Apple Metal.
- Make sure you have at least 4-5GB free disk space for phi and all-minilm models.
