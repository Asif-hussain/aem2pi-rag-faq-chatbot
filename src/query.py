#!/usr/bin/env python3
"""
query.py

RAG-based query pipeline:
1. Load pre-computed embeddings and chunks
2. Accept user questions
3. Embed questions
4. Retrieve top-k relevant chunks (cosine similarity)
5. Generate answer using LLM (OpenRouter)
6. Return structured JSON: {user_question, system_answer, chunks_related}
"""
import os
import json
import pickle
import numpy as np
import requests
import logging
import time
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Check if using local model
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"

if USE_LOCAL_MODEL:
    logger.info("Using LOCAL model for embeddings and generation")
    try:
        from src.local_model import generate_local_embedding, generate_local_answer, check_ollama_running
    except ModuleNotFoundError:
        from local_model import generate_local_embedding, generate_local_answer, check_ollama_running

    # Verify Ollama is running
    if not check_ollama_running():
        raise RuntimeError("Ollama server is not running. Start it with 'ollama serve' or 'docker-compose up -d'")

# Configuration
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai")

if USE_LOCAL_MODEL:
    EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-minilm")
    LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama2")
else:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

TOP_K = int(os.getenv("TOP_K", 5))

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.pkl")


def load_chunks():
    """Load pre-processed text chunks from pickle file."""
    if not os.path.exists(CHUNKS_PATH):
        logger.error(f"Chunks file not found: {CHUNKS_PATH}")
        raise FileNotFoundError(f"Run 'python src/build_index.py' first.")
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    logger.info(f"Loaded {len(chunks)} chunks from disk.")
    return chunks


def embed_text(text, retries=2):
    """Generate embedding vector for text using local Ollama or OpenRouter API."""
    text = text.strip()
    if not text:
        logger.warning("Empty text provided for embedding.")
        return None

    # Use local model if enabled
    if USE_LOCAL_MODEL:
        start_time = time.time()
        logger.info(f"Generating embedding using LOCAL model: {EMBEDDING_MODEL}")
        embedding = generate_local_embedding(text, model=EMBEDDING_MODEL)
        elapsed = time.time() - start_time
        logger.info(f"Embedding generated in {elapsed:.2f}s")
        return embedding

    # Otherwise use OpenRouter API
    if not OPENROUTER_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set in .env")

    start_time = time.time()
    logger.info(f"Generating embedding using API model: {EMBEDDING_MODEL}")
    url = f"{OPENROUTER_BASE}/api/v1/embeddings"
    headers = {"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"}
    body = {"model": EMBEDDING_MODEL, "input": [text]}

    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=30)
            response.raise_for_status()
            data = response.json()
            embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)
            elapsed = time.time() - start_time
            logger.info(f"Embedding generated in {elapsed:.2f}s")
            return embedding
        except Exception as e:
            logger.warning(f"Embedding attempt {attempt+1} failed: {e}")
    logger.error("All embedding attempts failed.")
    return None


def cosine_similarity_search(query_embedding, embeddings_matrix, top_k=5):
    """k-NN cosine similarity search."""
    if query_embedding is None:
        raise ValueError("Query embedding is None.")

    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    normalized_embeddings = embeddings_matrix / np.where(norms==0, 1, norms)

    query_norm = np.linalg.norm(query_embedding)
    normalized_query = query_embedding / (query_norm if query_norm > 0 else 1)

    similarities = normalized_embeddings @ normalized_query
    top_indices = np.argsort(-similarities)[:top_k]
    top_scores = similarities[top_indices]

    logger.info(f"Top-{top_k} chunks retrieved.")
    return top_indices, top_scores


def search_index(query_embedding, top_k=TOP_K):
    """Search embedding index using FAISS (optional) or numpy fallback."""
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"Run 'python src/build_index.py' first to generate embeddings.")

    embeddings = np.load(EMBEDDINGS_PATH)

    try:
        import faiss
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        query_norm = query_embedding.reshape(1, -1).copy()
        faiss.normalize_L2(query_norm)
        distances, indices = index.search(query_norm, top_k)
        return indices[0], distances[0]
    except (ImportError, Exception):
        logger.info("FAISS unavailable, falling back to numpy cosine similarity.")
        return cosine_similarity_search(query_embedding, embeddings, top_k=top_k)


def generate_answer(question, context_chunks, retries=2):
    """Generate answer from LLM using retrieved chunks."""
    context = "\n\n---\n\n".join(context_chunks)

    # Use local model if enabled
    if USE_LOCAL_MODEL:
        start_time = time.time()
        logger.info(f"Generating answer using LOCAL LLM: {LLM_MODEL}")
        logger.info(f"This may take 10-30 seconds depending on your machine...")
        answer = generate_local_answer(question, context, model=LLM_MODEL)
        elapsed = time.time() - start_time
        logger.info(f"Answer generated in {elapsed:.2f}s")
        return answer

    # Otherwise use OpenRouter API
    start_time = time.time()
    logger.info(f"Generating answer using API LLM: {LLM_MODEL}")
    system_prompt = (
        "You are a helpful HR assistant. Answer ONLY using the context below. "
        "If the answer is not present, respond: 'I don't have enough information.'"
    )
    user_prompt = f"Context:\n{context}\n\nUser question: {question}"

    url = f"{OPENROUTER_BASE}/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"}
    body = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
        "max_tokens": 512,
        "temperature": 0.3
    }

    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=60)
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            elapsed = time.time() - start_time
            logger.info(f"Answer generated in {elapsed:.2f}s")
            return answer
        except Exception as e:
            logger.warning(f"LLM attempt {attempt+1} failed: {e}")
    logger.error("All LLM attempts failed.")
    return "I couldn't generate an answer at this time."


def answer_query(question, top_k=TOP_K):
    """Main RAG query function."""
    if not question or not isinstance(question, str):
        raise ValueError("Question must be a non-empty string.")

    total_start = time.time()

    # Display configuration
    mode = "LOCAL" if USE_LOCAL_MODEL else "API"
    logger.info("="*60)
    logger.info(f"RAG Query Pipeline - {mode} Mode")
    logger.info(f"Question: {question}")
    logger.info(f"Embedding Model: {EMBEDDING_MODEL}")
    logger.info(f"LLM Model: {LLM_MODEL}")
    logger.info("="*60)

    chunks = load_chunks()

    # Step 1: Embed query
    logger.info("\n[Step 1/3] Embedding query...")
    embedding = embed_text(question)
    if embedding is None:
        return {"user_question": question,
                "system_answer": "Failed to embed question.",
                "chunks_related": []}

    # Step 2: Retrieve chunks
    logger.info("\n[Step 2/3] Retrieving relevant chunks...")
    search_start = time.time()
    indices, scores = search_index(embedding, top_k=top_k)
    retrieved_chunks = [chunks[i] for i in indices]
    search_elapsed = time.time() - search_start
    logger.info(f"Retrieved {len(retrieved_chunks)} chunks in {search_elapsed:.2f}s")

    # Step 3: Generate answer
    logger.info("\n[Step 3/3] Generating answer...")
    answer = generate_answer(question, retrieved_chunks)

    total_elapsed = time.time() - total_start
    logger.info("\n" + "="*60)
    logger.info(f"Query completed in {total_elapsed:.2f}s total")
    logger.info("="*60)

    return {
        "user_question": question,
        "system_answer": answer,
        "chunks_related": retrieved_chunks
    }


if __name__ == "__main__":
    import sys
    from datetime import datetime

    if len(sys.argv) < 2:
        logger.info("Usage: python src/query.py \"your question here\" [--save]")
        logger.info("  --save: Save the response to outputs/query_<timestamp>.json")
        sys.exit(1)

    question = sys.argv[1]
    save_output = "--save" in sys.argv

    try:
        result = answer_query(question)

        # Always print to stdout
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Optionally save to file
        if save_output:
            output_dir = os.path.join(ROOT_DIR, "outputs")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "sample_queries.json")

            # Load existing queries or start with empty list
            existing_queries = []
            if os.path.exists(output_file):
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        existing_queries = json.load(f)
                except (json.JSONDecodeError, IOError):
                    logger.warning(f"Could not read existing file, starting fresh")

            # Append new result
            existing_queries.append(result)

            # Save updated list
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(existing_queries, f, indent=2, ensure_ascii=False)

            logger.info(f"\n✓ Response appended to: {output_file}")
            logger.info(f"✓ Total queries in file: {len(existing_queries)}")
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        sys.exit(1)
