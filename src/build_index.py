#!/usr/bin/env python3
"""
build_index.py

Data pipeline script that:
1. Loads the FAQ document from data/faq_document.txt
2. Chunks the text into semantic segments using sentence-based chunking
3. Generates embeddings for each chunk using OpenAI's embedding model via OpenRouter
4. Saves embeddings to data/embeddings.npy for vector search
5. Saves chunks to data/chunks.pkl for retrieval

Technical choices:
- Sentence-based chunking: Preserves semantic coherence by keeping related sentences together
  while maintaining manageable chunk sizes (100 words max per chunk)
- OpenAI text-embedding-3-small: Balances quality and cost, produces 1536-dimensional vectors
- FAISS not used for storage here to maintain compatibility with query.py's numpy-based search
"""
import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import logging

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Check if using local model
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"

if USE_LOCAL_MODEL:
    logger.info("Using LOCAL model for embeddings")
    try:
        from src.local_model import generate_local_embeddings_batch, check_ollama_running
    except ModuleNotFoundError:
        from local_model import generate_local_embeddings_batch, check_ollama_running

    # Verify Ollama is running
    if not check_ollama_running():
        raise RuntimeError("Ollama server is not running. Start it with 'ollama serve' or 'docker-compose up -d'")
else:
    logger.info("Using OPENROUTER API for embeddings")
    # Initialize OpenRouter client with OpenAI-compatible API
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

def chunk_text(text, max_words=100):
    """
    Break a document into chunks without splitting sentences. Each chunk stays
    within the max_words limit.
    """

    if not text or not isinstance(text, str):
        logger.warning("chunk_text: Received empty or invalid text input.")
        return []

    # Basic sentence split (keeps things simple but avoids empty entries)
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]

    chunks = []
    current_chunk = []
    current_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        # If a single sentence is too large, make it a chunk on its own
        if word_count > max_words:
            if current_chunk:
                chunks.append(". ".join(current_chunk) + ".")
                current_chunk = []
                current_count = 0

            chunks.append(sentence + ".")
            logger.debug(f"Sentence exceeded limit and stored directly: {sentence[:40]}...")
            continue

        # Start a new chunk if adding this sentence exceeds the limit
        if current_count + word_count > max_words and current_chunk:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentence]
            current_count = word_count
        else:
            current_chunk.append(sentence)
            current_count += word_count

    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    logger.info(f"Created {len(chunks)} chunks from document.")
    return chunks



def get_embeddings(texts, model="openai/text-embedding-3-small", retries=2, batch_size=16):
    """
    Generate embeddings for single text or a list of texts.
    Uses local Ollama model if USE_LOCAL_MODEL=true, otherwise OpenRouter API.

    Args:
        texts (str or List[str]): Input text(s)
        model (str): Embedding model
        retries (int): Number of retries per batch (API only)
        batch_size (int): Number of texts per API call

    Returns:
        np.ndarray: Embeddings array of shape (n_texts, embedding_dim)
    """
    if isinstance(texts, str):
        texts = [texts]

    # Clean texts
    clean_texts = []
    for t in texts:
        if not t or not isinstance(t, str):
            logger.warning(f"Skipping invalid text input: {t}")
            continue
        t = t.strip()
        if t:
            clean_texts.append(t)

    if not clean_texts:
        logger.warning("No valid texts to embed.")
        return np.array([])

    # Use local model if enabled
    if USE_LOCAL_MODEL:
        logger.info(f"Generating embeddings using LOCAL model")
        embeddings = generate_local_embeddings_batch(clean_texts, model=os.getenv("LOCAL_EMBEDDING_MODEL"))
        return embeddings

    # Otherwise use OpenRouter API
    embeddings_list = []

    # Process in batches
    for i in range(0, len(clean_texts), batch_size):
        batch = clean_texts[i:i + batch_size]
        for attempt in range(retries + 1):
            try:
                response = client.embeddings.create(
                    model=model,
                    input=batch
                )

                # Validate response
                if not hasattr(response, "data") or len(response.data) != len(batch):
                    raise ValueError("Unexpected embedding response format")

                batch_embeddings = np.array([d.embedding for d in response.data], dtype="float32")
                embeddings_list.append(batch_embeddings)
                logger.debug(f"Batch {i//batch_size + 1}: Generated {len(batch)} embeddings")
                break

            except Exception as e:
                logger.error(f"Batch {i//batch_size + 1} failed (attempt {attempt + 1}/{retries + 1}): {e}")
        else:
            logger.error(f"Batch {i//batch_size + 1} failed after all retries.")
            embeddings_list.append(np.zeros((len(batch), 1536), dtype="float32"))  # fallback

    # Combine all batches
    embeddings = np.vstack(embeddings_list)
    return embeddings


def build_index():
    """
    Pipeline to load a document, chunk it, generate embeddings, and save to disk.

    Saves:
    - data/embeddings.npy: numpy array of shape (num_chunks, embedding_dim)
    - data/chunks.pkl: pickled list of chunk strings
    """
    # Configuration
    if USE_LOCAL_MODEL:
        embedding_model = os.getenv("LOCAL_EMBEDDING_MODEL", "all-minilm")
    else:
        embedding_model = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

    doc_path = "data/faq_document.txt"
    embeddings_path = "data/embeddings.npy"
    chunks_path = "data/chunks.pkl"
    batch_size = 10

    # Validate document
    if not os.path.exists(doc_path):
        logger.error(f"FAQ document not found at {doc_path}")
        raise FileNotFoundError(f"FAQ document not found at {doc_path}")

    # Load document
    logger.info(f"Loading document from {doc_path}")
    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()

    logger.info(f"Document length: {len(text)} characters, {len(text.split())} words")

    # Chunk document
    chunks = chunk_text(text)
    logger.info(f"Created {len(chunks)} chunks from document")

    if not chunks:
        logger.warning("No chunks created. Aborting index build.")
        return

    # Generate embeddings for chunks in batch
    logger.info(f"Generating embeddings using model: {embedding_model}")

    embeddings_array = []
    total_chunks = len(chunks)

    # Using tqdm to show progress
    for i in tqdm(range(0, total_chunks, batch_size), desc="Embedding batches"):
        batch = chunks[i:i+batch_size]
        batch_embeddings = get_embeddings(batch, model=embedding_model, batch_size=batch_size)
        if batch_embeddings.size == 0:
            logger.warning(f"Skipping batch {i//batch_size + 1} due to embedding failure")
            continue
        embeddings_array.append(batch_embeddings)

    if not embeddings_array:
        logger.error("Failed to generate any embeddings. Aborting.")
        return

    embeddings_array = np.vstack(embeddings_array)
    logger.info(f"Generated embeddings with shape: {embeddings_array.shape}")

    # Save embeddings
    np.save(embeddings_path, embeddings_array)
    logger.info(f"Saved embeddings to {embeddings_path}")

    # Save corresponding chunks
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks[:len(embeddings_array)], f)
    logger.info(f"Saved {len(chunks[:len(embeddings_array)])} chunks to {chunks_path}")

    logger.info("Index building complete! Ready for querying.")

if __name__ == "__main__":
    build_index()
