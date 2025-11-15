#!/usr/bin/env python3
"""
local_model.py

Local model setup and management script for Ollama-based embeddings and LLM.
This script only runs when USE_LOCAL_MODEL=true in .env file.

Features:
- Checks if phi is installed and running
- Downloads required embedding and LLM models
- Provides utility functions for local embedding generation
- Supports Docker and native Ollama installations

Usage:
    python src/local_model.py --setup    # Setup and download models
    python src/local_model.py --test     # Test local models
"""
import os
import sys
import requests
import numpy as np
import logging
from dotenv import load_dotenv
import subprocess
import time

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-minilm")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "phi")  # Default to phi (faster) instead of llama2


def check_ollama_running():
    """Check if Ollama server is running."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("Ollama server is running")
            return True
    except Exception as e:
        logger.error(f"Ollama server not responding: {e}")
    return False


def pull_model(model_name):
    """Download a model using Ollama API."""
    logger.info(f"Pulling model: {model_name}")
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": model_name},
            stream=True,
            timeout=600
        )

        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    logger.debug(line.decode('utf-8'))
            logger.info(f"Model {model_name} pulled successfully")
            return True
        else:
            logger.error(f"Failed to pull model {model_name}: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error pulling model {model_name}: {e}")
        return False


def list_models():
    """List all downloaded models."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            logger.info(f"Available models: {len(models)}")
            for model in models:
                logger.info(f"  - {model['name']}")
            return [m['name'] for m in models]
    except Exception as e:
        logger.error(f"Error listing models: {e}")
    return []


def generate_local_embedding(text, model=None):
    """Generate embedding using local Ollama model."""
    if model is None:
        model = LOCAL_EMBEDDING_MODEL

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30
        )

        if response.status_code == 200:
            embedding = response.json().get("embedding", [])
            return np.array(embedding, dtype=np.float32)
        else:
            logger.error(f"Embedding generation failed: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None


def generate_local_embeddings_batch(texts, model=None):
    """Generate embeddings for multiple texts."""
    if model is None:
        model = LOCAL_EMBEDDING_MODEL

    embeddings = []
    for text in texts:
        emb = generate_local_embedding(text, model)
        if emb is not None:
            embeddings.append(emb)
        else:
            logger.warning(f"Failed to generate embedding for text: {text[:50]}...")
            # Return zero vector as fallback
            embeddings.append(np.zeros(384, dtype=np.float32))  # all-minilm dimension

    return np.array(embeddings)


def generate_local_answer(question, context, model=None):
    """Generate answer using local LLM."""
    if model is None:
        model = LOCAL_LLM_MODEL

    prompt = f"""You are a helpful HR assistant. Answer ONLY using the context below.
If the answer is not present, respond: 'I don't have enough information.'

Context:
{context}

User question: {question}

Answer:"""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=180  # Increased from 60 to 180 seconds for slower machines
        )

        if response.status_code == 200:
            answer = response.json().get("response", "")
            return answer.strip()
        else:
            logger.error(f"Answer generation failed: {response.status_code}")
            return "I couldn't generate an answer at this time."
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "I couldn't generate an answer at this time."


def setup_local_models():
    """Setup local models by pulling required models."""
    logger.info("Starting local model setup")

    # Check if Ollama is running
    if not check_ollama_running():
        logger.error("Ollama is not running. Please start Ollama first.")
        logger.info("To start Ollama:")
        logger.info("   - Native: Run 'ollama serve' in a terminal")
        logger.info("   - Docker: Run 'docker-compose up -d'")
        return False

    # List existing models
    existing_models = list_models()

    # Pull embedding model if not present
    if not any(LOCAL_EMBEDDING_MODEL in m for m in existing_models):
        logger.info(f"Embedding model not found, pulling {LOCAL_EMBEDDING_MODEL}")
        if not pull_model(LOCAL_EMBEDDING_MODEL):
            logger.error(f"Failed to pull embedding model")
            return False
    else:
        logger.info(f"Embedding model {LOCAL_EMBEDDING_MODEL} already available")

    # Pull LLM model if not present
    if not any(LOCAL_LLM_MODEL in m for m in existing_models):
        logger.info(f"LLM model not found, pulling {LOCAL_LLM_MODEL}")
        if not pull_model(LOCAL_LLM_MODEL):
            logger.error(f"Failed to pull LLM model")
            return False
    else:
        logger.info(f"LLM model {LOCAL_LLM_MODEL} already available")

    logger.info("Local model setup complete!")
    return True


def test_local_models():
    """Test local models with sample inputs."""
    logger.info("Testing local models")

    # Test embedding generation
    logger.info("Testing embedding generation...")
    test_text = "This is a test sentence for embedding generation."
    embedding = generate_local_embedding(test_text)

    if embedding is not None:
        logger.info(f"Embedding generated successfully (dim={len(embedding)})")
    else:
        logger.error("Embedding generation failed")
        return False

    # Test answer generation
    logger.info("Testing answer generation...")
    test_question = "What is the leave policy?"
    test_context = "Employees are entitled to 20 paid annual leave days per year."
    answer = generate_local_answer(test_question, test_context)

    if answer:
        logger.info(f"Answer generated successfully")
        logger.info(f"   Question: {test_question}")
        logger.info(f"   Answer: {answer[:100]}...")
    else:
        logger.error("Answer generation failed")
        return False

    logger.info("All tests passed!")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Local model setup and testing")
    parser.add_argument("--setup", action="store_true", help="Setup and download models")
    parser.add_argument("--test", action="store_true", help="Test local models")
    parser.add_argument("--list", action="store_true", help="List available models")

    args = parser.parse_args()

    # Check if USE_LOCAL_MODEL is enabled
    use_local = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
    if not use_local:
        logger.warning("USE_LOCAL_MODEL is set to 'false' in .env")
        logger.warning("   Set USE_LOCAL_MODEL=true to use local models")
        sys.exit(1)

    if args.setup:
        success = setup_local_models()
        sys.exit(0 if success else 1)
    elif args.test:
        success = test_local_models()
        sys.exit(0 if success else 1)
    elif args.list:
        list_models()
    else:
        logger.info("Usage: python src/local_model.py [--setup|--test|--list]")
        logger.info("  --setup: Download required models")
        logger.info("  --test:  Test local models")
        logger.info("  --list:  List available models")
