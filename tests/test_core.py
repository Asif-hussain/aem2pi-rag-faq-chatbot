#!/usr/bin/env python3
"""
test_core.py

Basic test suite for the RAG FAQ chatbot system.

Tests verify:
1. Query pipeline can process questions and return valid responses
2. Response contains all required keys (user_question, system_answer, chunks_related)
3. Retrieved chunks are non-empty
4. System can handle various question types
5. Works with both local models (Ollama) and API models (OpenRouter)

To run tests:
    pytest tests/test_core.py -v
    OR
    python tests/test_core.py
"""
import os
import sys
import json
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()

# Check which model mode is being used
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
print(f"{'Testing with LOCAL models' if USE_LOCAL_MODEL else 'Testing with API models'}")

from src.query import answer_query


def test_basic_query():
    """
    Test that the system can answer a basic policy question.
    """
    question = "What is the company leave policy?"
    result = answer_query(question)

    # Verify required keys exist
    assert "user_question" in result, "Missing 'user_question' in response"
    assert "system_answer" in result, "Missing 'system_answer' in response"
    assert "chunks_related" in result, "Missing 'chunks_related' in response"

    # Verify data types
    assert isinstance(result["user_question"], str), "user_question should be string"
    assert isinstance(result["system_answer"], str), "system_answer should be string"
    assert isinstance(result["chunks_related"], list), "chunks_related should be list"

    # Verify non-empty results
    assert len(result["system_answer"]) > 0, "system_answer should not be empty"
    assert len(result["chunks_related"]) > 0, "chunks_related should not be empty"

    print("✅ Test passed: Basic query returns valid response")
    print(f"   Question: {question}")
    print(f"   Answer length: {len(result['system_answer'])} characters")
    print(f"   Chunks retrieved: {len(result['chunks_related'])}")


def test_password_reset_query():
    """
    Test password reset related question.
    """
    question = "How can I reset my password?"
    result = answer_query(question)

    assert "system_answer" in result
    assert "chunks_related" in result
    assert len(result["chunks_related"]) > 0

    print("✅ Test passed: Password reset query works")


def test_remote_work_query():
    """
    Test remote work policy question.
    """
    question = "Can I work remotely?"
    result = answer_query(question)

    assert "system_answer" in result
    assert "chunks_related" in result
    assert len(result["system_answer"]) > 0

    print("✅ Test passed: Remote work query works")


def test_chunks_are_strings():
    """
    Test that retrieved chunks are valid text strings.
    """
    question = "What are the payroll dates?"
    result = answer_query(question)

    chunks = result["chunks_related"]
    assert all(isinstance(chunk, str) for chunk in chunks), "All chunks should be strings"
    assert all(len(chunk) > 0 for chunk in chunks), "All chunks should be non-empty"

    print("✅ Test passed: All chunks are valid strings")


if __name__ == "__main__":
    print("Running RAG FAQ Chatbot Tests")
    print(f"Mode: {'LOCAL models (Ollama)' if USE_LOCAL_MODEL else 'API models (OpenRouter)'}")
    print("=" * 50 + "\n")

    try:
        # Check prerequisites
        if USE_LOCAL_MODEL:
            from src.local_model import check_ollama_running
            if not check_ollama_running():
                print("❌ Ollama server is not running!")
                print("💡 Start Ollama with: 'ollama serve' or 'docker-compose up -d'")
                sys.exit(1)
            print("✅ Ollama server is running\n")
        else:
            if not os.getenv("OPENROUTER_API_KEY"):
                print("❌ OPENROUTER_API_KEY not set in .env")
                sys.exit(1)
            print("✅ OpenRouter API key configured\n")

        # Run tests
        test_basic_query()
        print()
        test_password_reset_query()
        print()
        test_remote_work_query()
        print()
        test_chunks_are_strings()
        print("\n" + "=" * 50)
        print("✅ All tests passed!")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        sys.exit(1)
