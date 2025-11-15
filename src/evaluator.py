#!/usr/bin/env python3
"""
evaluator.py

Bonus: Automated evaluation agent for RAG answer quality assessment.

Evaluates RAG system responses based on:
1. Chunk relevance: Do retrieved chunks contain information related to the question?
2. Answer accuracy: Is the answer correct and grounded in the provided chunks?
3. Answer completeness: Does the answer fully address the user's question?

Returns a score from 0-10 with detailed reasoning for the evaluation.

Usage:
    from evaluator import evaluate
    score_data = evaluate(user_question, system_answer, chunks_related)
    print(f"Score: {score_data['score']}/10")
    print(f"Reason: {score_data['reason']}")
"""
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "gpt-3.5-turbo")


def evaluate(user_question, system_answer, chunks):
    """
    Evaluates the quality of a RAG system response.

    Uses an LLM to score the answer on a 0-10 scale based on relevance,
    accuracy, and completeness. Provides detailed reasoning for the score.

    Args:
        user_question: The original user question
        system_answer: The system's generated answer
        chunks: List of retrieved context chunks used to generate the answer

    Returns:
        Dictionary with keys:
        - score (float): Quality score from 0-10
        - reason (str): Detailed explanation of the score

    Scoring rubric:
    - 9-10: Perfect answer, fully addresses question with accurate info from chunks
    - 7-8: Good answer, mostly complete and accurate with minor gaps
    - 5-6: Adequate answer, addresses question but missing details or clarity
    - 3-4: Weak answer, partially addresses question or has inaccuracies
    - 0-2: Poor answer, doesn't address question or contains major errors
    """
    # Format chunks for evaluation
    chunks_text = "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(chunks)])

    # Evaluation prompt with clear scoring criteria
    evaluation_prompt = f"""You are an expert evaluator assessing the quality of answers from a RAG (Retrieval-Augmented Generation) FAQ chatbot system.

Evaluate the system's answer based on three criteria:
1. **Chunk Relevance** (0-3 points): Do the retrieved chunks contain relevant information to answer the question?
2. **Answer Accuracy** (0-4 points): Is the answer factually correct and grounded in the provided chunks? No hallucinations?
3. **Answer Completeness** (0-3 points): Does the answer fully address what the user asked?

USER QUESTION:
{user_question}

SYSTEM ANSWER:
{system_answer}

RETRIEVED CHUNKS:
{chunks_text}

Provide your evaluation as JSON with exactly these keys:
{{
  "score": <float from 0-10>,
  "reason": "<detailed explanation covering relevance, accuracy, and completeness>"
}}

Be objective and critical. Return ONLY valid JSON, no other text."""

    if not OPENROUTER_KEY:
        return {
            "score": 0.0,
            "reason": "OPENROUTER_API_KEY not set. Cannot perform evaluation."
        }

    try:
        url = f"{OPENROUTER_BASE}/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        }
        messages = [{"role": "user", "content": evaluation_prompt}]
        body = {
            "model": EVALUATOR_MODEL,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.2  # Low temperature for consistent evaluation
        }

        response = requests.post(url, headers=headers, json=body, timeout=30)

        if response.status_code != 200:
            return {
                "score": 0.0,
                "reason": f"Evaluation API call failed: {response.status_code} {response.text}"
            }

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # Try to parse JSON response
        try:
            # Remove markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            evaluation_result = json.loads(content.strip())

            # Validate score is in range
            score = float(evaluation_result.get("score", 0))
            score = max(0.0, min(10.0, score))  # Clamp to 0-10 range

            return {
                "score": score,
                "reason": evaluation_result.get("reason", "No reason provided")
            }

        except (json.JSONDecodeError, ValueError):
            # If JSON parsing fails, return raw content as reason
            return {
                "score": 0.0,
                "reason": f"Failed to parse evaluation response. Raw output: {content}"
            }

    except Exception as e:
        return {
            "score": 0.0,
            "reason": f"Evaluation error: {str(e)}"
        }


if __name__ == "__main__":
    # Demo evaluation with sample data
    print("Running sample evaluation...\n")

    sample_question = "How can I reset my password?"
    sample_answer = (
        "To reset your password, use the 'Forgot Password' link on the login page. "
        "This will send you a secure one-time link via email that is valid for 30 minutes. "
        "If you experience account lockouts or suspicious activity, contact IT Security immediately."
    )
    sample_chunks = [
        "All employees must use the company Single Sign-On (SSO) to access internal applications. "
        "Passwords must be at least 12 characters with a mix of upper/lowercase, digits, and symbols. "
        "Multi-factor authentication (MFA) is mandatory for all admin accounts and recommended for all employees. "
        "If an employee forgets their password, they can use the 'Forgot Password' link on the login page, "
        "which triggers an email with a secure one-time link valid for 30 minutes. "
        "For account lockouts or suspicious activity, employees should contact IT Security immediately."
    ]

    result = evaluate(sample_question, sample_answer, sample_chunks)

    print(f"Question: {sample_question}")
    print(f"\nAnswer: {sample_answer}")
    print(f"\nEvaluation Score: {result['score']}/10")
    print(f"Reason: {result['reason']}")
