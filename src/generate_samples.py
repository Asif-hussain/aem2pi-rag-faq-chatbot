#!/usr/bin/env python3
"""
generate_samples.py

Generate sample query-response pairs to demonstrate the RAG system.
This script runs predefined queries through the system and saves the output
to outputs/sample_queries.json with all required keys.
"""
import os
import json
import logging
from query import answer_query

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Predefined sample queries to demonstrate the system
SAMPLE_QUESTIONS = [
    "How many leave days do I get per year?",
    "How can I reset my password?",
    "What is the remote work policy?",
]

def generate_sample_queries(output_path="outputs/sample_queries.json", append=False):
    """
    Generate sample query-response pairs and save them to a JSON file.

    Args:
        output_path: Path to save the output JSON file
        append: If True, append to existing file; if False, replace it
    """
    # Ensure outputs directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load existing queries if appending
    results = []
    if append and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            logger.info(f"Loaded {len(results)} existing queries from {output_path}")
        except (json.JSONDecodeError, IOError):
            logger.warning(f"Could not read existing file, starting fresh")

    logger.info("="*60)
    logger.info("Generating sample query-response pairs")
    logger.info(f"Total queries: {len(SAMPLE_QUESTIONS)}")
    logger.info("="*60)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        logger.info(f"\nProcessing query {i}/{len(SAMPLE_QUESTIONS)}: {question}")
        try:
            # Get answer from the RAG system
            result = answer_query(question)

            # Validate required keys
            required_keys = ["user_question", "system_answer", "chunks_related"]
            for key in required_keys:
                if key not in result:
                    logger.error(f"Missing required key: {key}")
                    result[key] = "" if key != "chunks_related" else []

            results.append(result)
            logger.info(f"✓ Query {i} completed successfully")

        except Exception as e:
            logger.error(f"✗ Error processing query {i}: {e}")
            # Add a placeholder result with error message
            results.append({
                "user_question": question,
                "system_answer": f"Error: {str(e)}",
                "chunks_related": []
            })

    # Save results to JSON file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("\n" + "="*60)
        logger.info(f"✓ Successfully saved {len(results)} query-response pairs to {output_path}")
        logger.info("="*60)

        # Display summary
        logger.info("\nSummary:")
        for i, result in enumerate(results, 1):
            logger.info(f"{i}. {result['user_question']}")
            logger.info(f"   Answer length: {len(result['system_answer'])} chars")
            logger.info(f"   Chunks retrieved: {len(result['chunks_related'])}")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise

if __name__ == "__main__":
    import sys

    # Allow custom output path from command line
    output_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/sample_queries.json"

    try:
        generate_sample_queries(output_path)
    except Exception as e:
        logger.error(f"Failed to generate sample queries: {e}")
        sys.exit(1)
