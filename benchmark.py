"""Simple retrieval benchmark runner for the RAG project."""
import argparse
import os

from config import GOOGLE_API_KEY

DEFAULT_QUESTIONS = [
    "What was total revenue?",
    "What were the main risks mentioned?",
    "What was the company strategy?",
    "What did management say about supply chain?",
    "What were the key financial highlights?",
]


def main():
    parser = argparse.ArgumentParser(description="Benchmark retrieval latency and chunk counts.")
    parser.add_argument("--doc-id", default=None, help="Restrict benchmark to one document ID.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve per question.")
    parser.add_argument(
        "--questions",
        nargs="*",
        default=DEFAULT_QUESTIONS,
        help="Optional list of question strings to benchmark.",
    )
    args = parser.parse_args()

    if not GOOGLE_API_KEY:
        print("Benchmark skipped: GOOGLE_API_KEY is missing. Add it to your .env file before running retrieval tests.")
        raise SystemExit(1)

    from query import benchmark_retrieval

    metrics = benchmark_retrieval(args.questions, doc_id=args.doc_id, top_k=args.top_k)
    print("Retrieval benchmark results")
    print("-" * 28)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
