"""
main.py -- command line entry point for the RAG pipeline.

Usage:
    python main.py build            # ingest + chunk + embed + index everything in data/
    python main.py ask "question"   # retrieve + generate an answer
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from embed_store import build_index  # noqa: E402
from generate import answer_question  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1]
    if command == "build":
        build_index(data_dir=str(Path(__file__).resolve().parent / "data"))
    elif command == "ask":
        question = " ".join(sys.argv[2:])
        if not question:
            print("Usage: python main.py ask \"your question\"")
            return
        print(answer_question(question, show_sources=True))
    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
