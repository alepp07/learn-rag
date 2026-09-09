"""
embed_store.py -- Steps 2 & 3: turn chunks into vectors, then store/search them.

WHAT'S AN EMBEDDING?
A model turns text into a list of numbers (a vector) such that texts with
similar MEANING end up close together in that vector space. "dog" and "puppy"
land near each other; "dog" and "tax return" land far apart. This lets us
find relevant chunks by math (cosine similarity) instead of exact keyword
matching -- so a question phrased differently from the source text can still
retrieve it.

This version embeds text LOCALLY using sentence-transformers -- free, no API
key, runs on your own CPU. The first run downloads a small model (~90MB)
from Hugging Face; after that it's cached and works offline.

Vectors are stored in a plain numpy array on disk. That's the whole "vector
database" for a project this size -- no need for Chroma/FAISS/Pinecone until
your corpus is too big to hold in memory.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import load_and_chunk_directory

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough to learn on
INDEX_DIR = Path(__file__).resolve().parent.parent / "index"

_model = None  # loaded lazily so `python src/ingest.py` alone doesn't pull it in


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading local embedding model ({EMBED_MODEL_NAME})...")
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Batch-embed a list of strings in one call."""
    model = get_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors.astype(np.float32)


def build_index(data_dir: str = "data") -> None:
    chunks = load_and_chunk_directory(data_dir)
    if not chunks:
        raise ValueError(f"No .txt/.md/.pdf files found in {data_dir}/")

    print(f"Embedding {len(chunks)} chunks...")
    vectors = embed_texts([c.text for c in chunks])

    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vectors)
    with open(INDEX_DIR / "chunks.json", "w") as f:
        json.dump(
            [{"text": c.text, "source": c.source, "chunk_id": c.chunk_id} for c in chunks],
            f,
        )
    print(f"Index saved to {INDEX_DIR}/ ({len(chunks)} chunks, {vectors.shape[1]}-dim vectors)")


def load_index() -> tuple[np.ndarray, list[dict]]:
    vectors = np.load(INDEX_DIR / "vectors.npy")
    with open(INDEX_DIR / "chunks.json") as f:
        chunks = json.load(f)
    return vectors, chunks


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / np.linalg.norm(query_vec)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix_norm @ query_norm


def search(query: str, k: int = 4) -> list[dict]:
    vectors, chunks = load_index()
    query_vec = embed_texts([query])[0]
    scores = cosine_similarity(query_vec, vectors)
    top_k_idx = np.argsort(scores)[::-1][:k]
    return [{**chunks[i], "score": float(scores[i])} for i in top_k_idx]


if __name__ == "__main__":
    build_index()