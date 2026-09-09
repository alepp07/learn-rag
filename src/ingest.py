"""
ingest.py -- Step 1 of the RAG pipeline: load documents and break them into chunks.

WHY CHUNK AT ALL?
Embedding models and LLMs have limited context, and even within that limit,
one giant vector for a whole document blurs together everything in it, so
retrieval can't tell "the intro" from "the results section." Instead we split
text into small, roughly self-contained pieces (chunks) so each one can be
embedded and retrieved on its own.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import tiktoken

ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def load_text_from_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(str(path))
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks measured in TOKENS, not characters,
    since that's what embedding models and LLMs actually consume.

    Overlap matters: without it, a sentence that spans a chunk boundary gets
    cut in half, and neither resulting chunk contains the full idea.
    """
    tokens = ENCODING.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(ENCODING.decode(chunk_tokens))
        if end >= len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def load_and_chunk_directory(data_dir: str = "data") -> list[Chunk]:
    all_chunks = []
    chunk_id = 0
    for path in sorted(Path(data_dir).glob("*")):
        if path.suffix.lower() not in (".txt", ".md", ".pdf"):
            continue
        raw_text = load_text_from_file(path)
        raw_text = re.sub(r"\s+", " ", raw_text).strip()
        for piece in chunk_text(raw_text):
            all_chunks.append(Chunk(text=piece, source=path.name, chunk_id=chunk_id))
            chunk_id += 1
    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunk_directory()
    print(f"Loaded {len(chunks)} chunks from data/")
    if chunks:
        print("\n--- Sample chunk ---")
        print(f"[{chunks[0].source}] {chunks[0].text[:200]}...")