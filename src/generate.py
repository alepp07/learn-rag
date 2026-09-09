"""
generate.py -- Steps 4 & 5: augment a prompt with retrieved chunks, then generate an answer.

This is the "AG" in RAG (Augmented Generation). Nothing here fine-tunes a
model on your documents -- we just paste the relevant chunks into the prompt
at question time. Simple, but it's the entire trick behind RAG.

Uses Groq's free API for generation (console.groq.com) -- free API key, no
credit card required, and fast inference on open-source models.
"""

import os

from groq import Groq

from embed_store import search

client = Groq()  # reads GROQ_API_KEY from the environment
MODEL = "openai/gpt-oss-20b"


def build_prompt(question: str, retrieved: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[Source: {r['source']}]\n{r['text']}" for r in retrieved
    )
    return f"""Answer the question using ONLY the context below.
If the context doesn't contain the answer, say so honestly instead of guessing.

CONTEXT:
{context_block}

QUESTION: {question}

ANSWER:"""


def answer_question(question: str, k: int = 4, show_sources: bool = False) -> str:
    retrieved = search(question, k=k)
    prompt = build_prompt(question, retrieved)

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content

    if show_sources:
        sources = ", ".join(f"{r['source']} (score={r['score']:.2f})" for r in retrieved)
        answer += f"\n\n[Retrieved from: {sources}]"
    return answer


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is this document about?"
    print(answer_question(q, show_sources=True))