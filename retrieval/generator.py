"""
generator.py

Takes retrieved chunks and a question, and calls Claude to produce a
grounded answer that cites its sources explicitly. The model is instructed to answer only from
the provided context and to cite the source file + page for every claim,
or say plainly that the context doesn't contain the answer.

"""

import os
from dataclasses import dataclass
from typing import List

import anthropic

from retriever import RetrievedChunk

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a project intelligence assistant for infrastructure \
and engineering teams. You answer questions ONLY using the provided context \
chunks below. Follow these rules strictly:

1. Every factual claim in your answer must be followed by a citation in the \
   format [source_file, p.page_number].
2. If the provided context does not contain enough information to answer the \
   question, say so explicitly. Do not guess or use outside knowledge.
3. If multiple chunks disagree or are ambiguous, point that out rather than \
   silently picking one.
4. Keep answers concise and direct - this is a working tool for engineers and \
   project staff, not a long-form essay.
"""


@dataclass
class GroundedAnswer:
    answer: str
    sources_used: List[str]


def format_context(chunks: List[RetrievedChunk]) -> str:
    blocks = []
    for chunk in chunks:
        blocks.append(
            f"[{chunk.source_file}, p.{chunk.page_number}]\n{chunk.content}"
        )
    return "\n\n---\n\n".join(blocks)


def generate_answer(question: str, chunks: List[RetrievedChunk]) -> GroundedAnswer:
    context = format_context(chunks)

    user_message = f"""Context chunks:

{context}

---

Question: {question}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer_text = response.content[0].text
    sources_used = sorted({f"{c.source_file} (p.{c.page_number})" for c in chunks})

    return GroundedAnswer(answer=answer_text, sources_used=sources_used)


if __name__ == "__main__":
    import sys
    from retriever import retrieve

    question = sys.argv[1] if len(sys.argv) > 1 else "What are the interconnection timeline requirement?"
    department = sys.argv[2] if len(sys.argv) > 2 else None
    allowed = [department] if department else None

    chunks = retrieve(question, allowed_departments=allowed)
    result = generate_answer(question, chunks)

    print("Answer:\n", result.answer)
    print("\nChunks retrieved from:\n", "\n".join(result.sources_used))
