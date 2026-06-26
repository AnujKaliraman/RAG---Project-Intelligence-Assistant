"""
retriever.py

Core retrieval logic: takes a user question and an optional department/role
filter, embeds the question, and runs a cosine-similarity nearest-neighbor
search against the document_chunks table in pgvector.

The department filtering happens at the SQL level, before results ever reach the generation step, so a user without engineering access never sees engineering chunks even as
retrieval candidates.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import psycopg2

import sys
sys.path.append("../ingestion")
from embedding_client import get_embedding  # noqa: E402


DB_CONFIG = {
    "host": os.environ.get("PGVECTOR_HOST", "localhost"),
    "port": os.environ.get("PGVECTOR_PORT", "5432"),
    "dbname": os.environ.get("PGVECTOR_DB", "postgres"),
    "user": os.environ.get("PGVECTOR_USER"),
    "password": os.environ.get("PGVECTOR_PASSWORD"),
}

TOP_K = 5


@dataclass
class RetrievedChunk:
    source_file: str
    page_number: int
    department: str
    content: str
    similarity: float  # higher = more similar (1 - cosine distance)


def retrieve(
    question: str,
    allowed_departments: Optional[List[str]] = None,
    top_k: int = TOP_K,
) -> List[RetrievedChunk]:
    """
    Retrieves the top_k most relevant chunks for a question.

    allowed_departments: if provided, restricts retrieval to chunks tagged
    with one of these departments - this is the permission-aware filter.
    If None, searches across all departments (e.g. for an admin role).
    """
    query_vector = get_embedding(question)

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    if allowed_departments:
        cur.execute(
            """
            SELECT source_file, page_number, department, content,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            WHERE department = ANY(%s)
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vector, allowed_departments, query_vector, top_k),
        )
    else:
        cur.execute(
            """
            SELECT source_file, page_number, department, content,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vector, query_vector, top_k),
        )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        RetrievedChunk(
            source_file=row[0],
            page_number=row[1],
            department=row[2],
            content=row[3],
            similarity=round(row[4], 4),
        )
        for row in rows
    ]


if __name__ == "__main__":
    # Quick manual test
    results = retrieve(
        "What are the interconnection timeline requirements?",
        allowed_departments=["engineering"],
    )
    for r in results:
        print(f"[{r.source_file} p.{r.page_number}] (sim={r.similarity}) {r.content[:120]}...")
