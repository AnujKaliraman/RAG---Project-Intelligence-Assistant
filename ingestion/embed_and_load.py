"""
embed_and_load.py

Takes the chunks produced by ingest.py, generates embeddings for each,
and loads them into the PostgreSQL + pgvector `document_chunks` table.

Mirrors the embeddings-load pattern from the streaming recommender project
(item_emb / user_emb tables), applied here to text chunks instead of
product/user vectors.
"""

import os
from typing import List

import psycopg2
from psycopg2.extras import execute_values

from ingest import ingest_all_documents, Chunk

# Use any embedding model you have API access to. Example shown uses
# OpenAI-style embeddings API; swap for your provider of choice.
# (Anthropic does not currently offer a first-party embeddings endpoint,
# so this step typically calls out to an embeddings-specific provider.)
from embedding_client import get_embedding  # thin wrapper, see embedding_client.py


DB_CONFIG = {
    "host": os.environ.get("PGVECTOR_HOST", "localhost"),
    "port": os.environ.get("PGVECTOR_PORT", "5432"),
    "dbname": os.environ.get("PGVECTOR_DB", "postgres"),
    "user": os.environ.get("PGVECTOR_USER"),
    "password": os.environ.get("PGVECTOR_PASSWORD"),
    "sslmode": "require",
}


def embed_chunks(chunks: List[Chunk]):
    """Yields (chunk, embedding_vector) pairs."""
    for chunk in chunks:
        vector = get_embedding(chunk.content)
        yield chunk, vector


def load_into_pgvector(chunks_with_embeddings):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    rows = [
        (
            chunk.source_file,
            chunk.page_number,
            chunk.department,
            chunk.chunk_index,
            chunk.content,
            vector,
        )
        for chunk, vector in chunks_with_embeddings
    ]

    execute_values(
        cur,
        """
        INSERT INTO document_chunks
            (source_file, page_number, department, chunk_index, content, embedding)
        VALUES %s
        """,
        rows,
    )

    conn.commit()
    cur.close()
    conn.close()
    print(f"Loaded {len(rows)} chunks into document_chunks.")


if __name__ == "__main__":
    print("Ingesting documents...")
    chunks = ingest_all_documents()

    print(f"Embedding {len(chunks)} chunks (this calls the embeddings API once per chunk)...")
    pairs = list(embed_chunks(chunks))

    print("Loading into pgvector...")
    load_into_pgvector(pairs)
