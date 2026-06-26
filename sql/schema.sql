-- Source-Linked RAG: Document Chunks Schema
-- Mirrors the embeddings table pattern from the streaming recommender project,
-- applied here to text chunks instead of product/user embeddings.

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS document_chunks;

CREATE TABLE document_chunks (
    id              SERIAL PRIMARY KEY,
    source_file     TEXT NOT NULL,          -- e.g. "ercot_interconnection_guide.pdf"
    page_number     INT NOT NULL,           -- page the chunk came from, for citation
    department      TEXT NOT NULL,          -- mock access tag: engineering | legal | finance | procurement
    chunk_index     INT NOT NULL,           -- position of chunk within the source doc
    content         TEXT NOT NULL,          -- the actual chunk text
    embedding       VECTOR(384) NOT NULL,  -- adjust dimension to match embedding model used
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Index for fast approximate nearest-neighbor search.
-- IVFFlat is a reasonable choice at this small scale; HNSW is preferable at larger scale.
CREATE INDEX ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);

-- Helpful index for permission-aware filtering before/alongside the similarity search
CREATE INDEX ON document_chunks (department);

-- Quick sanity check query after loading data:
-- SELECT source_file, department, COUNT(*) FROM document_chunks GROUP BY source_file, department;
