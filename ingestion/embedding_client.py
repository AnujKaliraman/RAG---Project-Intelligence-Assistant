"""
embedding_client.py

Thin wrapper around an embeddings API call, kept in its own module so the
underlying provider can be swapped without touching ingestion or retrieval
logic. (OpenAI, Cohere, Voyage AI, or a self-hosted sentence-transformers model all work
the same way from the rest of the pipeline's perspective).
"""

import os
from typing import List

from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    vector = _model.encode(text, normalize_embeddings=True)
    return vector.tolist()
