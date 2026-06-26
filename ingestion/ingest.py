"""
ingest.py

Loads source documents (PDFs), splits them into overlapping text chunks,
and tags each chunk with the metadata needed for citation later:
source filename, page number, and a mock department/access tag.

It produces a list of chunk dicts ready to be embedded and loaded into pgvector (see embed_and_load.py).
"""

import os
import re
from dataclasses import dataclass, asdict
from typing import List

import pypdf


# --- Config -----------------------------------------------------------------

DATA_DIR = "../data/sample_docs"
CHUNK_SIZE_WORDS = 220       # ~ roughly 300-400 tokens depending on content
CHUNK_OVERLAP_WORDS = 40     # overlap so we don't split a relevant sentence across chunks

# Mock department/access tags per source file.
# In a real deployment this would come from a document management system
# (SharePoint metadata, folder permissions, etc.) rather than a hardcoded map.
DEPARTMENT_MAP = {
    "ercot_interconnection_handbook.pdf": "engineering",
    "ercot_large_load_qa.pdf": "engineering",
    "rioo_interconnection_guide.pdf": "engineering",
    "om_procedures_manual.pdf": "engineering",
    "epc_construction_schedule.pdf": "engineering",
    "vendor_pricing_summary.pdf": "finance",
    "procurement_terms.pdf": "procurement",
    "sample_ppa_term_sheet.pdf": "legal",
}


@dataclass
class Chunk:
    source_file: str
    page_number: int
    department: str
    chunk_index: int
    content: str


def load_pdf_pages(filepath: str) -> List[str]:
    """Returns a list of page texts, one string per page."""
    reader = pypdf.PdfReader(filepath)
    return [page.extract_text() or "" for page in reader.pages]


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_page_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Splits a page's text into overlapping word-count windows."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def ingest_document(filepath: str) -> List[Chunk]:
    filename = os.path.basename(filepath)
    department = DEPARTMENT_MAP.get(filename, "general")

    pages = load_pdf_pages(filepath)
    all_chunks: List[Chunk] = []

    for page_num, page_text in enumerate(pages, start=1):
        cleaned = clean_text(page_text)
        if not cleaned:
            continue

        page_chunks = chunk_page_text(cleaned, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
        for idx, chunk_text in enumerate(page_chunks):
            all_chunks.append(
                Chunk(
                    source_file=filename,
                    page_number=page_num,
                    department=department,
                    chunk_index=idx,
                    content=chunk_text,
                )
            )

    return all_chunks


def ingest_all_documents(data_dir: str = DATA_DIR) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            print(f"Ingesting {filename}...")
            chunks = ingest_document(filepath)
            print(f"  -> {len(chunks)} chunks")
            all_chunks.extend(chunks)
    return all_chunks


if __name__ == "__main__":
    chunks = ingest_all_documents()
    print(f"\nTotal chunks across all documents: {len(chunks)}")

    # Quick peek at the first chunk to sanity-check
    if chunks:
        print("\nSample chunk:")
        print(asdict(chunks[0]))
