"""RAG Index Builder for LedgerLock TDS Tax Rules.

Chunks and indexes rag/tds_rules.md into a clean local ChromaDB collection: 'tds_rules'.
Extracts exact statutory rates for Sections 194-O, 194H, 194C, and 194J.
"""

from __future__ import annotations

import os
import re
import chromadb
from typing import List, Dict, Any

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag", "chroma_db")
DOC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tds_rules.md")

# Definitive section rate mapping for statutory Indian Tax Act
SECTION_STATUTORY_RATES = {
    "194-O": 0.01,
    "194H": 0.05,
    "194C": 0.02,
    "194J": 0.10,
}


def chunk_markdown(content: str) -> List[Dict[str, Any]]:
    """Chunk markdown into section-level knowledge items with precise statutory metadata."""
    sections = re.split(r"\n(?=## Section )", content)
    chunks = []

    for idx, sec in enumerate(sections):
        sec = sec.strip()
        if not sec:
            continue

        sec_match = re.search(r"## Section ([\w\-]+)", sec)
        sec_id = sec_match.group(1).upper() if sec_match else f"GEN_{idx}"

        # Match statutory rate from mapping or regex
        rate = SECTION_STATUTORY_RATES.get(sec_id)
        if rate is None:
            rate_match = re.search(r"([\d\.]+)%\s*(\([\d\.]+\))?", sec)
            rate = float(rate_match.group(1)) / 100.0 if rate_match else 0.01

        chunks.append({
            "id": f"tds_{sec_id.lower()}",
            "text": sec,
            "metadata": {
                "section": sec_id,
                "statutory_rate": rate,
                "source": "tds_rules.md",
            },
        })

    return chunks


def build_tds_index(persist_dir: str = CHROMA_DIR) -> chromadb.Collection:
    """Build or refresh ChromaDB index with TDS knowledge base cleanly."""
    os.makedirs(persist_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_dir)

    # Delete existing collection to avoid stale chunk collisions
    try:
        client.delete_collection(name="tds_rules")
    except Exception:
        pass

    try:
        from chromadb.utils import embedding_functions
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        collection = client.create_collection(
            name="tds_rules",
            embedding_function=ef,
        )
    except Exception as e:
        print(f"Notice: SentenceTransformer embedding function fallback ({e}). Using default Chroma embeddings.")
        collection = client.create_collection(name="tds_rules")

    with open(DOC_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = chunk_markdown(content)

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"Successfully created clean index with {len(chunks)} TDS rule sections in ChromaDB:")
    for c in chunks:
        print(f"  - {c['metadata']['section']}: statutory rate = {c['metadata']['statutory_rate']*100:.1f}%")
    return collection


if __name__ == "__main__":
    build_tds_index()
