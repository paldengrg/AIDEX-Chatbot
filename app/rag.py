"""Retrieval-Augmented Generation (RAG) over the lab's documents.

How it works:
  1. INGEST: every markdown file in data/ is split into small chunks (one per
     heading section). ChromaDB turns each chunk into an embedding (a vector of
     numbers capturing its meaning) using a small local model, and stores it.
  2. RETRIEVE: for a question, ChromaDB embeds the question the same way and
     returns the chunks whose vectors are closest in meaning.

The retrieved chunks are pasted into the prompt so the model answers from the
lab's real content instead of guessing. This is the shared "domain knowledge"
store; per-user memory is kept separately (from Phase 2).

Rebuild the index after editing data/:   python -m app.rag
"""

import re
from dataclasses import dataclass
from pathlib import Path

from app.config import settings

COLLECTION_NAME = "aidx_documents"

# Chunks further than this (cosine distance, 0 = identical meaning) are treated
# as irrelevant, so off-topic questions don't get random "sources".
MAX_DISTANCE = 0.75


@dataclass
class Chunk:
    id: str        # unique id, e.g. "people.md#2"
    source: str    # file name, shown to users as the source
    heading: str   # section heading
    text: str      # text that gets embedded and shown to the model


# --- Chunking (pure Python, easy to unit test) -------------------------------

def chunk_markdown(text: str, source: str) -> list[Chunk]:
    """Split a markdown document into one chunk per '#'/'##' section.

    The document title is prefixed to every chunk, so a section like
    "## Contact" still carries its context ("Join the lab and contact us").
    """
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)  # drop editor notes

    title = ""
    chunks: list[Chunk] = []
    heading, lines = "", []

    def flush():
        body = "\n".join(lines).strip()
        if body:
            label = f"{title} > {heading}" if heading and heading != title else title
            chunks.append(Chunk(
                id=f"{source}#{len(chunks)}",
                source=source,
                heading=heading or title,
                text=f"{label}\n{body}" if label else body,
            ))

    for line in text.splitlines():
        match = re.match(r"^(#{1,2})\s+(.*)", line)
        if match:
            flush()
            lines = []
            heading = match.group(2).strip()
            if match.group(1) == "#":
                title = heading
        else:
            lines.append(line)
    flush()
    return chunks


def load_chunks(data_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(data_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), path.name))
    return chunks


# --- ChromaDB ----------------------------------------------------------------

_collection = None


def get_collection():
    """Open (or create) the persistent Chroma collection, ingesting if empty."""
    global _collection
    if _collection is None:
        import chromadb  # imported lazily so pure helpers are testable alone

        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        _collection = client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        if _collection.count() == 0:
            ingest()
    return _collection


def ingest() -> int:
    """(Re)build the index from data/*.md. Returns the number of chunks."""
    global _collection
    import chromadb

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    try:
        client.delete_collection(COLLECTION_NAME)  # start clean: no stale chunks
    except Exception:
        pass  # collection did not exist yet
    _collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    chunks = load_chunks(settings.data_dir)
    if chunks:
        _collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "heading": c.heading} for c in chunks],
        )
    return len(chunks)


def retrieve(query: str, k: int | None = None) -> list[dict]:
    """Return up to k relevant chunks as dicts: text, source, heading, distance."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    result = collection.query(
        query_texts=[query], n_results=min(k or settings.rag_top_k, collection.count())
    )
    hits = []
    for text, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        if dist <= MAX_DISTANCE:
            hits.append({"text": text, "source": meta["source"],
                         "heading": meta["heading"], "distance": round(dist, 3)})
    return hits


if __name__ == "__main__":
    print(f"Indexed {ingest()} chunks from {settings.data_dir}")
