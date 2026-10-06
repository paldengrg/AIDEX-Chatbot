"""Retrieval-Augmented Generation (RAG) over the lab's knowledge base.

How it works:
  1. INGEST: knowledge.py turns the data files into search entries (one per
     paper, person, theme and page section, plus overview lists). ChromaDB
     turns each entry into an embedding (a vector of numbers capturing its
     meaning) using a small local model, and stores it.
  2. RETRIEVE: for a question, ChromaDB embeds the question the same way and
     returns the entries whose vectors are closest in meaning.

The retrieved entries are pasted into the prompt so the model answers from the
lab's real content instead of guessing. This is the shared "domain knowledge"
store; per-user memory is kept separately (memory.py).

The index rebuilds itself on start-up whenever anything in data/ has changed.
To force a rebuild:   python -m app.rag
"""

from app import knowledge
from app.config import settings

COLLECTION_NAME = "aidx_documents"

# Entries further than this (cosine distance, 0 = identical meaning) are treated
# as irrelevant, so off-topic questions don't get random "sources".
MAX_DISTANCE = 0.75

_collection = None


def get_collection():
    """Open the persistent Chroma collection, (re)building it if missing or out of date."""
    global _collection
    if _collection is None:
        import chromadb  # imported lazily so pure helpers are testable alone

        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        try:
            existing = client.get_collection(COLLECTION_NAME)
        except Exception:  # not created yet
            existing = None
        stored = (existing.metadata or {}).get("fingerprint") if existing else None
        if existing is None or existing.count() == 0 or \
                stored != knowledge.fingerprint(settings.data_dir):
            ingest()
        else:
            _collection = existing
    return _collection


def ingest() -> int:
    """(Re)build the index from data/. Returns the number of entries.

    The data is loaded and checked *before* the old index is deleted, so a
    broken data file raises KnowledgeError and leaves the old index in place.
    """
    global _collection
    import chromadb

    entries = knowledge.build_entries(settings.data_dir)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    try:
        client.delete_collection(COLLECTION_NAME)  # start clean: no stale entries
    except Exception:
        pass  # collection did not exist yet
    _collection = client.create_collection(COLLECTION_NAME, metadata={
        "hnsw:space": "cosine",
        "fingerprint": knowledge.fingerprint(settings.data_dir),
    })
    if entries:
        _collection.add(
            ids=[e.id for e in entries],
            documents=[e.text for e in entries],
            metadatas=[{"kind": e.kind, "title": e.title, "url": e.url, "source": e.source}
                       for e in entries],
        )
    return len(entries)


def retrieve(query: str, k: int | None = None) -> list[dict]:
    """Return up to k relevant entries as dicts: text, kind, id, title, url, source, distance."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    result = collection.query(
        query_texts=[query], n_results=min(k or settings.rag_top_k, collection.count())
    )
    hits = []
    for entry_id, text, meta, dist in zip(result["ids"][0], result["documents"][0],
                                          result["metadatas"][0], result["distances"][0]):
        if dist <= MAX_DISTANCE:
            hits.append({"text": text, "kind": meta["kind"], "id": entry_id,
                         "title": meta["title"], "url": meta["url"],
                         "source": meta["source"], "distance": round(dist, 3)})
    return hits


if __name__ == "__main__":
    print(f"Indexed {ingest()} entries from {settings.data_dir}")
