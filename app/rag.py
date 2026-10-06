"""Retrieval-Augmented Generation (RAG) over the lab's knowledge base.

How it works:
  1. INGEST: knowledge.py turns the data files into search entries (one per
     paper, person, theme and page section, plus overview lists). ChromaDB
     turns each entry into an embedding (a vector of numbers capturing its
     meaning) using a small local model, and stores it.
  2. RETRIEVE: for a question, ChromaDB embeds the question the same way and
     returns the entries whose vectors are closest in meaning.
  3. Because the small embedding model is weak on names, acronyms and list
     questions, simple helpers back it up: people_named() and papers_named()
     find people and papers mentioned by name, and overviews() returns the
     lists of all themes, papers and people, which chat.py always gives to
     the model.

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

# Always given to the model: every theme, paper and person in one list each.
OVERVIEW_IDS = ["overview:themes", "overview:publications", "overview:people"]

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
            metadatas=[{"kind": e.kind, "title": e.title, "url": e.url, "source": e.source,
                        "role": e.role} for e in entries],
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
    return [_hit(entry_id, text, meta, round(dist, 3))
            for entry_id, text, meta, dist in zip(result["ids"][0], result["documents"][0],
                                                  result["metadatas"][0], result["distances"][0])
            if dist <= MAX_DISTANCE]


def overviews() -> list[dict]:
    """The three overview entries (all themes, all papers, all people).

    Search alone ranks these too low for list questions ("what has the lab
    published?"), so chat.py always gives them to the model.
    """
    result = get_collection().get(ids=OVERVIEW_IDS, include=["documents", "metadatas"])
    found = {i: (text, meta) for i, text, meta
             in zip(result["ids"], result["documents"], result["metadatas"])}
    return [_hit(i, *found[i], None) for i in OVERVIEW_IDS if i in found]


def people_named(query: str) -> list[dict]:
    """Person entries the query names ("Hussain", "Nazmul") or asks about by
    role ("the director", "PhD candidates").

    The embedding model is weak on names, so a question naming someone could
    otherwise miss their entry entirely.
    """
    asked, lowered = knowledge.words(query), query.lower()
    return [hit for hit, meta in _entries_of_kind("person")
            if knowledge.name_tokens(meta["title"]) & asked
            or (meta.get("role") and meta["role"].lower() in lowered)]


def papers_named(query: str) -> list[dict]:
    """Publication entries the query names by acronym ("EMFE") or title words."""
    papers = _entries_of_kind("publication")
    named = set(knowledge.papers_named(query, {hit["id"]: hit["title"] for hit, _ in papers}))
    return [hit for hit, _ in papers if hit["id"] in named]


def _entries_of_kind(kind: str) -> list[tuple[dict, dict]]:
    """Every entry of one kind, as (hit, metadata) pairs."""
    result = get_collection().get(where={"kind": kind}, include=["documents", "metadatas"])
    return [(_hit(i, text, meta, None), meta) for i, text, meta
            in zip(result["ids"], result["documents"], result["metadatas"])]


def _hit(entry_id: str, text: str, meta: dict, distance: float | None) -> dict:
    """One retrieved entry; distance is None when it was not found by meaning."""
    return {"text": text, "kind": meta["kind"], "id": entry_id, "title": meta["title"],
            "url": meta["url"], "source": meta["source"], "distance": distance}


if __name__ == "__main__":
    print(f"Indexed {ingest()} entries from {settings.data_dir}")
