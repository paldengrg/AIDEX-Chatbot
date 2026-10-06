"""Search index tests: real ChromaDB and embedding model, no LLM.

The first run downloads the small embedding model (~80 MB).
"""

import dataclasses
import shutil

import chromadb
import pytest

from app import knowledge, rag
from app.config import settings


@pytest.fixture
def temp_index(tmp_path, monkeypatch):
    """Point rag at a copy of the record files plus one page, with an empty index folder."""
    data = tmp_path / "data"
    data.mkdir()
    for path in settings.data_dir.glob("*.json"):
        shutil.copy(path, data / path.name)
    (data / "news.md").write_text("# News\n\n## Seminar\nA seminar on decision support.\n",
                                  encoding="utf-8")
    monkeypatch.setattr(rag, "settings", dataclasses.replace(
        settings, data_dir=data, chroma_dir=tmp_path / "chroma"))
    monkeypatch.setattr(rag, "_collection", None)
    return data


def test_index_is_rebuilt_only_when_data_changes(temp_index, monkeypatch):
    rag.get_collection()  # first start: builds the index
    calls = []
    real_ingest = rag.ingest
    monkeypatch.setattr(rag, "ingest", lambda: calls.append(1) or real_ingest())

    monkeypatch.setattr(rag, "_collection", None)  # restart, data unchanged
    rag.get_collection()
    assert calls == []

    (temp_index / "news.md").write_text(
        "# News\n\n## Seminar\nA seminar on quantum gardening.\n", encoding="utf-8")
    monkeypatch.setattr(rag, "_collection", None)  # restart after an edit
    rag.get_collection()
    assert calls == [1]
    assert any("quantum gardening" in h["text"] for h in rag.retrieve("quantum gardening seminar"))


def test_bad_data_edit_keeps_the_old_index(temp_index):
    count = rag.get_collection().count()
    (temp_index / "themes.json").write_text("{ broken", encoding="utf-8")
    with pytest.raises(knowledge.KnowledgeError, match="themes.json: not valid JSON"):
        rag.ingest()
    client = chromadb.PersistentClient(path=str(rag.settings.chroma_dir))
    assert client.get_collection(rag.COLLECTION_NAME).count() == count


def test_hits_carry_kind_title_and_https_link(temp_index):
    hits = rag.retrieve("Who is the director of the lab?")
    assert hits
    for h in hits:
        assert set(h) == {"text", "kind", "id", "title", "url", "source", "distance"}
        assert h["url"].startswith("https://")
    # The people overview names the Director; the cut-off is tuned in the quality tests below.
    assert {"overview:people", "person:walayat-hussain"} & {h["id"] for h in hits}
