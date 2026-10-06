"""Search index tests: real ChromaDB and embedding model, no LLM.

The first run downloads the small embedding model (~80 MB).
"""

import dataclasses
import shutil

import chromadb
import pytest

from app import chat, knowledge, rag
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


# --- Retrieval quality on the real data ----------------------------------------------
# What the model gets for a question = people named in it + the entries closest
# in meaning + the three overview lists (always). Measured on 2026-10-07: search
# alone ranked the overviews 10th-21st for list questions and scored person
# entries 0.76-0.86 even when the question named the person.

OVERVIEWS = {"overview:themes", "overview:publications", "overview:people"}


def ids(docs):
    return [d["id"] for d in docs]


@pytest.mark.parametrize("query, expected_any", [
    ("I'm interested in AI agents that adapt based on student behaviour",
     {"publication:reimagining-student-success", "publication:static-to-dynamic-personalization"}),
    ("Find papers about agentic AI", {"theme:agentic-ai", "publication:human-layer-agentic-memory"}),
    ("What research does the lab do?", {"overview:themes"}),  # thanks to its question-shaped opening
])
def test_everyday_questions_find_the_right_records(query, expected_any):
    found = ids(rag.retrieve(query))
    assert expected_any & set(found), f"{query!r} retrieved {found}"


@pytest.mark.parametrize("query", [
    "Which papers has the lab published?", "Who works here?", "Write me a Python game"])
def test_overview_lists_are_always_in_the_prompt_context(query):
    context, _ = chat.gather_context(query)
    assert OVERVIEWS <= set(ids(context))


def test_healthcare_question_reaches_the_health_papers():
    context, _ = chat.gather_context("Does the lab have anything on healthcare?")
    text = "\n".join(d["text"] for d in context)
    assert "Deep Learning in Stroke Care" in text
    assert "malaria cell classification" in text


@pytest.mark.parametrize("query, person", [
    ("What has Walayat Hussain published?", "person:walayat-hussain"),
    ("what has hussain worked on", "person:walayat-hussain"),
    ("Tell me about Nazmul's research", "person:nazmul-hossain"),
])
def test_people_named_in_the_question_are_sources(query, person):
    _, sources = chat.gather_context(query)
    assert person in ids(sources)


def test_name_matching_does_not_confuse_hussain_and_hossain():
    assert ids(rag.people_named("What has Hossain published?")) == ["person:nazmul-hossain"]


def test_no_person_is_matched_without_a_name():
    assert rag.people_named("What research does the lab do?") == []


def test_overviews_are_listed_as_sources_only_when_search_finds_them():
    context, sources = chat.gather_context("bibliometric analysis of the Journal of Management")
    assert OVERVIEWS <= set(ids(context))
    assert not OVERVIEWS & set(ids(sources))


def test_special_characters_survive_indexing():
    hits = {h["id"]: h for h in rag.retrieve("bibliometric analysis of the Journal of Management")}
    jmo = hits["publication:jmo-bibliometric-analysis"]
    assert "Merigó, J. M." in jmo["text"]
    assert "215–266" in jmo["text"]
