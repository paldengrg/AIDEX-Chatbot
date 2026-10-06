"""Unit tests for the knowledge base loader and entry builder (no ChromaDB, no LLM)."""

import copy
import json
import re

import pytest

from app import knowledge
from app.config import settings

THEMES = [
    {"id": "agentic-ai", "number": 1, "name": "Agentic AI", "description": "Agents that adapt."},
    {"id": "learning", "number": 2, "name": "AI for Learning", "description": "AI in education."},
]
PUBLICATIONS = [
    {"id": "paper-a", "title": "Paper A", "authors": ["Hossain, M. N.", "Hussain, W."],
     "year": 2026, "type": "book_chapter", "venue": "Some Book.",
     "url": "https://doi.org/10.1/a", "themes": ["agentic-ai", "learning"],
     "abstract": "We study agents.", "abstract_source": "https://doi.org/10.1/a"},
    {"id": "paper-b", "title": "Paper B", "authors": ["Hussain, W."],
     "year": 2025, "type": "preprint", "venue": "arXiv.",
     "url": "https://arxiv.org/abs/1", "themes": ["learning"],
     "abstract": None, "abstract_source": None},
]
PEOPLE = [
    {"id": "director", "name": "Prof Director", "role": "Director", "bio": "Leads the lab.",
     "author_names": ["Hussain, W."], "links": {"profile": "https://example.edu/director"}},
    {"id": "student", "name": "PhD Student", "role": "PhD Candidate", "bio": "Studies agents.",
     "author_names": ["Hossain, M. N."], "links": {"scholar": "https://scholar.example/s"}},
]


def write_data(folder, themes=THEMES, publications=PUBLICATIONS, people=PEOPLE, note=None):
    """Write the three record files into `folder` and return it."""
    for name, items in (("themes.json", themes), ("publications.json", publications),
                        ("people.json", people)):
        data = {"retrieved_from": "https://example.org/", "retrieved_on": "2026-10-07",
                "items": items}
        if note and name == "people.json":
            data["note"] = note
        (folder / name).write_text(json.dumps(data), encoding="utf-8")
    return folder


@pytest.fixture
def data_dir(tmp_path):
    return write_data(tmp_path)


def broken(mutate):
    """Copies of the sample records after `mutate(themes, publications, people)`."""
    themes, pubs, people = (copy.deepcopy(THEMES), copy.deepcopy(PUBLICATIONS),
                            copy.deepcopy(PEOPLE))
    mutate(themes, pubs, people)
    return themes, pubs, people


# --- Loading and checking -------------------------------------------------------

def test_valid_data_loads(data_dir):
    kb = knowledge.load(data_dir)
    assert [t["id"] for t in kb.themes] == ["agentic-ai", "learning"]
    assert [p["id"] for p in kb.publications] == ["paper-a", "paper-b"]
    assert [p["id"] for p in kb.people] == ["director", "student"]
    assert kb.notes == {}


def test_optional_note_is_kept(tmp_path):
    write_data(tmp_path, note="More team members are coming.")
    assert knowledge.load(tmp_path).notes == {"people.json": "More team members are coming."}


@pytest.mark.parametrize("mutate, message", [
    (lambda t, p, h: p[0]["themes"].append("robotics"),
     "publications.json: paper-a: unknown theme id 'robotics'"),
    (lambda t, p, h: p.append(copy.deepcopy(p[0])),
     "publications.json: duplicate id 'paper-a'"),
    (lambda t, p, h: p[0].pop("title"),
     "publications.json: paper-a: 'title' is missing or not text"),
    (lambda t, p, h: p[0].update(year="2026"),
     "publications.json: paper-a: 'year' is missing or not a whole number"),
    (lambda t, p, h: p[0].update(year=True),
     "publications.json: paper-a: 'year' is missing or not a whole number"),
    (lambda t, p, h: p[0].update(type="blog_post"),
     "publications.json: paper-a: unknown type 'blog_post'"),
    (lambda t, p, h: p[0].update(url="javascript:alert(1)"),
     "publications.json: paper-a: links must start with https://"),
    (lambda t, p, h: p[1].update(abstract="Some text"),
     "publications.json: paper-b: 'abstract' and 'abstract_source' must both be set or both be null"),
    (lambda t, p, h: p[1].pop("abstract"),
     "publications.json: paper-b: 'abstract' and 'abstract_source' are required"),
    (lambda t, p, h: p[0].update(abstract_source="http://doi.org/10.1/a"),
     "publications.json: paper-a: links must start with https://"),
    (lambda t, p, h: h[0]["links"].update(profile="http://example.edu"),
     "people.json: director: links must start with https://"),
    (lambda t, p, h: h[0]["links"].update(twitter="https://x.com/a"),
     "people.json: director: unknown link 'twitter'"),
    (lambda t, p, h: h[0].update(author_names=[""]),
     "people.json: director: 'author_names' must be a list of non-empty strings"),
    (lambda t, p, h: t[0].update(name=""),
     "themes.json: agentic-ai: 'name' is missing or not text"),
])
def test_bad_data_is_rejected_with_a_clear_message(tmp_path, mutate, message):
    themes, pubs, people = broken(mutate)
    write_data(tmp_path, themes, pubs, people)
    with pytest.raises(knowledge.KnowledgeError, match=re.escape(message)):
        knowledge.load(tmp_path)


def test_missing_file_and_invalid_json_are_reported(tmp_path):
    with pytest.raises(knowledge.KnowledgeError, match="themes.json: file not found"):
        knowledge.load(tmp_path)
    write_data(tmp_path)
    (tmp_path / "people.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(knowledge.KnowledgeError, match="people.json: not valid JSON"):
        knowledge.load(tmp_path)


def test_items_list_is_required(tmp_path):
    write_data(tmp_path)
    (tmp_path / "themes.json").write_text('{"themes": []}', encoding="utf-8")
    with pytest.raises(knowledge.KnowledgeError, match='themes.json: expected an object with an "items" list'):
        knowledge.load(tmp_path)


# --- The real data files ------------------------------------------------------------

def test_real_data_files_pass_every_check():
    kb = knowledge.load(settings.data_dir)
    assert len(kb.themes) == 5
    assert len(kb.publications) == 11
    assert len(kb.people) == 2
    with_abstract = {p["id"] for p in kb.publications if p["abstract"]}
    assert with_abstract == {"jmo-bibliometric-analysis", "emfe-malaria",
                             "credibility-weighted-llm-mcdm"}


def test_real_data_keeps_accented_names_and_dashes():
    kb = knowledge.load(settings.data_dir)
    jmo = next(p for p in kb.publications if p["id"] == "jmo-bibliometric-analysis")
    assert "Merigó, J. M." in jmo["authors"]
    assert "215–266" in jmo["venue"]
    mcdm = next(p for p in kb.publications if p["id"] == "credibility-weighted-llm-mcdm")
    assert "LLM–MCDM" in mcdm["title"]
