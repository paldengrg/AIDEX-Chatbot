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


# --- Relationships ------------------------------------------------------------------

def test_person_is_matched_to_exactly_their_papers(data_dir):
    kb = knowledge.load(data_dir)
    director, student = kb.people
    assert [p["id"] for p in knowledge.papers_by(kb, director)] == ["paper-a", "paper-b"]
    assert [p["id"] for p in knowledge.papers_by(kb, student)] == ["paper-a"]


def test_person_themes_come_from_their_papers_most_frequent_first(data_dir):
    kb = knowledge.load(data_dir)
    director, student = kb.people
    # Director: learning is on 2 papers, agentic on 1.
    assert knowledge.person_themes(kb, director) == ["AI for Learning", "Agentic AI"]
    # Student: one paper with both themes, so ties keep theme order.
    assert knowledge.person_themes(kb, student) == ["Agentic AI", "AI for Learning"]


@pytest.mark.parametrize("name, tokens", [
    ("Associate Professor Walayat Hussain", {"walayat", "hussain"}),  # titles ignored
    ("Md Nazmul Hossain", {"nazmul", "hossain"}),                      # short words ignored
    ("Dr Elena León-Castro", {"elena", "león", "castro"}),             # accents kept
])
def test_name_tokens_are_the_distinctive_words_of_a_name(name, tokens):
    assert knowledge.name_tokens(name) == tokens


# --- Search entries -------------------------------------------------------------------

def entries_by_id(folder):
    return {e.id: e for e in knowledge.build_entries(folder)}


def test_publication_entry_starts_with_title_and_themes(data_dir):
    e = entries_by_id(data_dir)["publication:paper-a"]
    lines = e.text.splitlines()
    assert lines[0] == "Publication: Paper A"
    assert lines[1] == "Research themes: Agentic AI, AI for Learning"
    assert lines[2] == "Authors: Hossain, M. N.; Hussain, W."
    assert lines[3] == "Book chapter, 2026. Some Book."
    assert lines[4] == "Abstract: We study agents."
    assert (e.kind, e.title, e.url, e.source) == (
        "publication", "Paper A", "https://doi.org/10.1/a", "publications.json")


def test_edited_book_lists_editors(tmp_path):
    pubs = copy.deepcopy(PUBLICATIONS)
    pubs[1]["type"] = "edited_book"
    write_data(tmp_path, publications=pubs)
    assert "Editors: Hussain, W." in entries_by_id(tmp_path)["publication:paper-b"].text


def test_paper_without_abstract_is_marked_citation_only(data_dir):
    text = entries_by_id(data_dir)["publication:paper-b"].text
    assert text.splitlines()[-1] == "Citation only: no abstract available."


def test_person_entry_lists_all_their_papers_and_links_best_profile(data_dir):
    entries = entries_by_id(data_dir)
    director = entries["person:director"]
    # A question-shaped opening line, so "what has X published?" finds it.
    assert director.text.startswith("Publications and research by Prof Director\n"
                                    "Prof Director, Director\n")
    assert "Research themes (from their publications): AI for Learning, Agentic AI" in director.text
    assert "Publications (2):\n- Paper A (2026)\n- Paper B (2025)" in director.text
    assert director.url == "https://example.edu/director"           # profile first
    assert entries["person:student"].url == "https://scholar.example/s"  # then scholar


def test_person_without_papers_or_links_still_builds(tmp_path):
    newcomer = {"id": "new", "name": "New Member", "role": "Research assistant",
                "bio": "Joined recently.", "author_names": ["Nobody, X."], "links": {}}
    write_data(tmp_path, people=[*PEOPLE, newcomer])
    e = entries_by_id(tmp_path)["person:new"]
    assert "No publications listed yet." in e.text
    assert "Research themes" not in e.text
    assert e.url == knowledge.SITE_URL + "#people"


def test_theme_entry_lists_its_papers_and_people(data_dir):
    e = entries_by_id(data_dir)["theme:agentic-ai"]
    assert e.text.startswith("Research theme 1: Agentic AI\nAgents that adapt.")
    assert "Publications in this theme (1):\n- Paper A (2026)" in e.text
    assert "Paper B" not in e.text
    assert "Lab members with publications in this theme: Prof Director, PhD Student" in e.text
    assert (e.kind, e.url, e.source) == ("theme", knowledge.SITE_URL + "#research", "themes.json")


def test_overviews_list_everything(tmp_path):
    write_data(tmp_path, note="More team members are coming.")
    entries = entries_by_id(tmp_path)
    pubs = entries["overview:publications"].text
    assert pubs.startswith("What has the AIDEX Lab published? All 2 publications:")
    assert "Book chapters:\n- Paper A (2026)" in pubs
    assert "Preprints:\n- Paper B (2025)" in pubs
    people = entries["overview:people"].text
    assert people.startswith("Who works at the AIDEX Lab? People (2 listed):")
    assert "- Prof Director: Director\n- PhD Student: PhD Candidate" in people
    assert people.endswith("More team members are coming.")
    themes = entries["overview:themes"].text
    assert themes.startswith("What does the AIDEX Lab research? The lab's 2 research themes:")
    assert "1. Agentic AI: Agents that adapt.\n2. AI for Learning: AI in education." in themes


def test_markdown_pages_become_page_entries_with_site_links(data_dir):
    (data_dir / "about.md").write_text(
        "<!-- editor note -->\n# About\n\n## Mission\nBetter decisions.\n", encoding="utf-8")
    (data_dir / "extra.md").write_text("# Extra\n\nSome text.\n", encoding="utf-8")
    entries = entries_by_id(data_dir)
    about = entries["about.md#0"]
    assert (about.kind, about.title, about.url, about.source) == (
        "page", "Mission", knowledge.SITE_URL + "#about", "about.md")
    assert about.text == "About > Mission\nBetter decisions."
    assert entries["extra.md#0"].url == knowledge.SITE_URL  # unknown page -> site root


def test_real_data_entries_are_unique_linked_and_complete():
    entries = knowledge.build_entries(settings.data_dir)
    assert len({e.id for e in entries}) == len(entries)
    assert all(e.url.startswith("https://") for e in entries)
    assert {e.kind for e in entries} == {"publication", "person", "theme", "overview", "page"}
    walayat = next(e for e in entries if e.id == "person:walayat-hussain")
    assert "Publications (11):" in walayat.text


# --- Fingerprint ------------------------------------------------------------------

def test_fingerprint_changes_only_when_data_changes(data_dir):
    (data_dir / "news.md").write_text("# News\n\n## Event\nA boot camp.\n", encoding="utf-8")
    before = knowledge.fingerprint(data_dir)
    assert knowledge.fingerprint(data_dir) == before
    (data_dir / "news.md").write_text("# News\n\n## Event\nTwo boot camps.\n", encoding="utf-8")
    assert knowledge.fingerprint(data_dir) != before


def test_fingerprint_changes_with_index_version(data_dir, monkeypatch):
    before = knowledge.fingerprint(data_dir)
    monkeypatch.setattr(knowledge, "INDEX_VERSION", "test")
    assert knowledge.fingerprint(data_dir) != before
