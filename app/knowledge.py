"""The lab's knowledge base: load the data files and turn them into search entries.

data/ holds two kinds of content:
  - Records (JSON): themes.json, publications.json and people.json, one item per
    research theme, paper and person, each with a link to where it came from.
  - Pages (markdown): about.md, join-and-contact.md, news.md, ...

load() reads and checks the records, so a typo in a data file stops the app
with a clear message instead of quietly weakening the answers.

build_entries() turns records and pages into a flat list of Entry objects;
rag.py stores each one in the search index. Besides one entry per record it
adds three "overview" entries (all papers, all people, all themes), so list
questions such as "what has the lab published?" get a complete answer instead
of only the few records that happen to be closest to the question.

Pure Python (no ChromaDB), so everything here is quick to unit test.
"""

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

SITE_URL = "https://aidxlab.github.io/"

# Bump this when the way entries are built changes: the index is then rebuilt
# on the next start even though no data file changed.
INDEX_VERSION = "2"

# Where each markdown page lives on the lab website (its citation link).
PAGE_URLS = {
    "about.md": SITE_URL + "#about",
    "join-and-contact.md": SITE_URL + "#join",
    "news.md": SITE_URL + "#news",
    "human-layer-paper.md": "https://doi.org/10.1007/978-3-032-31204-4_3",
}

# type id -> (singular label, plural label)
PUBLICATION_TYPES = {
    "journal_article": ("Journal article", "Journal articles"),
    "edited_book": ("Edited book", "Edited books"),
    "book_chapter": ("Book chapter", "Book chapters"),
    "conference_paper": ("Conference paper", "Conference papers"),
    "preprint": ("Preprint", "Preprints"),
}

LINK_KINDS = ("profile", "scholar", "linkedin")

# Required fields of each record file and their types.
SCHEMAS = {
    "themes.json": {"id": str, "number": int, "name": str, "description": str},
    "publications.json": {"id": str, "title": str, "authors": list, "year": int,
                          "type": str, "venue": str, "url": str, "themes": list},
    "people.json": {"id": str, "name": str, "role": str, "bio": str,
                    "author_names": list, "links": dict},
}

# How a type is named in error messages.
TYPE_NAMES = {str: "text", int: "a whole number", list: "a list", dict: "an object"}


class KnowledgeError(ValueError):
    """A data file is missing, malformed or inconsistent."""


@dataclass
class Entry:
    """One item in the search index."""
    id: str      # unique, e.g. "publication:emfe-malaria" or "about.md#0"
    kind: str    # "publication" | "person" | "theme" | "overview" | "page"
    title: str   # shown to users as the source's name
    url: str     # citation link, always https://
    source: str  # data file it came from, e.g. "publications.json"
    text: str    # embedded for search and shown to the model


@dataclass
class Knowledge:
    themes: list[dict]
    publications: list[dict]
    people: list[dict]
    notes: dict[str, str] = field(default_factory=dict)  # file name -> its "note"


# --- Loading and checking ----------------------------------------------------

def load(data_dir: Path) -> Knowledge:
    """Read and check the three record files. Raises KnowledgeError on any problem."""
    items, notes = {}, {}
    for name, schema in SCHEMAS.items():
        data = _read(data_dir / name)
        _check_items(name, data["items"], schema)
        items[name] = data["items"]
        if isinstance(data.get("note"), str) and data["note"].strip():
            notes[name] = data["note"].strip()
    kb = Knowledge(items["themes.json"], items["publications.json"], items["people.json"], notes)
    _check_publications(kb)
    _check_people(kb)
    return kb


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise KnowledgeError(f"{path.name}: file not found in {path.parent}") from None
    except json.JSONDecodeError as exc:
        raise KnowledgeError(f"{path.name}: not valid JSON ({exc})") from None
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise KnowledgeError(f'{path.name}: expected an object with an "items" list')
    return data


def _check_items(name: str, items: list, schema: dict) -> None:
    seen = set()
    for n, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise KnowledgeError(f"{name}: item {n} is not an object")
        label = f"{name}: {item.get('id') or f'item {n}'}"
        for key, kind in schema.items():
            value = item.get(key)
            # bool counts as int in Python, so it is rejected explicitly.
            if not isinstance(value, kind) or isinstance(value, bool) or value in ("", []):
                raise KnowledgeError(f"{label}: '{key}' is missing or not {TYPE_NAMES[kind]}")
            if kind is list and not all(isinstance(v, str) and v.strip() for v in value):
                raise KnowledgeError(f"{label}: '{key}' must be a list of non-empty strings")
        if item["id"] in seen:
            raise KnowledgeError(f"{name}: duplicate id '{item['id']}'")
        seen.add(item["id"])


def _check_url(label: str, url) -> None:
    if not isinstance(url, str) or not url.startswith("https://"):
        raise KnowledgeError(f"{label}: links must start with https:// (got {url!r})")


def _check_publications(kb: Knowledge) -> None:
    theme_ids = {t["id"] for t in kb.themes}
    for p in kb.publications:
        label = f"publications.json: {p['id']}"
        if p["type"] not in PUBLICATION_TYPES:
            raise KnowledgeError(f"{label}: unknown type '{p['type']}' "
                                 f"(allowed: {', '.join(PUBLICATION_TYPES)})")
        for theme in p["themes"]:
            if theme not in theme_ids:
                raise KnowledgeError(f"{label}: unknown theme id '{theme}'")
        _check_url(label, p["url"])
        if "abstract" not in p or "abstract_source" not in p:
            raise KnowledgeError(f"{label}: 'abstract' and 'abstract_source' are required "
                                 "(use null when there is no official abstract)")
        abstract, source = p["abstract"], p["abstract_source"]
        if (abstract is None) != (source is None):
            raise KnowledgeError(f"{label}: 'abstract' and 'abstract_source' must both be set "
                                 "or both be null")
        if abstract is not None:
            if not isinstance(abstract, str) or not abstract.strip():
                raise KnowledgeError(f"{label}: 'abstract' must be text or null")
            _check_url(label, source)


def _check_people(kb: Knowledge) -> None:
    for person in kb.people:
        label = f"people.json: {person['id']}"
        for kind, url in person["links"].items():
            if kind not in LINK_KINDS:
                raise KnowledgeError(f"{label}: unknown link '{kind}' "
                                     f"(allowed: {', '.join(LINK_KINDS)})")
            _check_url(label, url)


# --- Relationships worked out from the records -------------------------------

def papers_by(kb: Knowledge, person: dict) -> list[dict]:
    """The person's publications: those listing one of their author_names."""
    names = set(person["author_names"])
    return [p for p in kb.publications if names & set(p["authors"])]


def _theme_names(kb: Knowledge, theme_ids) -> list[str]:
    names = {t["id"]: t["name"] for t in kb.themes}
    return [names[t] for t in theme_ids]


def person_themes(kb: Knowledge, person: dict) -> list[str]:
    """Theme names of the person's papers, most frequent first (ties: theme number)."""
    counts = Counter(t for p in papers_by(kb, person) for t in p["themes"])
    number = {t["id"]: t["number"] for t in kb.themes}
    return _theme_names(kb, sorted(counts, key=lambda t: (-counts[t], number[t])))


# --- Search entries ------------------------------------------------------------
# Titles, theme names and headings come first in every entry: the embedding
# model only reads about the first 200 words, so that is what search matches.

def _paper_line(p: dict) -> str:
    return f"- {p['title']} ({p['year']})"


def _publication_entry(kb: Knowledge, p: dict) -> Entry:
    singular, _ = PUBLICATION_TYPES[p["type"]]
    who = "Editors" if p["type"] == "edited_book" else "Authors"
    abstract = (f"Abstract: {p['abstract']}" if p["abstract"]
                else "Citation only: no abstract available.")
    text = "\n".join([
        f"Publication: {p['title']}",
        f"Research themes: {', '.join(_theme_names(kb, p['themes']))}",
        f"{who}: {'; '.join(p['authors'])}",
        f"{singular}, {p['year']}. {p['venue']}",
        abstract,
    ])
    return Entry(f"publication:{p['id']}", "publication", p["title"], p["url"],
                 "publications.json", text)


def _person_entry(kb: Knowledge, person: dict) -> Entry:
    papers = papers_by(kb, person)
    themes = person_themes(kb, person)
    lines = [f"{person['name']}, {person['role']}"]
    if themes:
        lines.append(f"Research themes (from their publications): {', '.join(themes)}")
    lines.append(person["bio"])
    if papers:
        lines.append(f"Publications ({len(papers)}):")
        lines.extend(_paper_line(p) for p in papers)
    else:
        lines.append("No publications listed yet.")
    links = person["links"]
    url = links.get("profile") or links.get("scholar") or SITE_URL + "#people"
    return Entry(f"person:{person['id']}", "person", person["name"], url, "people.json",
                 "\n".join(lines))


def _theme_entry(kb: Knowledge, theme: dict) -> Entry:
    papers = [p for p in kb.publications if theme["id"] in p["themes"]]
    paper_ids = {p["id"] for p in papers}
    people = [person["name"] for person in kb.people
              if any(p["id"] in paper_ids for p in papers_by(kb, person))]
    lines = [f"Research theme {theme['number']}: {theme['name']}", theme["description"]]
    if papers:
        lines.append(f"Publications in this theme ({len(papers)}):")
        lines.extend(_paper_line(p) for p in papers)
    if people:
        lines.append(f"Lab members with publications in this theme: {', '.join(people)}")
    return Entry(f"theme:{theme['id']}", "theme", theme["name"], SITE_URL + "#research",
                 "themes.json", "\n".join(lines))


def _overview_entries(kb: Knowledge) -> list[Entry]:
    pubs = [f"AIDEX Lab publications ({len(kb.publications)} in total)"]
    for type_id, (_, plural) in PUBLICATION_TYPES.items():
        group = [p for p in kb.publications if p["type"] == type_id]
        if group:
            pubs.append(f"{plural}:")
            pubs.extend(_paper_line(p) for p in group)
    people = [f"AIDEX Lab people ({len(kb.people)} listed)"]
    people.extend(f"- {p['name']}: {p['role']}" for p in kb.people)
    themes = [f"AIDEX Lab research themes ({len(kb.themes)})"]
    themes.extend(f"{t['number']}. {t['name']}: {t['description']}"
                  for t in sorted(kb.themes, key=lambda t: t["number"]))
    for lines, name in ((pubs, "publications.json"), (people, "people.json"),
                        (themes, "themes.json")):
        if name in kb.notes:
            lines.append(kb.notes[name])
    return [
        Entry("overview:publications", "overview", "All publications",
              SITE_URL + "#publications", "publications.json", "\n".join(pubs)),
        Entry("overview:people", "overview", "All people",
              SITE_URL + "#people", "people.json", "\n".join(people)),
        Entry("overview:themes", "overview", "All research themes",
              SITE_URL + "#research", "themes.json", "\n".join(themes)),
    ]


def chunk_markdown(text: str, source: str) -> list[Entry]:
    """Split a markdown page into one entry per '#'/'##' section.

    The page title is prefixed to every section, so a section like "## Contact"
    still carries its context ("Join the lab and contact us").
    """
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)  # drop editor notes
    url = PAGE_URLS.get(source, SITE_URL)

    title = ""
    entries: list[Entry] = []
    heading, lines = "", []

    def flush():
        body = "\n".join(lines).strip()
        if body:
            label = f"{title} > {heading}" if heading and heading != title else title
            entries.append(Entry(
                id=f"{source}#{len(entries)}",
                kind="page",
                title=heading or title or source,
                url=url,
                source=source,
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
    return entries


def build_entries(data_dir: Path) -> list[Entry]:
    """Every search entry: records, overviews and markdown page sections.

    Raises KnowledgeError if a record file is broken.
    """
    kb = load(data_dir)
    entries = [_publication_entry(kb, p) for p in kb.publications]
    entries += [_person_entry(kb, p) for p in kb.people]
    entries += [_theme_entry(kb, t) for t in kb.themes]
    entries += _overview_entries(kb)
    for path in sorted(data_dir.glob("*.md")):
        entries += chunk_markdown(path.read_text(encoding="utf-8"), path.name)
    return entries


def fingerprint(data_dir: Path) -> str:
    """A hash of every data file plus INDEX_VERSION: changes whenever the index is out of date."""
    digest = hashlib.sha256(INDEX_VERSION.encode())
    for path in sorted([*data_dir.glob("*.json"), *data_dir.glob("*.md")]):
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
