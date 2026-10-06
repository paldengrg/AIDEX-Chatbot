"""The lab's knowledge base: load the data files and turn them into search entries.

data/ holds two kinds of content:
  - Records (JSON): themes.json, publications.json and people.json, one item per
    research theme, paper and person, each with a link to where it came from.
  - Pages (markdown): about.md, join-and-contact.md, news.md, ...

load() reads and checks the records, so a typo in a data file stops the app
with a clear message instead of quietly weakening the answers.

Pure Python (no ChromaDB), so everything here is quick to unit test.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

SITE_URL = "https://aidxlab.github.io/"

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
