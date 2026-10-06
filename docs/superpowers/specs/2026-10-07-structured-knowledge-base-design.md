# Part 1: Structured knowledge base — design

Date: 2026-10-07
Status: approved 2026-10-07; section 5 updated after content research (see Open items)

## Context

The AIDEX Assistant is being reshaped into an evidence-grounded research concierge:
it helps visitors find out what the AIDEX Lab researches, who does it, what they
have published and how to get involved. The work is split into four parts, each
with its own spec, plan and build:

1. **Structured knowledge base** (this document)
2. Answer policy and persona: lab facts vs general AI knowledge, scope, citation
   format, tone, explanation levels, "I couldn't find that" behaviour
3. Opening screen with six starter actions
4. Guided enquiry flows (PhD, collaboration)

Part 1 comes first because publication discovery, researcher matching and linked
citations all depend on having one searchable record per paper, person and theme.

## Problem

- `data/publications.md` is chunked per heading, so "Book chapters" is a single
  chunk holding three papers. A search for one paper returns its whole category.
- No file in `data/` contains a DOI or link, and only one paper has a summary.
- Retrieval returns the top 4 chunks, so list questions fail: Walayat Hussain is
  an author on all 11 papers, but "What has he published?" can surface at most 4.
- The index is only built when empty. After editing `data/`, the bot keeps serving
  old content until someone remembers to run `python -m app.rag`.
- Sources shown in the UI are file names (`People (people.md)`), with no links.

## Goals

- One search entry per publication, person and theme, carrying a link.
- Overview entries so "list everything" questions get complete answers.
- Every field traceable to a page; empty rather than invented when unknown.
- Sources in the UI show readable titles and open the publication or profile.
- The index rebuilds itself when `data/` changes.

## Non-goals (later parts or not planned)

- Changing the system prompt's rules, tone, citation format or scope (Part 2).
- Showing only the sources a reply actually cited (Part 2).
- Starter buttons and enquiry flows (Parts 3 and 4).
- An admin editor for the knowledge base. Files are edited by the developer.
- Putting the whole knowledge base in every prompt instead of retrieval.

## Design

### 1. Files and record shapes

Three new files in `data/`. Each is a JSON object with provenance and an `items`
list:

```json
{ "retrieved_from": "https://aidxlab.github.io/",
  "retrieved_on": "2026-10-07",
  "items": [ ... ] }
```

**`themes.json`** — five items.

| Field | Type | Notes |
|---|---|---|
| `id` | string | One of `decision-intelligence`, `trustworthy-ai`, `agentic-ai`, `learning`, `organisations-society` |
| `number` | int | 1–5, the site's order |
| `name` | string | Site wording, e.g. "Agentic AI & Adaptive Systems" |
| `description` | string | From the site |

**`publications.json`** — eleven items.

| Field | Type | Notes |
|---|---|---|
| `id` | string | Short slug, e.g. `human-layer-agentic-memory` |
| `title` | string | |
| `authors` | list of strings | As cited, e.g. `"Hussain, W."` |
| `year` | int | As listed on the site |
| `type` | string | `journal_article`, `edited_book`, `book_chapter`, `conference_paper` or `preprint` |
| `venue` | string | Full venue text from the citation |
| `url` | string | DOI, publisher, arXiv, SSRN or Google Scholar link from the site; must be `https://` |
| `themes` | list of theme ids | At least one; from the website's theme panels (see section 5) |
| `abstract` | string or null | Official abstract, word for word; `null` if not found |
| `abstract_source` | string or null | URL the abstract was taken from; `null` exactly when `abstract` is `null` |

**`people.json`** — two items.

| Field | Type | Notes |
|---|---|---|
| `id` | string | e.g. `walayat-hussain` |
| `name` | string | e.g. "Associate Professor Walayat Hussain" |
| `role` | string | e.g. "Director", "PhD Candidate" |
| `bio` | string | From the site |
| `author_names` | list of strings | How the person appears in `authors`, e.g. `["Hussain, W."]` |
| `links` | object | Any of `profile`, `scholar`, `linkedin`; each `https://` |

`people.json` may also have a top-level `note` (e.g. that more team members are
coming), which is added to the people overview entry.

There is no `interests` field. The site lists none for the Director, so a
person's themes are **worked out from the theme tags of their papers** instead.

**Files changed in `data/`:**

- Deleted (replaced by the JSON files): `people.md`, `publications.md`,
  `research-themes.md`. They remain recoverable from git.
- Kept as prose pages: `about.md`, `join-and-contact.md`, `news.md`,
  `human-layer-paper.md` (the lab's longer write-up; its DRAFT note stays).

### 2. New module `app/knowledge.py`

Responsibilities: load the three JSON files, check them, and turn records and
pages into search entries. It does not talk to ChromaDB; `app/rag.py` keeps that
job.

**Checks** (run at startup and in tests; failure raises `KnowledgeError` naming
the file, item id and problem):

- Required fields present with the right type.
- Ids unique within each file.
- Every publication theme id exists in `themes.json`.
- `type` is one of the allowed values.
- Every URL starts with `https://`.
- `abstract` and `abstract_source` are both set or both `null`.

**Search entries** (`Entry`: `id`, `kind`, `title`, `url`, `source`, `text`):

| Kind | Entry id | Text, in this order | `url` | `source` |
|---|---|---|---|---|
| `publication` | `publication:<id>` | Title; theme names; authors; year, type, venue; abstract, or "Citation only: no abstract available." | `url` | `publications.json` |
| `person` | `person:<id>` | "Publications and research by <name>"; name and role; themes (from their papers, most frequent first); bio; all their papers as "title (year)" | `links.profile`, else `links.scholar`, else site `#people` | `people.json` |
| `theme` | `theme:<id>` | Number and name; description; every paper tagged with it; people who authored those papers | site `#research` | `themes.json` |
| `overview` | `overview:publications` | "What has the AIDEX Lab published? All N publications:", then titles and years grouped by type | site `#publications` | `publications.json` |
| `overview` | `overview:people` | "Who works at the AIDEX Lab? People (N listed):", then all people with roles | site `#people` | `people.json` |
| `overview` | `overview:themes` | "What does the AIDEX Lab research? The lab's 5 research themes:", then each name and description | site `#research` | `themes.json` |
| `page` | `<file>#<n>` | Markdown section, chunked as today by `chunk_markdown` | site anchor for the file (below) | file name |

Page anchors: `about.md` → `#about`, `join-and-contact.md` → `#join`,
`news.md` → `#news`, `human-layer-paper.md` → that paper's DOI. Any other file →
the site root.

A person's papers are those whose `authors` list contains one of the person's
`author_names`, compared exactly.

The local embedding model reads only about the first 200 words of an entry.
Titles, theme names and headings therefore come first, so long entries are found
by their opening lines. The full text still goes to the model once retrieved.

### 3. Indexing and retrieval (`app/rag.py`)

**Ingest** stores each entry under its id, with metadata `{kind, title, url, source}`.

**Automatic rebuild.** `ingest()` saves a fingerprint in the collection metadata:
a SHA-256 over every `data/*.md` and `data/*.json` file (relative path plus bytes)
plus an `INDEX_VERSION` constant. Bump the constant when how entries are built
changes. `get_collection()` rebuilds when the stored fingerprint differs from
the current one, or when the collection is empty. `python -m app.rag` still
forces a rebuild.

**Retrieve** returns hits as `{text, kind, id, title, url, source, distance}`.
The `heading` key is replaced by `title`. `source` keeps the file name, so the
research log (`main.py`) and analytics are unchanged.

**Settings.** `RAG_TOP_K` default goes from 4 to 6 (`config.py` and
`.env.example`). `MAX_DISTANCE` stays 0.75. Both are then tuned against the
retrieval tests (see Testing), and the final values are recorded in the plan.

**Hybrid context (decided 2026-10-07 after measuring retrieval).** With the
local embedding model, list questions ranked the overview entries 10th–21st,
person entries scored 0.76–0.86 even when the question named the person, and an
off-topic question ("Write me a Python game", 0.713) scored closer than on-topic
ones. A distance cut-off therefore cannot be the off-topic filter. So:

- The three overview entries are **always** added to the prompt.
- A person's entry is added when the question contains one of the distinctive
  words of their name (4+ letters; titles such as "Associate" and "Professor"
  ignored; accented letters kept).
- Semantic search adds the closest entries: `RAG_TOP_K` 8, `MAX_DISTANCE` 0.75.
- Sources shown to the user are the name matches and semantic hits; the
  always-present overviews are listed only when search also found them.
- Overview and person entries open with question-shaped lines (table above).
- Declining off-topic questions is left to the answer rules (Part 2).

### 4. Prompt, sources and UI

**Prompt** (`chat.build_system_prompt`): document tags carry the new metadata.
Double quotes in values become single quotes so a title can't break the tag;
`&` is kept as-is so links the model repeats still work:

```
<document kind="publication" title="The Human Layer of Agentic AI Memory…" url="https://doi.org/…">
…
</document>
```

No rule changes in the prompt; Part 2 decides how the model cites. The mock
provider's pattern in `llm_client._mock_complete` is updated to match the new tag.

**Sources** (`chat.answer`): de-duplicated by `(title, url)`, each
`{kind, title, url, source}`.

**UI** (`static/js/chat.js`): the Sources panel lists each source as a small icon
for its kind, the title and the kind label, for example
"📄 The Human Layer of Agentic AI Memory · Publication". The title is a link only
when `url` starts with `https://`; links open in a new tab with
`rel="noopener noreferrer"`. All text is inserted with `textContent`. The panel
still lists everything retrieved.

### 5. Collecting the content

Rule: every field comes from a page that can be pointed to, or it stays empty.

1. **Site data.** Download the raw HTML of https://aidxlab.github.io/ and take
   titles, authors, years, venues, URLs, people, roles, bios, links and theme
   descriptions from it directly. Do not rely on model-summarised page content
   for URLs or citations.
2. **Abstracts.** For each paper, open its URL and extract the abstract from the
   raw page: `citation_abstract`, `dc.description` or `og:description` metadata,
   or arXiv's abstract block. If the publisher page blocks scripted access, the
   publisher's own abstract as deposited with Crossref (`api.crossref.org`, the
   official DOI registry) is also accepted. For papers with only a Google
   Scholar link, search for the exact title and accept an official source only
   if the title matches exactly. If nothing is available, `abstract` stays
   `null`. Never store a paraphrase; only markup and typographic hyphens
   (U+2011) are normalised.
3. **Theme tags.** Taken from the lab website's own theme panels: the `THEMES`
   list in https://aidxlab.github.io/script.js links each theme to its related
   publications. No tags are drafted by hand.
4. **Conflicts.** Where the site and the current markdown disagree, the site wins
   and the difference is reported to the user. Known case: the two ICCETM
   conference papers are listed as 2026 on the site and 2025 in
   `publications.md`; 2026 is used.

## Testing

**Unit tests** — new `tests/test_knowledge_units.py`, no model or network:

- Each check fails with a clear message: unknown theme id, duplicate id, missing
  field, bad `type`, non-`https` URL, abstract without source.
- The real files in `data/` load and pass every check.
- A person is matched to exactly their papers; their derived themes are correct.
- Overview entries list every paper, person and theme; the publication count is
  right.
- Entry text starts with the title; a `null` abstract produces "Citation only".
- Fingerprint: changing a file in a temporary data folder changes the
  fingerprint; an unchanged folder does not.

**Retrieval tests** — real ChromaDB and embedding model, no LLM:

| Query | Expected in results |
|---|---|
| "AI agents that adapt based on student behaviour" | Adaptive nudging chapter or FSLSM paper |
| "anything on healthcare?" | Stroke care and malaria titles in the prompt context |
| "What research does the lab do?" | `overview:themes` |
| "What has Walayat Hussain published?" | `person:walayat-hussain` (name match) |
| "What has Hossain published?" | `person:nazmul-hossain` only (no Hussain/Hossain mix-up) |
| any question | all three overview entries in the prompt context |

**Existing tests.** `tests/test_api.py` expects `people.json` instead of
`people.md`, and checks that each source has `kind`, `title` and an `https` url.
`tests/test_units.py` prompt tests are updated for the new document tag.

**Eval set.** About six questions added to `eval/questions.json` (category
`publications`), for example "Find papers about agentic AI", "What is the DOI of
the Human Layer chapter?" and "Does the lab do any research on healthcare?".
`python -m eval.run_eval --mock` checks the plumbing; the real-model run is
started by the user because it uses their API key.

**Manual check.** Run the app and confirm the Sources panel shows titles, kinds
and working links. This needs the packages in `requirements.txt`; how to install
them is agreed with the user first.

## Acceptance criteria

- `data/` holds `themes.json` (5), `publications.json` (11) and `people.json` (2),
  all passing the checks; the three replaced markdown files are gone.
- Every publication has an `https` URL taken from the site; every abstract is
  word for word with its source URL, or `null`.
- Theme tags match the website's theme panels.
- Editing any file in `data/` causes a rebuild on the next start.
- All unit, retrieval and API tests pass; `eval --mock` runs without errors.
- The Sources panel shows readable titles with working links.

## Open items

- Resolved: ICCETM papers use 2026, as on the site.
- Resolved: official abstracts were available for 3 of 11 papers (JMO via
  Cambridge, EMFE via arXiv, LLM–MCDM via Crossref). Springer, IEEE,
  ScienceDirect and SSRN pages block scripted access and Crossref holds no
  abstract for the rest, so 8 papers are citation only.
- Resolved: the lab is "AIDEX" everywhere, matching the website and logo
  (renamed before Part 1; internal names such as `aidx.db` and cookies unchanged).
