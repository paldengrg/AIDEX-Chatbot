# Structured Knowledge Base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the lab's markdown people/publications/themes files with validated JSON records, so search returns one entry per paper, person and theme (each with a link), list questions get complete answers, and the index rebuilds itself when data changes.

**Architecture:** A new pure-Python module `app/knowledge.py` loads and checks `data/*.json`, and turns records plus markdown pages into `Entry` objects (including three overview entries). `app/rag.py` keeps only the ChromaDB work: it indexes those entries, stores a fingerprint of `data/` to detect stale indexes, and returns hits with `kind/title/url`. `chat.py` passes that metadata to the model, and `chat.js` shows sources as titled links.

**Tech Stack:** Python 3.14 (project `.venv`), FastAPI, ChromaDB 1.5 (default local embedding model), pytest, plain JavaScript.

**Spec:** `docs/superpowers/specs/2026-10-07-structured-knowledge-base-design.md`

## Global Constraints

- Run Python with `.venv/Scripts/python` (Git Bash on Windows). No new dependencies.
- Read and write every data file with `encoding="utf-8"`; Windows defaults to cp1252.
- Every URL in data, entries, sources and links must start with `https://`.
- Abstracts are official and word for word, or `null`. Never write or paraphrase one.
- Do not change the system prompt's rules or tone; that is Part 2. Only the `<document>` tag format changes.
- Keep each source's `source` key equal to the data file name (`people.json`, `about.md`); the research log and analytics rely on it.
- User-visible text in the browser is inserted with `textContent`, never `innerHTML`.
- Data-file bios do not name the lab (the website says "AIDEX", the app says "AIDX"; undecided).
- Stage only the files each task lists. The working tree also holds the user's own uncommitted work.
- Commit messages end with: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Review Focus

1. **A bad edit to a data file when an index already exists** — startup must fail with a message naming the file and item, and the existing index must be left intact, not wiped. Pinned in Task 3 (`test_bad_data_edit_keeps_the_old_index`).
2. **Accented names and typographic dashes** (Merigó, León-Castro, 215–266, LLM–MCDM) must survive loading on Windows and come back unchanged through retrieval and the API. Pinned in Task 1 (`test_real_data_keeps_accented_names_and_dashes`) and Task 4 (`test_special_characters_survive_indexing`).
3. **A non-https or `javascript:` URL** must never become a clickable link. Rejected at load in Task 2 (`test_bad_data_is_rejected_with_a_clear_message`), and guarded again in the browser in Task 5.
4. **Quotes or `&` in titles and URLs** inside `<document>` tags: a quote must not break the tag, and `&` in URLs must stay intact so links the model repeats still work. Pinned in Task 3 (`test_prompt_tag_survives_quotes_and_keeps_ampersands_in_links`).
5. **A new team member with no papers and no links** must still produce an entry that links to the site's `#people` section. Pinned in Task 2 (`test_person_without_papers_or_links_still_builds`).

## Before you start

- [ ] **Check the user's uncommitted work is committed.** Run `git status --short`. If it lists anything other than files this plan creates, stop and ask the user to commit their Phase 3–4 work first (or to approve committing it as-is). Otherwise this plan's commits to `app/chat.py`, `app/config.py`, `app/llm_client.py`, `static/js/chat.js`, `static/css/styles.css` and `.env.example` would include their changes.
- [ ] **Confirm the branch and a green baseline.**

Run: `git branch --show-current && .venv/Scripts/python -m pytest -q`
Expected: `feature/structured-knowledge-base`, then `96 passed`.

## File map

| File | Change | Responsibility |
|---|---|---|
| `data/themes.json`, `data/publications.json`, `data/people.json` | Create | The records |
| `data/people.md`, `data/publications.md`, `data/research-themes.md` | Delete (Task 3) | Replaced by the JSON files |
| `app/knowledge.py` | Create | Load and check records; build search entries; data fingerprint |
| `app/rag.py` | Rewrite | ChromaDB only: index entries, rebuild when stale, retrieve |
| `app/chat.py` | Modify | `<document>` tag metadata; sources list |
| `app/llm_client.py` | Modify | Mock provider's tag pattern |
| `app/config.py`, `.env.example` | Modify | `RAG_TOP_K` default |
| `static/js/chat.js`, `static/css/styles.css` | Modify | Sources panel rows with links |
| `tests/test_knowledge_units.py` | Create | Unit tests for `knowledge.py` |
| `tests/test_retrieval.py` | Create | Index rebuild and retrieval quality (real ChromaDB, no LLM) |
| `tests/test_units.py`, `tests/test_api.py` | Modify | New entry and source shapes |
| `eval/questions.json` | Modify | Publication-discovery questions |

---

### Task 1: Record files and loader with checks

**Files:**
- Create: `data/themes.json`, `data/publications.json`, `data/people.json`
- Create: `app/knowledge.py`
- Test: `tests/test_knowledge_units.py`

**Interfaces:**
- Consumes: nothing.
- Produces (in `app/knowledge.py`):
  - `SITE_URL: str = "https://aidxlab.github.io/"`
  - `PUBLICATION_TYPES: dict[str, tuple[str, str]]` — type id → (singular, plural) label
  - `LINK_KINDS: tuple[str, ...] = ("profile", "scholar", "linkedin")`
  - `class KnowledgeError(ValueError)`
  - `@dataclass class Knowledge: themes: list[dict]; publications: list[dict]; people: list[dict]; notes: dict[str, str]` (`notes` maps file name → its optional top-level `"note"`)
  - `load(data_dir: Path) -> Knowledge` — raises `KnowledgeError`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_knowledge_units.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_knowledge_units.py -q`
Expected: collection error, `ImportError: cannot import name 'knowledge' from 'app'`.

- [ ] **Step 3: Create the three data files**

These were generated on 2026-10-07 from the site's HTML, the theme panels in the site's `script.js`, and the official abstracts. Copy them exactly; save as UTF-8.

`data/themes.json`:

```json
{
  "retrieved_from": "https://aidxlab.github.io/ (research section and script.js theme panels)",
  "retrieved_on": "2026-10-07",
  "items": [
    {
      "id": "decision-intelligence",
      "number": 1,
      "name": "Decision Intelligence & Human-AI Collaboration",
      "description": "Advancing AI systems that support better human decision-making."
    },
    {
      "id": "trustworthy-ai",
      "number": 2,
      "name": "Trustworthy AI & Governance",
      "description": "Building responsible, explainable, fair, and accountable AI."
    },
    {
      "id": "agentic-ai",
      "number": 3,
      "name": "Agentic AI & Adaptive Systems",
      "description": "Developing intelligent agents that can reason, adapt, and collaborate safely."
    },
    {
      "id": "learning",
      "number": 4,
      "name": "AI for Learning & Capability Development",
      "description": "Enhancing education, assessment, and personalised learning through AI."
    },
    {
      "id": "organisations-society",
      "number": 5,
      "name": "AI for Organisations & Society",
      "description": "Applying AI to create measurable impact across business, education, government, and communities."
    }
  ]
}
```

`data/publications.json`:

```json
{
  "retrieved_from": "https://aidxlab.github.io/",
  "retrieved_on": "2026-10-07",
  "items": [
    {
      "id": "jmo-bibliometric-analysis",
      "title": "From regional roots to global reach: A 30-year bibliometric analysis of the Journal of Management & Organization",
      "authors": ["Hussain, W.", "Merigó, J. M.", "Baraheem, H.", "Ratten, V."],
      "year": 2026,
      "type": "journal_article",
      "venue": "Journal of Management & Organization, 32(1), 215–266.",
      "url": "https://doi.org/10.1017/jmo.2025.10066",
      "themes": ["organisations-society"],
      "abstract": "This study provides the large-scale bibliometric assessment of the Journal of Management & Organization (JMO), offering insights into its intellectual trajectory and positioning within the management field. Covering 1,083 documents published between 1995 and 2024, indexed in Web of Science and Scopus, the analysis applies performance metrics, citation structures, and science mapping using VOSviewer and Bibliometrix. The study is further grounded in institutional and field-theoretic perspectives, interpreting JMO’s evolution as a process of legitimacy-building and scientific capital accumulation within a global knowledge field. The results show that JMO’s growth has been marked by cyclical expansion, with a sharp increase in productivity since the mid-2000s but uneven citation impact, heavily reliant on a small set of landmark articles. Co-citation and bibliographic coupling analyses reveal intellectual roots in organizational behavior, psychology, and strategy, while keyword and thematic mapping highlight enduring strengths in leadership, human resource management, and job satisfaction. At the same time, new research frontiers have emerged in governance, innovation, sustainability, and work-life balance, reflecting JMO’s responsiveness to global challenges such as COVID-19 and digital transformation. Collaboration networks confirm the journal’s Australasian anchoring, yet also demonstrate growing integration into international research systems, particularly through linkages with the United States, China, and Europe. This study contributes to understanding JMO’s evolving role within management and organizational scholarship, identifying both its achievements and challenges. The findings offer insights for scholars, institutions, and editors on how JMO can consolidate high-impact niches, diversify its author base, and strengthen its influence in shaping global management debates.",
      "abstract_source": "https://doi.org/10.1017/jmo.2025.10066"
    },
    {
      "id": "llm-energy-forecasting",
      "title": "Harnessing AI-driven large language models (LLMs) for enhanced forecasting and optimisation of wind and solar energy generation and load demand profiles",
      "authors": ["Aleassa, O.", "Bekhit, M.", "Hussain, W.", "Fattah, I. M. R."],
      "year": 2026,
      "type": "journal_article",
      "venue": "Renewable and Sustainable Energy Reviews, 239, 117171.",
      "url": "https://www.sciencedirect.com/science/article/pii/S1364032126004703",
      "themes": ["organisations-society"],
      "abstract": null,
      "abstract_source": null
    },
    {
      "id": "ai-driven-mathematics-education",
      "title": "AI-Driven Mathematics Education",
      "authors": ["Sahni, M.", "León-Castro, E.", "Hussain, W."],
      "year": 2026,
      "type": "edited_book",
      "venue": "Studies in Big Data, Vol. 201. Springer, Cham.",
      "url": "https://doi.org/10.1007/978-3-032-31204-4",
      "themes": ["learning"],
      "abstract": null,
      "abstract_source": null
    },
    {
      "id": "human-layer-agentic-memory",
      "title": "The Human Layer of Agentic AI Memory: What Self-Improving LLM Systems Actually Learn in Production",
      "authors": ["Parameswaran, A.", "Hussain, W.", "Hossain, M. N."],
      "year": 2026,
      "type": "book_chapter",
      "venue": "In M. Sahni, E. León-Castro & W. Hussain (Eds.), AI-Driven Mathematics Education (pp. 35–52). Springer, Cham.",
      "url": "https://doi.org/10.1007/978-3-032-31204-4_3",
      "themes": ["agentic-ai"],
      "abstract": null,
      "abstract_source": null
    },
    {
      "id": "reimagining-student-success",
      "title": "Reimagining Student Success: An Agentic Framework for Early Detection and Adaptive Nudging in Higher Education",
      "authors": ["Hossain, M. N.", "Hussain, W.", "Bekhit, M."],
      "year": 2026,
      "type": "book_chapter",
      "venue": "In M. Sahni, E. León-Castro & W. Hussain (Eds.), AI-Driven Mathematics Education. Springer, Cham.",
      "url": "https://doi.org/10.1007/978-3-032-31204-4_11",
      "themes": ["agentic-ai", "learning"],
      "abstract": null,
      "abstract_source": null
    },
    {
      "id": "deep-learning-stroke-care",
      "title": "Deep Learning in Stroke Care",
      "authors": ["Zafari, Y.", "Rashed, E. A.", "Hussain, W.", "Mabrok, M."],
      "year": 2026,
      "type": "book_chapter",
      "venue": "In Deep Learning Applications in Neuroinformatics (pp. 49–79).",
      "url": "https://scholar.google.com/scholar?q=%22Deep%20learning%20in%20stroke%20care%22",
      "themes": ["organisations-society"],
      "abstract": null,
      "abstract_source": null
    },
    {
      "id": "llm-education-agents-k12",
      "title": "AI Intelligence Within Teaching: A Multidimensional Evaluation of Large Language Model Education Agents in K-12 Education",
      "authors": ["Hussain, W.", "Varanasi, M. R.", "New, L."],
      "year": 2026,
      "type": "conference_paper",
      "venue": "2025 International Conference on Computational Engineering, Sensing Technology and Management (ICCETM), Sydney, Australia (pp. 1–6). IEEE.",
      "url": "https://doi.org/10.1109/ICCETM66557.2025.11558109",
      "themes": ["learning"],
      "abstract": null,
      "abstract_source": null
    },
    {
      "id": "static-to-dynamic-personalization",
      "title": "From Static to Dynamic Personalization: Applying Agentic AI to the Felder–Silverman Learning Style Model (FSLSM)",
      "authors": ["Hossain, M. N.", "Rupai, A. A. A.", "Hussain, W."],
      "year": 2026,
      "type": "conference_paper",
      "venue": "2025 International Conference on Computational Engineering, Sensing Technology and Management (ICCETM). IEEE.",
      "url": "https://scholar.google.com/scholar?q=%22From%20Static%20to%20Dynamic%20Personalization%3A%20Applying%20Agentic%20AI%20to%20the%20Felder-Silverman%20Learning%20Style%20Model%22",
      "themes": ["agentic-ai", "learning"],
      "abstract": null,
      "abstract_source": null
    },
    {
      "id": "twin-minds-cyber-defense",
      "title": "Twin Minds in Cyber-Defense: A Dual-Agent Framework for Safe Automated Assessment in Security Education",
      "authors": ["Bekhit, M.", "Rashed, E. A.", "Othman, W.", "Hussain, W."],
      "year": 2026,
      "type": "conference_paper",
      "venue": "2026 23rd International Learning and Technology Conference (L&T), 23, 71–75.",
      "url": "https://scholar.google.com/scholar?q=%22Twin%20Minds%20in%20Cyber-Defense%3A%20A%20Dual-Agent%20Framework%20for%20Safe%20Automated%20Assessment%20in%20Security%20Education%22",
      "themes": ["trustworthy-ai", "agentic-ai", "learning"],
      "abstract": null,
      "abstract_source": null
    },
    {
      "id": "emfe-malaria",
      "title": "EMFE: A lightweight, explainable machine learning framework for malaria cell classification",
      "authors": ["Kafi, M. A. A.", "Hussain, W.", "Karmakar, M.", "Banshal, S. K.", "Al Marouf, A."],
      "year": 2026,
      "type": "preprint",
      "venue": "arXiv preprint arXiv:2608.24793.",
      "url": "https://arxiv.org/abs/2608.24793",
      "themes": ["trustworthy-ai"],
      "abstract": "Automated malaria diagnosis from stained blood-smear microscopy is dominated by deep convolutional neural networks that are accurate but computationally expensive, poorly interpretable, and rarely validated with patient-level rigor. We present EMFE (Efficient Mathematical Feature Extraction), a five-feature framework for classifying single red-blood-cell images as parasitized or uninfected using Gray World color normalization, adaptive green-channel thresholding, morphological spot detection, and classical machine learning. Using the NIH LHNCBC malaria dataset (27,558 images from 200 patients), we evaluate Random Forest, Histogram Gradient Boosting, and Support Vector Machine classifiers under patient-grouped nested cross-validation (K_outer=20, K_inner=3), ensuring that cells from each patient remain within a single fold. The optimized Random Forest achieves 94.6% pooled out-of-fold accuracy (95% CI [93.6, 95.7]), corroborated by an untouched 40-patient holdout test (94.3%) and a patient-level permutation test (p<0.001, 1,000 permutations). Ablation experiments quantify the contribution of individual features and pipeline stages. Hardware-matched comparisons with retrained DenseNet121, ResNet50, and MobileNetV2 models assess the accuracy-efficiency trade-off. Synthetic perturbations characterize three failure modes, while explainability analysis identifies spot saturation as the dominant discriminative feature. Patient-level aggregation further quantifies sensitivity-specificity trade-offs and false-positive accumulation. These results demonstrate a statistically rigorous, interpretable, and computationally lightweight alternative to deep learning, while explicitly quantifying its limitations.",
      "abstract_source": "https://arxiv.org/abs/2608.24793"
    },
    {
      "id": "credibility-weighted-llm-mcdm",
      "title": "Credibility-Weighted Evidence Aggregation Using Hybrid LLM–MCDM for Review-Driven Decision Analysis",
      "authors": ["Rajaeian, M.", "Hussain, W."],
      "year": 2026,
      "type": "preprint",
      "venue": "SSRN preprint 6277363.",
      "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6277363",
      "themes": ["decision-intelligence"],
      "abstract": "Digital marketplaces host vast volumes of user-generated feedback that contain rich evaluative signals but are unstructured, noisy, and highly variable in credibility. Conventional rating- and sentiment-based methods collapse this complexity into coarse summary scores, obscuring reviewer trustworthiness, argument quality, and temporal relevance—factors essential for reliable decision analytics. This study introduces a hybrid large language model–multi-criteria decision-making (LLM–MCDM) framework that converts free-text reviews into structured, credibility-weighted decision evidence. The approach integrates: (i) Consumer Feedback Attributes capturing reviewer credibility and temporal freshness, (ii) LLM-based extraction of criterion-specific sentiment evidence, and (iii) a multi-criteria aggregation mechanism combining Analytic Hierarchy Process weighting with additive utility modelling. By embedding credibility and recency directly into the aggregation process, the framework yields transparent, interpretable, and criterion-resolved evaluations. A temporal extension further enables instantaneous and trend-based early-warning signals when credible negative evidence surpasses configurable risk thresholds. The model is evaluated on 1,054 TrustRadius reviews spanning 30 infrastructure-as-a-service products. Results show that credibility-weighted aggregation produces rankings that diverge markedly from star-rating baselines, underscoring the inadequacy of ratings as proxies for structured, credibility-aware assessments. Robustness analyses across alternative credibility metrics, recency decay settings, and baseline comparators confirm the stability of rankings and the practical value of the early-warning signals. The findings demonstrate the potential of hybrid soft-computing aggregation to strengthen review-driven decision analytics in complex and dynamically evolving environments.",
      "abstract_source": "https://api.crossref.org/works/10.2139/ssrn.6277363"
    }
  ]
}
```

`data/people.json`:

```json
{
  "retrieved_from": "https://aidxlab.github.io/ (people section)",
  "retrieved_on": "2026-10-07",
  "note": "The lab website says further team members will be added as details are provided.",
  "items": [
    {
      "id": "walayat-hussain",
      "name": "Associate Professor Walayat Hussain",
      "role": "Director",
      "bio": "Director of the lab and Head of Discipline, Information Technology and Systems, Peter Faber Business School, Australian Catholic University. More than two decades of experience across academia and industry, with over 90 publications including ERA A* and A ranked venues. Associate Editor of IET Communications, International Journal of Web Information Systems and Forecasting, and a Fellow of the European Alliance for Innovation.",
      "author_names": ["Hussain, W."],
      "links": {"profile": "https://www.acu.edu.au/research-and-enterprise/our-people/walayat-hussain", "scholar": "https://scholar.google.com/citations?hl=en&user=HaA3MowAAAAJ", "linkedin": "https://www.linkedin.com/in/walayat-hussain-3104317a/"}
    },
    {
      "id": "nazmul-hossain",
      "name": "Md Nazmul Hossain",
      "role": "PhD Candidate",
      "bio": "PhD Candidate, Peter Faber Business School, Australian Catholic University. Recent work on agentic frameworks for early detection and adaptive nudging in higher education, and on what self-improving LLM systems learn in production.",
      "author_names": ["Hossain, M. N."],
      "links": {"scholar": "https://scholar.google.com/citations?user=WHNwDPUAAAAJ&hl=en", "linkedin": "https://www.linkedin.com/in/md-nazmul-hossain-bd/"}
    }
  ]
}
```

- [ ] **Step 4: Write the loader**

Create `app/knowledge.py`:

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_knowledge_units.py -q`
Expected: `20 passed`.

- [ ] **Step 6: Cross-check the publications against the live site**

Save this as `check_site.py` in the session scratchpad (not in the repo) and run it with `.venv/Scripts/python <scratchpad>/check_site.py`:

```python
"""One-off check: data/publications.json and data/themes.json match aidxlab.github.io."""
import html, json, re, sys, urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
DATA = Path(r"C:\UNI Folders\ITEC 320\AIDEX\data")
UA = {"User-Agent": "AIDX-Assistant-content-check/1.0"}
get = lambda u: urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30).read().decode("utf-8")
text = lambda f: re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", f))).strip()

page, js = get("https://aidxlab.github.io/"), get("https://aidxlab.github.io/script.js")
pubs = json.loads((DATA / "publications.json").read_text(encoding="utf-8"))["items"]
themes = json.loads((DATA / "themes.json").read_text(encoding="utf-8"))["items"]
problems = []

site = re.findall(r'<li class="pub"[^>]*>(.*?)</li>', page, re.S)
if len(site) != len(pubs):
    problems.append(f"site lists {len(site)} publications, data has {len(pubs)}")
for li, p in zip(site, pubs):
    authors = ", ".join(p["authors"][:-1]) + ", & " + p["authors"][-1] if len(p["authors"]) > 1 else p["authors"][0]
    if p["type"] == "edited_book":
        authors += " (Eds.)"
    expected = {
        "title": text(re.search(r"<h3>(.*?)</h3>", li, re.S).group(1)),
        "year": int(re.search(r'<div class="pub-meta[^"]*"><span[^>]*>[^<]*</span><span>(\d{4})</span>', li).group(1)),
        "authors": text(re.search(r'<p class="au">(.*?)</p>', li, re.S).group(1)),
        "venue": text(re.search(r'<p class="venue">(.*?)</p>', li, re.S).group(1)),
        "url": html.unescape(re.search(r'<div class="pub-links[^"]*"><a href="([^"]+)"', li).group(1)),
    }
    ours = {"title": p["title"], "year": p["year"], "authors": authors, "venue": p["venue"], "url": p["url"]}
    problems += [f"{p['id']}: {k} differs\n  site: {expected[k]!r}\n  data: {ours[k]!r}"
                 for k in expected if expected[k] != ours[k]]

site_themes = re.findall(r"\{ t: '([^']+)', d: '([^']+)', rel: \[([\d, ]*)\] \}", js)
for (name, desc, rel), t in zip(site_themes, themes):
    if (name, desc) != (t["name"], t["description"]):
        problems.append(f"theme {t['id']}: name/description differs from the site")
    tagged = sorted(n for n, p in enumerate(pubs) if t["id"] in p["themes"])
    if tagged != sorted(int(k) for k in rel.split(",")):
        problems.append(f"theme {t['id']}: tagged papers differ from the site's theme panel")

print("\n".join(problems) or "OK: publications and themes match the website")
```

Expected: `OK: publications and themes match the website`. If it reports differences, the site has changed since 2026-10-07: update the JSON from the site, rerun Step 5, and tell the user what changed.

- [ ] **Step 7: Run the full suite (nothing else should change yet)**

Run: `.venv/Scripts/python -m pytest -q`
Expected: `116 passed` (96 existing + 20 new).

- [ ] **Step 8: Commit**

```bash
git add data/themes.json data/publications.json data/people.json app/knowledge.py tests/test_knowledge_units.py
git commit -m "Add structured lab records and a checked loader

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Search entries and data fingerprint

**Files:**
- Modify: `app/knowledge.py` (add imports, constants and functions below)
- Test: `tests/test_knowledge_units.py` (append)

**Interfaces:**
- Consumes: `load`, `Knowledge`, `SITE_URL`, `PUBLICATION_TYPES` from Task 1.
- Produces (in `app/knowledge.py`):
  - `INDEX_VERSION: str = "2"`
  - `PAGE_URLS: dict[str, str]`
  - `@dataclass class Entry: id: str; kind: str; title: str; url: str; source: str; text: str` — `kind` ∈ `{"publication", "person", "theme", "overview", "page"}`
  - `papers_by(kb: Knowledge, person: dict) -> list[dict]`
  - `person_themes(kb: Knowledge, person: dict) -> list[str]` (theme names)
  - `chunk_markdown(text: str, source: str) -> list[Entry]`
  - `build_entries(data_dir: Path) -> list[Entry]`
  - `fingerprint(data_dir: Path) -> str` (hex SHA-256)
  - Entry ids: `publication:<id>`, `person:<id>`, `theme:<id>`, `overview:publications`, `overview:people`, `overview:themes`, `<file>#<n>`

`app/rag.py` still has its own `chunk_markdown` until Task 3 removes it. The duplication lasts one task.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_knowledge_units.py`:

```python
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
    assert director.text.startswith("Prof Director, Director\n")
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
    assert pubs.startswith("AIDX Lab publications (2 in total)")
    assert "Book chapters:\n- Paper A (2026)" in pubs
    assert "Preprints:\n- Paper B (2025)" in pubs
    people = entries["overview:people"].text
    assert "- Prof Director: Director\n- PhD Student: PhD Candidate" in people
    assert people.endswith("More team members are coming.")
    themes = entries["overview:themes"].text
    assert themes.startswith("AIDX Lab research themes (2)")
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_knowledge_units.py -q`
Expected: the 13 new tests FAIL with `AttributeError: module 'app.knowledge' has no attribute ...`; the 20 from Task 1 pass.

- [ ] **Step 3: Implement entries and fingerprint**

In `app/knowledge.py`, replace the module docstring and imports with:

```python
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
```

Below `SITE_URL = ...`, add:

```python
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
```

Below the `KnowledgeError` class, add:

```python
@dataclass
class Entry:
    """One item in the search index."""
    id: str      # unique, e.g. "publication:emfe-malaria" or "about.md#0"
    kind: str    # "publication" | "person" | "theme" | "overview" | "page"
    title: str   # shown to users as the source's name
    url: str     # citation link, always https://
    source: str  # data file it came from, e.g. "publications.json"
    text: str    # embedded for search and shown to the model
```

At the end of the file, add:

```python
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
    pubs = [f"AIDX Lab publications ({len(kb.publications)} in total)"]
    for type_id, (_, plural) in PUBLICATION_TYPES.items():
        group = [p for p in kb.publications if p["type"] == type_id]
        if group:
            pubs.append(f"{plural}:")
            pubs.extend(_paper_line(p) for p in group)
    people = [f"AIDX Lab people ({len(kb.people)} listed)"]
    people.extend(f"- {p['name']}: {p['role']}" for p in kb.people)
    themes = [f"AIDX Lab research themes ({len(kb.themes)})"]
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_knowledge_units.py -q`
Expected: `33 passed`.

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: `129 passed`.

- [ ] **Step 6: Commit**

```bash
git add app/knowledge.py tests/test_knowledge_units.py
git commit -m "Build search entries and a data fingerprint from the knowledge base

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Index and retrieve entries; pass titles and links to the prompt

**Files:**
- Rewrite: `app/rag.py`
- Modify: `app/chat.py` (imports, `build_system_prompt`, `answer`)
- Modify: `app/llm_client.py:76` (mock pattern)
- Modify: `app/config.py:43`, `.env.example:16` (`RAG_TOP_K` 4 → 6)
- Delete: `data/people.md`, `data/publications.md`, `data/research-themes.md`
- Test: create `tests/test_retrieval.py`; modify `tests/test_units.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: `knowledge.build_entries`, `knowledge.fingerprint`, `knowledge.KnowledgeError` from Tasks 1–2.
- Produces:
  - `rag.get_collection()`, `rag.ingest() -> int`, `rag.retrieve(query: str, k: int | None = None) -> list[dict]` with keys exactly `text, kind, id, title, url, source, distance`
  - `rag.MAX_DISTANCE`, `rag.COLLECTION_NAME = "aidx_documents"`
  - `chat.answer(...)` sources: `list[{"kind", "title", "url", "source"}]`
  - Prompt documents: `<document kind="…" title="…" url="…">`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_retrieval.py`:

```python
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
    assert "person:walayat-hussain" in [h["id"] for h in hits]
```

In `tests/test_units.py`, replace the import line `from app.rag import chunk_markdown, load_chunks` with:

```python
from app.knowledge import build_entries, chunk_markdown
```

and replace the three chunking tests and the first prompt test with:

```python
def test_chunks_one_per_section_with_title_context():
    chunks = chunk_markdown(SAMPLE, "join.md")
    assert [c.title for c in chunks] == ["Internships", "Contact"]
    assert chunks[1].text.startswith("Join the lab > Contact")
    assert chunks[0].id == "join.md#0"


def test_html_comments_are_removed():
    assert all("editor note" not in c.text for c in chunk_markdown(SAMPLE, "x.md"))


def test_real_data_folder_has_entries():
    entries = build_entries(settings.data_dir)
    assert len(entries) > 10
    assert len({e.id for e in entries}) == len(entries)  # ids are unique
```

```python
def test_prompt_contains_documents_and_identity_rule():
    doc = {"kind": "person", "title": "Walayat Hussain", "url": "https://example.edu/wh",
           "source": "people.json", "text": "Director info"}
    prompt = build_system_prompt([doc])
    assert '<document kind="person" title="Walayat Hussain" url="https://example.edu/wh">' in prompt
    assert "Director info" in prompt
    assert settings.assistant_name in prompt
    assert "Do not name any AI vendor" in prompt


def test_prompt_tag_survives_quotes_and_keeps_ampersands_in_links():
    doc = {"kind": "publication", "title": 'The "Human Layer" & memory',
           "url": "https://scholar.google.com/citations?hl=en&user=X",
           "source": "publications.json", "text": "Paper text"}
    prompt = build_system_prompt([doc])
    assert ('<document kind="publication" title="The \'Human Layer\' & memory" '
            'url="https://scholar.google.com/citations?hl=en&user=X">') in prompt
```

In `tests/test_api.py`, replace `test_chat_answers_from_documents` with:

```python
def test_chat_answers_from_documents(client):
    res = client.post("/api/chat", json={"message": "Who is the director of the lab?"})
    assert res.status_code == 200
    body = res.json()
    assert "Hussain" in body["reply"]
    assert any(s["source"] == "people.json" for s in body["sources"])
    for s in body["sources"]:
        assert set(s) == {"kind", "title", "url", "source"}
        assert s["url"].startswith("https://")
    assert "aidx_session" in res.cookies


def test_source_titles_keep_special_characters(client):
    res = client.post("/api/chat", json={
        "message": "bibliometric analysis of the Journal of Management & Organization"})
    titles = [s["title"] for s in res.json()["sources"]]
    assert ("From regional roots to global reach: A 30-year bibliometric analysis "
            "of the Journal of Management & Organization") in titles
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_retrieval.py tests/test_units.py tests/test_api.py -q`
Expected: failures — the hit-key assertions in `test_retrieval.py`, `DID NOT RAISE KnowledgeError`, the `<document kind=` assertions in `test_units.py`, and `people.json` / `kind` assertions in `test_api.py`.

- [ ] **Step 3: Rewrite `app/rag.py`**

Replace the whole file with:

```python
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
```

- [ ] **Step 4: Update `app/chat.py`**

Delete the line `from html import escape`.

Replace the `if docs:` block at the top of `build_system_prompt` with:

```python
    if docs:
        documents = "\n".join(
            f'<document kind="{_attr(d["kind"])}" title="{_attr(d["title"])}" '
            f'url="{_attr(d["url"])}">\n{d["text"]}\n</document>'
            for d in docs
        )
```

Add this function directly above `build_system_prompt`:

```python
def _attr(value: str) -> str:
    """Make a value safe inside a double-quoted tag attribute.

    Double quotes become single quotes so a title can't break out of the tag;
    '&' is left alone so links the model repeats still work.
    """
    return value.replace('"', "'").replace("\n", " ")
```

In `answer`, replace the de-duplication block with:

```python
    # De-duplicated list of sources to show under the reply.
    sources, seen = [], set()
    for d in docs:
        key = (d["title"], d["url"])
        if key not in seen:
            seen.add(key)
            sources.append({"kind": d["kind"], "title": d["title"], "url": d["url"],
                            "source": d["source"]})
```

- [ ] **Step 5: Update the mock provider, settings and data folder**

In `app/llm_client.py`, change the pattern in `_mock_complete` from `r'<document source="[^"]*">\s*(.*?)\s*</document>'` to:

```python
    match = re.search(r'<document [^>]*>\s*(.*?)\s*</document>', system, re.S)
```

In `app/config.py`, change `rag_top_k: int = _int("RAG_TOP_K", 4)` to `rag_top_k: int = _int("RAG_TOP_K", 6)`.
In `.env.example`, change `RAG_TOP_K=4` to `RAG_TOP_K=6`.

Delete the replaced markdown files:

```bash
git rm -q data/people.md data/publications.md data/research-themes.md
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all pass (`134 passed`). If `test_chat_answers_from_documents` fails on `"Hussain" in body["reply"]`, the mock echoed a different top entry; print `body["sources"]` and check retrieval before changing the test.

- [ ] **Step 7: Commit**

```bash
git add app/rag.py app/chat.py app/llm_client.py app/config.py .env.example tests/test_retrieval.py tests/test_units.py tests/test_api.py
git commit -m "Index structured entries and pass source titles and links to the prompt

Replaces people.md, publications.md and research-themes.md with the JSON
records. The index now rebuilds itself when data/ changes.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Retrieval quality tests and tuning

**Files:**
- Modify: `tests/test_retrieval.py` (append)
- Modify, if tuning requires: `app/rag.py` (`MAX_DISTANCE`), `app/config.py` and `.env.example` (`RAG_TOP_K`)

**Interfaces:**
- Consumes: `rag.retrieve`, `rag.get_collection` from Task 3.
- Produces: final `RAG_TOP_K` default and `MAX_DISTANCE`, recorded in the commit message.

- [ ] **Step 1: Write the quality tests**

Append to `tests/test_retrieval.py`:

```python
# --- Retrieval quality on the real data ----------------------------------------------
# Questions phrased in everyday words must find the right records, and an
# unrelated question must find nothing (so the bot can say it doesn't know).

@pytest.mark.parametrize("query, expected_any", [
    ("I'm interested in AI agents that adapt based on student behaviour",
     {"publication:reimagining-student-success", "publication:static-to-dynamic-personalization"}),
    ("Does the lab have anything on healthcare?",
     {"publication:deep-learning-stroke-care", "publication:emfe-malaria"}),
    ("What research does the lab do?", {"overview:themes"}),
    ("What has Walayat Hussain published?", {"person:walayat-hussain"}),
    ("Find papers about agentic AI", {"theme:agentic-ai", "publication:human-layer-agentic-memory"}),
])
def test_everyday_questions_find_the_right_records(query, expected_any):
    found = [h["id"] for h in rag.retrieve(query)]
    assert expected_any & set(found), f"{query!r} retrieved {found}"


def test_unrelated_question_finds_nothing():
    assert rag.retrieve("autonomous vehicle lidar perception") == []


def test_special_characters_survive_indexing():
    hits = {h["id"]: h for h in rag.retrieve("bibliometric analysis of the Journal of Management")}
    jmo = hits["publication:jmo-bibliometric-analysis"]
    assert "Merigó, J. M." in jmo["text"]
    assert "215–266" in jmo["text"]
```

- [ ] **Step 2: Run them**

Run: `.venv/Scripts/python -m pytest tests/test_retrieval.py -q`
Expected: either all pass (go to Step 5) or some fail. Either way, run Step 3 so the chosen values rest on numbers.

- [ ] **Step 3: Print the distances**

Run:

```bash
.venv/Scripts/python - <<'EOF'
from app import rag
queries = [
    "I'm interested in AI agents that adapt based on student behaviour",
    "Does the lab have anything on healthcare?",
    "What research does the lab do?",
    "What has Walayat Hussain published?",
    "Find papers about agentic AI",
    "bibliometric analysis of the Journal of Management",
    # off-topic: these should ideally find nothing
    "autonomous vehicle lidar perception",
    "What's the weather in Sydney tomorrow?",
    "Write me a Python game",
    "Who will win the World Cup?",
]
col = rag.get_collection()
for q in queries:
    r = col.query(query_texts=[q], n_results=8)
    print(q)
    for i, d in zip(r["ids"][0], r["distances"][0]):
        print(f"   {d:.3f}  {i}")
EOF
```

- [ ] **Step 4: Choose the values with this rule**

- `MAX_DISTANCE` must be **above** the distance of every expected entry for the six topic queries, and **below** the closest hit for "autonomous vehicle lidar perception". If 0.75 already does both, keep it. Otherwise use the midpoint between those two distances, rounded to two decimals. If no value can do both, stop and report the numbers to the user rather than weakening a test.
- `RAG_TOP_K`: keep 6 if every topic query's expected entry ranks in its top 6. Otherwise raise it to the smallest value up to 8 that works; if 8 is not enough, stop and report.
- Apply the values: `MAX_DISTANCE` in `app/rag.py`; `RAG_TOP_K` in both `app/config.py` and `.env.example`.
- The other off-topic queries are informational. Note which of them still return hits, for Part 2 (the prompt rules handle those).

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: `141 passed`.

- [ ] **Step 6: Commit**

```bash
git add tests/test_retrieval.py app/rag.py app/config.py .env.example
git commit -m "Pin retrieval quality with everyday questions; tune top-k and cut-off

RAG_TOP_K=<chosen>, MAX_DISTANCE=<chosen> (distances measured on the 2026-10-07 data).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Replace both `<chosen>` values in the message with the numbers from Step 4 before running it.

---

### Task 5: Sources panel shows titled links

**Files:**
- Modify: `static/js/chat.js` (new sources helpers above `// ---------- Safe rendering ----------`; `addList` and its first call at lines 119–132)
- Modify: `static/css/styles.css` (after line 384, `.sources h4`)

**Interfaces:**
- Consumes: API sources `{kind, title, url, source}` from Task 3.
- Produces: nothing used by later tasks.

- [ ] **Step 1: Add the sources helpers**

In `static/js/chat.js`, insert directly above the line `  // ---------- Safe rendering ----------`:

```javascript
  // ---------- Sources ----------

  // Icon and label for each kind of source the server returns.
  const SOURCE_KINDS = {
    publication: ["📄", "Publication"],
    person: ["👤", "Person"],
    theme: ["🔬", "Research theme"],
    overview: ["📚", "Overview"],
    page: ["🌐", "Lab website"],
  };

  // One row of the Sources panel: icon, title (a link only for https URLs), kind.
  // Everything is set with textContent, so a title can never inject markup.
  function sourceRow(source) {
    const [iconText, label] = SOURCE_KINDS[source.kind] || ["•", "Source"];
    const icon = document.createElement("span");
    icon.className = "source-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = iconText;

    const safeUrl = typeof source.url === "string" && source.url.startsWith("https://");
    const name = document.createElement(safeUrl ? "a" : "span");
    if (safeUrl) {
      name.href = source.url;
      name.target = "_blank";
      name.rel = "noopener noreferrer";
    }
    name.textContent = source.title || "Untitled";

    const kind = document.createElement("span");
    kind.className = "source-kind";
    kind.textContent = ` · ${label}`;
    return [icon, name, kind];
  }

```

- [ ] **Step 2: Use it in the panel**

Replace:

```javascript
        rows.forEach((text) => {
          const li = document.createElement("li");
          li.textContent = text;   // textContent: never interpreted as HTML
          list.appendChild(li);
        });
        details.append(h, list);
      };
      addList("Lab documents", sources.map((s) => `${s.heading} (${s.source})`));
```

with:

```javascript
        rows.forEach((row) => {
          const li = document.createElement("li");
          if (typeof row === "string") li.textContent = row;   // never interpreted as HTML
          else li.append(...row);                              // nodes built by sourceRow()
          list.appendChild(li);
        });
        details.append(h, list);
      };
      addList("Lab sources", sources.map(sourceRow));
```

- [ ] **Step 3: Style the rows**

In `static/css/styles.css`, add after the line `.sources h4 { margin: 6px 0 2px; font-size: .78rem; font-weight: 600; }`:

```css
.sources li a { color: inherit; text-decoration: underline; text-underline-offset: 2px; }
.sources .source-icon { margin-right: 4px; }
.sources .source-kind { opacity: .75; }
```

- [ ] **Step 4: Check the script parses**

Run: `node --check static/js/chat.js` (skip if `node` is not installed and say so in the report).
Expected: no output.

- [ ] **Step 5: Check it in the running app**

Start the server in the background with the free mock model:

```bash
LLM_PROVIDER=mock SECRET_KEY=dev-only-key .venv/Scripts/python -m uvicorn app.main:app --port 8765
```

Then check the API:

```bash
curl -s -X POST http://127.0.0.1:8765/api/chat -H "Content-Type: application/json" -d '{"message": "Find papers about agentic AI"}'
```

Expected: JSON whose `sources` items each have `kind`, `title` and an `https://` `url`, including at least one `"kind": "publication"`.

Then look at the page. If a headless browser is available (`.venv/Scripts/python -c "import playwright"` or `npx --no-install playwright --version`), open http://127.0.0.1:8765, send "Find papers about agentic AI", expand "Sources", and take a screenshot. **Look at it**: each row needs an icon, a title link and a kind label. Otherwise, ask the user to open http://127.0.0.1:8765, try it, and confirm. Stop the server afterwards.

- [ ] **Step 6: Commit**

```bash
git add static/js/chat.js static/css/styles.css
git commit -m "Show sources as titled links with their kind

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Eval questions and final verification

**Files:**
- Modify: `eval/questions.json` (insert after the `A22` line)
- Modify: `docs/superpowers/specs/2026-10-07-structured-knowledge-base-design.md` (status line)

**Interfaces:**
- Consumes: everything above.
- Produces: nothing.

- [ ] **Step 1: Add the questions**

In `eval/questions.json`, insert these lines directly after the line containing `"id": "A22"` (keep the comma at the end of the `A22` line, and each new line ends with a comma):

```json
  {"id": "A23", "tier": "anonymous", "category": "publications", "question": "Find papers about agentic AI.", "check": "keywords", "any": ["Human Layer", "Reimagining Student Success", "Felder", "Twin Minds"], "min_matches": 2},
  {"id": "A24", "tier": "anonymous", "category": "publications", "question": "What is the DOI of the Human Layer chapter?", "check": "keywords", "any": ["10.1007/978-3-032-31204-4_3"]},
  {"id": "A25", "tier": "anonymous", "category": "publications", "question": "Does the lab do any research related to healthcare?", "check": "keywords", "any": ["stroke", "malaria"]},
  {"id": "A26", "tier": "anonymous", "category": "publications", "question": "How many publications does the lab list?", "check": "keywords", "any": ["11", "eleven"]},
  {"id": "A27", "tier": "anonymous", "category": "research", "question": "Which publications belong to the Trustworthy AI and Governance theme?", "check": "keywords", "any": ["malaria", "Twin Minds"], "min_matches": 2},
  {"id": "A28", "tier": "anonymous", "category": "publications", "question": "What is the EMFE paper about?", "check": "keywords", "any": ["blood", "Random Forest", "parasit", "interpretable", "lightweight"], "min_matches": 2},
  {"id": "A29", "tier": "anonymous", "category": "people", "question": "Which papers has Md Nazmul Hossain co-authored?", "check": "keywords", "any": ["Human Layer", "Reimagining Student Success", "Felder"], "min_matches": 2},
```

Check the file still parses:

Run: `.venv/Scripts/python -c "import json; q = json.load(open('eval/questions.json', encoding='utf-8')); print(len(q), len({x['id'] for x in q}))"`
Expected: `39 39`.

- [ ] **Step 2: Run the eval in mock mode**

Run: `.venv/Scripts/python -m eval.run_eval --mock`
Expected: all 39 questions run and none shows an `HTTP` failure reason. Many keyword checks fail in mock mode, because the mock only echoes the top entry. That is expected: mock mode checks the plumbing only. The exit code is 1 because not everything passes.

- [ ] **Step 3: Run the full test suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: `141 passed`.

- [ ] **Step 4: Mark the spec implemented and commit**

In the spec, change the line starting `Status:` to `Status: implemented 2026-10-07`.

```bash
git add eval/questions.json docs/superpowers/specs/2026-10-07-structured-knowledge-base-design.md
git commit -m "Add publication-discovery eval questions; mark Part 1 implemented

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Report to the user**

Include: test counts, the tuned `RAG_TOP_K` and `MAX_DISTANCE`, which off-topic queries still retrieve something, the screenshot or their confirmation, and that the real-model eval (`.venv/Scripts/python -m eval.run_eval`, which uses their API key) is theirs to run.
