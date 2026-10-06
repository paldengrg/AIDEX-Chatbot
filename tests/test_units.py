"""Fast unit tests for the pure-Python parts (no ChromaDB, no web server)."""

from app.chat import build_system_prompt, retrieval_query
from app.config import settings
from app.knowledge import build_entries, chunk_markdown
from app.session_memory import SessionStore

SAMPLE = """<!-- editor note: should be removed -->
# Join the lab

## Internships
ACU students can apply for placements.

## Contact
North Sydney campus.
"""


# --- Chunking ---------------------------------------------------------------

def test_chunks_one_per_section_with_title_context():
    chunks = chunk_markdown(SAMPLE, "join.md")
    assert [c.title for c in chunks] == ["Join the lab: Internships", "Join the lab: Contact"]
    assert chunks[1].text.startswith("Join the lab > Contact")
    assert chunks[0].id == "join.md#0"


def test_html_comments_are_removed():
    assert all("editor note" not in c.text for c in chunk_markdown(SAMPLE, "x.md"))


def test_real_data_folder_has_entries():
    entries = build_entries(settings.data_dir)
    assert len(entries) > 10
    assert len({e.id for e in entries}) == len(entries)  # ids are unique


# --- Session memory ---------------------------------------------------------

def test_session_keeps_history_and_reuses_id():
    store = SessionStore(ttl_seconds=60)
    sid, session = store.get_or_create(None)
    store.add_exchange(session, "Hi", "Hello!")
    sid2, session2 = store.get_or_create(sid)
    assert sid2 == sid
    assert store.history(session2) == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]


def test_unknown_session_id_gets_a_fresh_session():
    store = SessionStore(ttl_seconds=60)
    sid, _ = store.get_or_create("made-up-id")
    assert sid != "made-up-id"


def test_history_is_capped():
    store = SessionStore(ttl_seconds=60)
    _, session = store.get_or_create(None)
    for i in range(settings.session_max_turns + 5):
        store.add_exchange(session, f"q{i}", f"a{i}")
    assert len(store.history(session)) == settings.session_max_turns * 2


def test_expired_sessions_are_forgotten():
    store = SessionStore(ttl_seconds=-1)  # everything is instantly expired
    sid, _ = store.get_or_create(None)
    sid2, _ = store.get_or_create(sid)
    assert sid2 != sid


def test_reset_forgets_session():
    store = SessionStore(ttl_seconds=60)
    sid, _ = store.get_or_create(None)
    store.reset(sid)
    assert store.get_or_create(sid)[0] != sid


# --- Prompt building ----------------------------------------------------------

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


def test_prompt_handles_no_documents():
    assert "No relevant documents" in build_system_prompt([])


def test_retrieval_query_adds_previous_user_question():
    history = [{"role": "user", "content": "Who is the director?"},
               {"role": "assistant", "content": "Walayat Hussain."}]
    assert retrieval_query("What has he published?", history) == \
        "Who is the director? What has he published?"
