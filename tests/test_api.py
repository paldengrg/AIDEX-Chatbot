"""End-to-end API tests: real FastAPI app + real ChromaDB index, mock LLM.

The first run downloads the small embedding model (~80 MB), so it can take a
minute; later runs are quick.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # "with" runs the startup code (index build)
        yield c


def test_home_page_shows_assistant_name(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "AIDX Assistant" in res.text
    assert "{{ASSISTANT_NAME}}" not in res.text


def test_chat_answers_from_documents(client):
    res = client.post("/api/chat", json={"message": "Who is the director of the lab?"})
    assert res.status_code == 200
    body = res.json()
    assert "Hussain" in body["reply"]
    assert any(s["source"] == "people.md" for s in body["sources"])
    assert "aidx_session" in res.cookies


def test_session_memory_is_used_for_follow_ups(client):
    client.post("/api/chat", json={"message": "Tell me about internships"})
    res = client.post("/api/chat", json={"message": "and for PhD?"})
    assert res.status_code == 200
    assert any(s["source"] == "join-and-contact.md" for s in res.json()["sources"])


def test_empty_and_oversized_messages_are_rejected(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/chat", json={"message": "   "}).status_code == 422
    assert client.post("/api/chat", json={"message": "x" * 5000}).status_code == 422


def test_reset_clears_session(client):
    client.post("/api/chat", json={"message": "hello"})
    assert client.post("/api/reset").status_code == 204
