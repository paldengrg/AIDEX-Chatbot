"""API tests for automatic learning, usage tracking, feedback and deactivation.

Uses the mock provider, so extraction runs on the offline keyword heuristic.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module", autouse=True)
def started_app():
    with TestClient(app):
        yield


def user_with_memory() -> TestClient:
    c = TestClient(app)
    name = "l" + uuid.uuid4().hex[:10]
    assert c.post("/api/auth/register", json={"username": name, "password": "password123"}).status_code == 201
    c.patch("/api/me", json={"memory_enabled": True})
    return c


def chat(c: TestClient, text: str) -> dict:
    res = c.post("/api/chat", json={"message": text})
    assert res.status_code == 200, res.text
    return res.json()


def items(c: TestClient) -> list[dict]:
    return c.get("/api/memory").json()["items"]


def test_preference_is_learned_and_applied_immediately():
    c = user_with_memory()
    body = chat(c, "From now on please keep answers short. What are the research themes?")
    assert len(body["learned"]) == 1
    learned = body["learned"][0]
    assert learned["category"] == "intent_calibration"
    assert [m["id"] for m in body["memory_used"]] == [learned["id"]]  # used in this reply
    assert body["exchange_id"] is not None
    assert "USED_MEMORY" not in body["reply"]                        # marker stripped

    item = items(c)[0]
    assert item["source"] == "explicit_instruction"
    assert item["times_retrieved"] == 1 and item["times_used"] == 1


def test_correction_counts_against_used_memory():
    c = user_with_memory()
    chat(c, "Please always use bullet points.")
    chat(c, "No, that's not what I asked.")
    assert items(c)[0]["times_corrected"] == 1


def test_thumbs_up_and_down():
    c = user_with_memory()
    first = chat(c, "Please always use bullet points.")
    assert c.post("/api/feedback", json={"exchange_id": first["exchange_id"], "rating": 1}).status_code == 204
    assert items(c)[0]["times_helpful"] == 1
    again = c.post("/api/feedback", json={"exchange_id": first["exchange_id"], "rating": -1})
    assert again.status_code == 409   # one rating per reply

    second = chat(c, "Who leads the lab?")
    c.post("/api/feedback", json={"exchange_id": second["exchange_id"], "rating": -1})
    assert items(c)[0]["times_corrected"] == 1


def test_cannot_rate_someone_elses_reply():
    alice, bob = user_with_memory(), user_with_memory()
    reply = chat(alice, "Please always use bullet points.")
    res = bob.post("/api/feedback", json={"exchange_id": reply["exchange_id"], "rating": -1})
    assert res.status_code == 404


def test_repeatedly_corrected_item_is_deactivated():
    c = user_with_memory()
    chat(c, "Please always answer in one sentence.")
    for _ in range(3):
        chat(c, "No, wrong.")
    item = items(c)[0]
    assert item["active"] is False
    assert item["deactivation_reason"] == "repeatedly linked to corrections"
    assert item["deactivated_at"] is not None

    # Resuming gives it a fresh start instead of being switched off again.
    c.patch(f"/api/memory/{item['id']}", json={"active": True})
    chat(c, "Thanks. Who leads the lab?")
    assert items(c)[0]["active"] is True


def test_clarify_preference_is_learned():
    c = user_with_memory()
    assert c.get("/api/me").json()["user"]["clarify_style"] == "neutral"
    chat(c, "Just answer, don't ask me questions.")
    assert c.get("/api/me").json()["user"]["clarify_style"] == "act"


def test_nothing_is_learned_without_opt_in():
    c = TestClient(app)
    name = "n" + uuid.uuid4().hex[:10]
    c.post("/api/auth/register", json={"username": name, "password": "password123"})
    body = chat(c, "Please always use bullet points.")
    assert body["learned"] == [] and body["exchange_id"] is None
    assert items(c) == []


def test_anonymous_users_get_no_learning():
    body = chat(TestClient(app), "Please always use bullet points.")
    assert body["learned"] == [] and body["exchange_id"] is None
