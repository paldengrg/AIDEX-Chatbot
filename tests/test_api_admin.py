"""API tests for the admin dashboard: access control, privacy rules, CSV export."""

import csv
import io
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module", autouse=True)
def started_app():
    with TestClient(app):
        yield


def admin() -> TestClient:
    c = TestClient(app)
    assert c.post("/api/admin/login", json={"username": "admin", "password": "test-admin-pass"}).status_code == 204
    return c


def user(participant: bool) -> TestClient:
    c = TestClient(app)
    c.post("/api/auth/register", json={"username": "a" + uuid.uuid4().hex[:10], "password": "password123"})
    c.patch("/api/me", json={"memory_enabled": True})
    if participant:
        version = c.get("/api/research/info").json()["consent_version"]
        c.post("/api/research/consent", json={"consent_version": version, "agree_information": True,
                                              "agree_logging": True, "agree_withdrawal": True})
    return c


def test_admin_page_and_data_need_login():
    anon = TestClient(app)
    assert anon.get("/admin").status_code == 200            # the login page itself
    assert anon.get("/api/admin/stats").status_code == 401
    assert anon.get("/api/admin/export/memory_items.csv").status_code == 401
    bad = anon.post("/api/admin/login", json={"username": "admin", "password": "wrong"})
    assert bad.status_code == 401


def test_normal_users_are_not_admins():
    assert user(False).get("/api/admin/stats").status_code == 401


def test_stats_and_privacy_rules():
    student, participant = user(False), user(True)
    student.post("/api/memory", json={"rule_text": "Student secret preference"})
    participant.post("/api/memory", json={"rule_text": "Participant prefers tables"})
    for c in (student, participant):
        c.post("/api/chat", json={"message": "What are the research themes?"})

    stats = admin().get("/api/admin/stats").json()
    assert stats["totals"]["memory_items"] >= 2
    assert stats["ratios"]["paper_reuse"] == 3.4
    texts = [i["rule_text"] for i in stats["top_reused"]]
    assert "Participant prefers tables" in texts
    assert "Student secret preference" not in texts            # hidden for students
    assert any(t.startswith("[hidden") for t in texts)
    assert stats["data_notice"]["contains_synthetic"] is True   # ethics not approved


def test_csv_export():
    user(True).post("/api/memory", json={"rule_text": "Likes short answers"})
    res = admin().get("/api/admin/export/memory_items.csv")
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/csv")
    rows = list(csv.DictReader(io.StringIO(res.text)))
    assert rows and "times_used" in rows[0] and "username" not in rows[0]
    assert admin().get("/api/admin/export/users.csv").status_code == 404   # not exportable
