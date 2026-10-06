"""Unit tests for the dashboard metrics (pure functions, no database)."""

from app.analytics import compute_stats, deactivations_by_day


def item(id, category, used=0, retrieved=0, helpful=0, corrected=0, active=True,
         deactivated_at=None, reason=None, source="inferred"):
    return {"id": id, "user": "abcd1234", "rule_text": f"rule {id}", "category": category,
            "source": source, "times_used": used, "times_retrieved": retrieved,
            "times_helpful": helpful, "times_corrected": corrected, "active": active,
            "deactivated_at": deactivated_at, "deactivation_reason": reason}


ITEMS = [
    item(1, "intent_calibration", used=10, retrieved=12, helpful=4, corrected=1),
    item(2, "intent_calibration", used=7, retrieved=10, helpful=2),
    item(3, "domain_knowledge", used=5, retrieved=20, helpful=1, corrected=2),
    item(4, "domain_knowledge", used=0, retrieved=10, active=False,
         deactivated_at="2026-10-03T10:00:00", reason="retrieved often but never used"),
]
EXCHANGES = [
    {"feedback": 1, "was_corrected": False},
    {"feedback": -1, "was_corrected": True},
    {"feedback": None, "was_corrected": False},
    {"feedback": 1, "was_corrected": False},
]
USERS = [{"tier": "participant"}, {"tier": "student"}, {"tier": "participant"}]


def test_reuse_and_count_ratios():
    s = compute_stats(ITEMS, EXCHANGES, USERS)
    assert s["ratios"]["reuse"] == 3.4      # (10 + 7) / 5
    assert s["ratios"]["count"] == 1.0      # 2 items each
    assert s["ratios"]["paper_reuse"] == 3.4


def test_category_rates():
    intent = compute_stats(ITEMS, EXCHANGES, USERS)["by_category"]["intent_calibration"]
    assert intent["use_rate"] == round(17 / 22, 3)
    assert intent["helpful_rate"] == round(6 / 17, 3)
    assert intent["correction_rate"] == round(1 / 17, 3)


def test_totals_feedback_and_top_items():
    s = compute_stats(ITEMS, EXCHANGES, USERS)
    assert s["totals"]["memory_items"] == 4 and s["totals"]["inactive"] == 1
    assert s["totals"]["users_by_tier"] == {"participant": 2, "student": 1}
    assert s["feedback"]["positive_rate"] == round(2 / 3, 3)
    assert s["feedback"]["corrected_rate"] == 0.25
    assert [i["id"] for i in s["top_reused"]] == [1, 2, 3]   # unused item 4 left out


def test_empty_database_does_not_crash():
    s = compute_stats([], [], [])
    assert s["ratios"]["reuse"] is None and s["feedback"]["positive_rate"] is None
    assert s["deactivations"] == []


def test_deactivations_grouped_by_day_and_reason():
    items = [
        item(1, "domain_knowledge", active=False, deactivated_at="2026-10-01T09:00", reason="a"),
        item(2, "domain_knowledge", active=False, deactivated_at="2026-10-01T15:00", reason="b"),
        item(3, "intent_calibration", active=False, deactivated_at="2026-09-30T15:00", reason="a"),
    ]
    assert deactivations_by_day(items) == [
        {"date": "2026-09-30", "total": 1, "reasons": {"a": 1}},
        {"date": "2026-10-01", "total": 2, "reasons": {"a": 1, "b": 1}},
    ]
