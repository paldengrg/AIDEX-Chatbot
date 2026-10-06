"""Unit tests for the self-improvement loop's pure logic (no database)."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.chat import build_system_prompt, split_used_memory
from app.learning import clarify_style, deactivation_reason, heuristic_extract, parse_extraction

NOW = datetime(2026, 10, 5, tzinfo=timezone.utc)


# --- Extraction ---------------------------------------------------------------------

def test_heuristic_finds_explicit_preference():
    result = heuristic_extract("From now on please keep answers under 100 words.")
    assert result.items[0]["category"] == "intent_calibration"
    assert result.items[0]["source"] == "explicit_instruction"
    assert not result.is_correction


def test_heuristic_finds_correction_and_meaning():
    result = heuristic_extract("No, I meant the publications, not the people.",
                               previous_reply="Here are the lab's people...")
    assert result.is_correction
    assert result.items[0]["source"] == "correction"


def test_heuristic_finds_fact_about_user():
    result = heuristic_extract("I'm a Bachelor of IT student. What can I join?")
    assert result.items[0]["category"] == "domain_knowledge"


def test_heuristic_ignores_plain_questions_and_sensitive_info():
    assert heuristic_extract("Who leads the lab?").items == []
    assert heuristic_extract("I'm a patient with a medical condition").items == []


def test_heuristic_clarify_signals():
    assert heuristic_extract("Just answer, don't ask me questions").clarify_signal == "wants_fewer"
    assert heuristic_extract("Please ask me first if you're unsure").clarify_signal == "wants_more"


def test_parse_extraction_accepts_valid_json_with_extra_text():
    raw = 'Sure: {"is_correction": true, "items": [{"rule_text": "Prefers tables", ' \
          '"category": "intent_calibration", "source": "explicit_instruction"}], ' \
          '"clarify_signal": "wants_fewer"}'
    result = parse_extraction(raw)
    assert result.is_correction and result.clarify_signal == "wants_fewer"
    assert result.items == [{"rule_text": "Prefers tables", "category": "intent_calibration",
                             "source": "explicit_instruction"}]


def test_parse_extraction_survives_garbage():
    for raw in ["", "not json", "{broken", '{"items": "nope"}',
                '{"items": [{"rule_text": "x", "category": "made_up"}]}']:
        result = parse_extraction(raw)
        assert result.items == [] and not result.is_correction


# --- Usage tracking -----------------------------------------------------------------

def test_used_memory_line_is_removed_and_parsed():
    reply, used = split_used_memory("Here you go.\nUSED_MEMORY: M3, M9, M3", {3, 9})
    assert reply == "Here you go." and used == [3, 9]


def test_used_memory_ignores_ids_not_in_prompt():
    _, used = split_used_memory("Hi\nUSED_MEMORY: M3, M999", {3})
    assert used == [3]   # M999 belonged to nobody in this prompt


def test_used_memory_none_or_missing():
    assert split_used_memory("Hi\nUSED_MEMORY: none", {1}) == ("Hi", [])
    assert split_used_memory("Hi", {1}) == ("Hi", [])


def test_prompt_labels_memory_items_and_clarify_style():
    mem = [{"id": 7, "category": "intent_calibration", "rule_text": "Prefers tables"}]
    prompt = build_system_prompt([], mem, "act")
    assert "[M7]" in prompt and "USED_MEMORY" in prompt
    assert "prefers direct answers" in prompt


def test_clarify_style_thresholds():
    assert clarify_style(0.1) == "act"
    assert clarify_style(0.5) == "neutral"
    assert clarify_style(0.9) == "ask"


# --- Deactivation review ------------------------------------------------------------

def item(**overrides):
    base = dict(times_corrected=0, times_helpful=0, times_retrieved=0, times_used=0,
                last_used_at=NOW, created_at=NOW)
    return SimpleNamespace(**{**base, **overrides})


def test_healthy_item_is_kept():
    assert deactivation_reason(item(times_used=5, times_helpful=2), NOW) is None


def test_repeatedly_corrected_item_is_switched_off():
    assert deactivation_reason(item(times_corrected=3, times_helpful=1), NOW) == \
        "repeatedly linked to corrections"
    # ...unless it has helped more often than it was corrected
    assert deactivation_reason(item(times_corrected=3, times_helpful=5), NOW) is None


def test_never_used_item_is_switched_off():
    assert deactivation_reason(item(times_retrieved=10, times_used=0), NOW) == \
        "retrieved often but never used"


def test_stale_item_is_switched_off():
    old = NOW - timedelta(days=120)
    assert "not used" in deactivation_reason(item(last_used_at=old, created_at=old), NOW)
