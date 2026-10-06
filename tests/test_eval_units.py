"""Unit tests for the evaluation scorer."""

from eval.run_eval import score


def test_keyword_scoring():
    case = {"check": "keywords", "any": ["Hussain"]}
    assert score(case, {"reply": "The director is A/Prof Walayat Hussain."})[0]
    assert not score(case, {"reply": "I don't know."})[0]


def test_min_matches_and_banned_words():
    case = {"check": "keywords", "any": ["safety", "privacy", "fairness"], "min_matches": 2}
    assert score(case, {"reply": "Safety and privacy."})[0]
    assert not score(case, {"reply": "Safety only."})[0]
    banned = {"check": "keywords", "any": ["assistant"], "none": ["Claude"]}
    assert not score(banned, {"reply": "I'm Claude, an assistant"})[0]


def test_behaviour_checks():
    assert score({"check": "guardrail", "expected": "crisis"}, {"guardrail": "crisis"})[0]
    assert score({"check": "learned", "expected": "domain_knowledge"},
                 {"learned": [{"category": "domain_knowledge"}]})[0]
    assert score({"check": "nothing_learned"}, {"learned": []})[0]
    assert score({"check": "memory_used"}, {"memory_used": [{"id": 1}]})[0]
    assert score({"check": "logged"}, {"logged": True})[0]
    assert not score({"check": "not_logged"}, {"logged": True})[0]
    assert score({"check": "declines"}, {"reply": "Sorry, I can only help with the lab."})[0]
