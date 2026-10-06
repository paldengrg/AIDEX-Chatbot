"""Research metrics for the admin dashboard.

Pure functions over plain dicts (no database), so they are easy to test and
to reuse in notebooks. api_admin.py loads the rows and calls compute_stats().

The headline metric mirrors the Human Layer paper: how often memory items are
actually *reused* (times_used), split into intent_calibration vs
domain_knowledge. The paper reports roughly 3.4 : 1 in production.
"""

from collections import Counter, defaultdict

PAPER_REUSE_RATIO = 3.4
CATEGORIES = ("intent_calibration", "domain_knowledge")


def rate(part: int, whole: int) -> float | None:
    """part / whole as a 0-1 fraction, or None when there is nothing to divide."""
    return round(part / whole, 3) if whole else None


def ratio(a: int, b: int) -> float | None:
    return round(a / b, 2) if b else None


def category_summary(items: list[dict]) -> dict:
    out = {}
    for cat in CATEGORIES:
        rows = [i for i in items if i["category"] == cat]
        total = lambda key: sum(i[key] for i in rows)  # noqa: E731
        used, retrieved = total("times_used"), total("times_retrieved")
        out[cat] = {
            "items": len(rows),
            "active": sum(1 for i in rows if i["active"]),
            "times_retrieved": retrieved,
            "times_used": used,
            "times_helpful": total("times_helpful"),
            "times_corrected": total("times_corrected"),
            "use_rate": rate(used, retrieved),                  # retrieved -> applied
            "helpful_rate": rate(total("times_helpful"), used),  # applied -> thumbs up
            "correction_rate": rate(total("times_corrected"), used),
        }
    return out


def deactivations_by_day(items: list[dict]) -> list[dict]:
    """[{date, total, reasons: {reason: n}}] for items switched off, oldest first."""
    days: dict[str, Counter] = defaultdict(Counter)
    for i in items:
        if not i["active"] and i.get("deactivated_at"):
            days[i["deactivated_at"][:10]][i.get("deactivation_reason") or "unknown"] += 1
    return [{"date": d, "total": sum(c.values()), "reasons": dict(c)}
            for d, c in sorted(days.items())]


def compute_stats(items: list[dict], exchanges: list[dict], users: list[dict],
                  top_n: int = 10) -> dict:
    """Everything the dashboard shows, from plain lists of row dicts."""
    cats = category_summary(items)
    intent, domain = cats["intent_calibration"], cats["domain_knowledge"]
    rated = [e for e in exchanges if e["feedback"] is not None]

    top = sorted((i for i in items if i["times_used"] > 0),
                 key=lambda i: (i["times_used"], i["times_helpful"]), reverse=True)[:top_n]

    return {
        "totals": {
            "memory_items": len(items),
            "active": sum(1 for i in items if i["active"]),
            "inactive": sum(1 for i in items if not i["active"]),
            "users": len(users),
            "users_by_tier": dict(Counter(u["tier"] for u in users)),
            "exchanges": len(exchanges),
        },
        "by_category": cats,
        "ratios": {
            "count": ratio(intent["items"], domain["items"]),          # items stored
            "reuse": ratio(intent["times_used"], domain["times_used"]),  # items reused
            "paper_reuse": PAPER_REUSE_RATIO,
        },
        "by_source": dict(Counter(i["source"] for i in items)),
        "feedback": {
            "rated": len(rated),
            "positive_rate": rate(sum(1 for e in rated if e["feedback"] > 0), len(rated)),
            "corrected_rate": rate(sum(1 for e in exchanges if e["was_corrected"]), len(exchanges)),
        },
        "top_reused": [
            {k: i[k] for k in ("id", "user", "rule_text", "category", "source", "times_used",
                               "times_helpful", "times_corrected", "active")}
            for i in top
        ],
        "deactivations": deactivations_by_day(items),
    }
