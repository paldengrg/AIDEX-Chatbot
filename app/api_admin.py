"""Admin dashboard API: research metrics and CSV export.

Privacy rules for admins:
  - Users appear only as a short pseudonym (first 8 characters), never a username.
  - The text of a memory item is shown only if its owner is a consenting
    research participant. Students' memories are counted but their text stays
    hidden ("[hidden: not a research participant]").
"""

import csv
import io

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import analytics, guardrails
from app.auth import ADMIN_COOKIE, ADMIN_HOURS, check_admin_login, make_admin_token, require_admin
from app.config import settings
from app.db import get_db
from app.models import ConversationLog, Exchange, MemoryItem, User

router = APIRouter(prefix="/api/admin")
HIDDEN = "[hidden: not a research participant]"


class AdminLogin(BaseModel):
    username: str
    password: str


@router.post("/login", status_code=204)
def admin_login(body: AdminLogin, request: Request, response: Response) -> None:
    guardrails.enforce(guardrails.auth_limiter, guardrails.client_key(request))
    if not settings.admin_password:
        raise HTTPException(503, "Admin dashboard is disabled: set ADMIN_PASSWORD in .env.")
    if not check_admin_login(body.username, body.password):
        raise HTTPException(401, "Incorrect admin username or password.")
    response.set_cookie(ADMIN_COOKIE, make_admin_token(), httponly=True, samesite="strict",
                        secure=settings.cookie_secure, max_age=ADMIN_HOURS * 3600)


@router.post("/logout", status_code=204)
def admin_logout(response: Response) -> None:
    response.delete_cookie(ADMIN_COOKIE)


# --- Loading rows as plain dicts (with the privacy rules applied) ---------------------

def load_items(db: Session) -> list[dict]:
    participants = {u.pseudonym_id for u in db.scalars(select(User)) if u.is_participant}
    rows = []
    for item in db.scalars(select(MemoryItem)):
        row = item.to_dict()
        row["user"] = item.user_id[:8]
        if item.user_id not in participants:
            row["rule_text"] = HIDDEN
        rows.append(row)
    return rows


def load_exchanges(db: Session) -> list[dict]:
    return [{"id": e.id, "user": e.user_id[:8], "created_at": e.created_at.isoformat(),
             "retrieved_ids": e.retrieved_ids, "used_ids": e.used_ids,
             "asked_clarification": e.asked_clarification, "feedback": e.feedback,
             "was_corrected": e.was_corrected}
            for e in db.scalars(select(Exchange))]


def load_users(db: Session) -> list[dict]:
    return [{"user": u.pseudonym_id[:8], "tier": u.tier} for u in db.scalars(select(User))]


# --- Routes -------------------------------------------------------------------------------

@router.get("/stats", dependencies=[Depends(require_admin)])
def stats(db: Session = Depends(get_db)) -> dict:
    result = analytics.compute_stats(load_items(db), load_exchanges(db), load_users(db))
    has_logs = db.scalar(select(ConversationLog.id).limit(1)) is not None
    synthetic = db.scalar(select(ConversationLog.id).where(ConversationLog.synthetic).limit(1))
    result["data_notice"] = {
        "ethics_approved": settings.ethics_approved,
        "contains_synthetic": (synthetic is not None) or not settings.ethics_approved,
        "has_research_logs": has_logs,
    }
    return result


EXPORTS = {
    "memory_items": (load_items, ["id", "user", "category", "source", "rule_text", "created_at",
                                  "times_retrieved", "times_used", "times_helpful",
                                  "times_corrected", "last_used_at", "active",
                                  "deactivated_at", "deactivation_reason"]),
    "exchanges": (load_exchanges, ["id", "user", "created_at", "retrieved_ids", "used_ids",
                                   "asked_clarification", "feedback", "was_corrected"]),
}


@router.get("/export/{table}.csv", dependencies=[Depends(require_admin)])
def export_csv(table: str, db: Session = Depends(get_db)) -> Response:
    """Download a table as CSV for analysis in Excel, R or Python."""
    if table not in EXPORTS:
        raise HTTPException(404, f"Unknown table. Choose one of: {', '.join(EXPORTS)}")
    loader, columns = EXPORTS[table]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(loader(db))
    return Response(buffer.getvalue(), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="aidex-{table}.csv"'})
