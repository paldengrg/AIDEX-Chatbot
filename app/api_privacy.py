"""Privacy and research-consent routes.

  - Research participants opt in through an explicit consent screen, which
    turns on full conversation logging (see ConversationLog).
  - Anyone can withdraw at any time. Withdrawing deletes their research data.
  - Every signed-in user can download all their data or delete their account.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app import memory
from app.auth import AUTH_COOKIE, require_user
from app.config import settings
from app.db import get_db
from app.models import ConversationLog, Exchange, MemoryItem, User, utcnow
from app.session_memory import SESSION_COOKIE, sessions

router = APIRouter(prefix="/api")

# Bump this whenever the participant information text in index.html changes,
# so we always know which version each person agreed to.
CONSENT_VERSION = "0.1-prototype"


class ConsentRequest(BaseModel):
    consent_version: str
    agree_information: bool    # "I have read the participant information"
    agree_logging: bool        # "I agree to my conversations being logged"
    agree_withdrawal: bool     # "I understand I can withdraw at any time"


class WithdrawRequest(BaseModel):
    delete_memory: bool = False   # also delete personal memory items?


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def delete_research_data(db: Session, user: User) -> int:
    """Delete conversation logs and exchange records. Returns rows removed."""
    logs = db.execute(delete(ConversationLog).where(ConversationLog.user_id == user.pseudonym_id))
    exchanges = db.execute(delete(Exchange).where(Exchange.user_id == user.pseudonym_id))
    db.commit()
    return logs.rowcount + exchanges.rowcount


# --- Research consent ------------------------------------------------------------------

@router.get("/research/info")
def research_info() -> dict:
    return {"consent_version": CONSENT_VERSION, "ethics_approved": settings.ethics_approved}


@router.post("/research/consent")
def give_consent(body: ConsentRequest, user: User = Depends(require_user),
                 db: Session = Depends(get_db)) -> dict:
    if body.consent_version != CONSENT_VERSION:
        raise HTTPException(409, "The participant information has changed. Please reload the page.")
    if not (body.agree_information and body.agree_logging and body.agree_withdrawal):
        raise HTTPException(400, "Please tick all three boxes to take part.")
    user.tier = "participant"
    user.consent_given_at = utcnow()
    user.consent_version = CONSENT_VERSION
    user.consent_withdrawn_at = None
    user.memory_enabled = True   # participants get persistent memory
    db.commit()
    return {"tier": user.tier, "consent_given_at": _iso(user.consent_given_at)}


@router.post("/research/withdraw")
def withdraw(body: WithdrawRequest, user: User = Depends(require_user),
             db: Session = Depends(get_db)) -> dict:
    """Stop taking part and delete research data (optionally memory too)."""
    removed = delete_research_data(db, user)
    if body.delete_memory:
        removed += memory.delete_all(db, user)
    user.tier = "student"
    user.consent_given_at = None
    user.consent_version = None
    user.consent_withdrawn_at = utcnow()
    db.commit()
    return {"tier": user.tier, "deleted_records": removed}


# --- Your data ----------------------------------------------------------------------------

@router.get("/me/export")
def export_my_data(user: User = Depends(require_user), db: Session = Depends(get_db)):
    """Everything stored about the signed-in user, as a downloadable JSON file."""
    mine = lambda model: db.scalars(select(model).where(model.user_id == user.pseudonym_id))  # noqa: E731
    data = {
        "exported_at": _iso(utcnow()),
        "account": {
            "username": user.username,
            "tier": user.tier,
            "memory_enabled": user.memory_enabled,
            "clarify_score": user.clarify_score,
            "created_at": _iso(user.created_at),
            "consent_given_at": _iso(user.consent_given_at),
            "consent_version": user.consent_version,
            "consent_withdrawn_at": _iso(user.consent_withdrawn_at),
        },
        "memory_items": [i.to_dict() for i in mine(MemoryItem)],
        "exchanges": [
            {"id": e.id, "created_at": _iso(e.created_at), "used_memory_ids": e.used_ids,
             "feedback": e.feedback, "was_corrected": e.was_corrected}
            for e in mine(Exchange)
        ],
        "conversation_logs": [
            {"created_at": _iso(c.created_at), "user_message": c.user_message,
             "assistant_reply": c.assistant_reply, "synthetic": c.synthetic}
            for c in mine(ConversationLog)
        ],
    }
    return JSONResponse(data, headers={
        "Content-Disposition": 'attachment; filename="aidx-my-data.json"'})


@router.delete("/me", status_code=204)
def delete_account(request: Request, response: Response, user: User = Depends(require_user),
                   db: Session = Depends(get_db)) -> None:
    """Permanently delete the account and everything linked to it."""
    delete_research_data(db, user)
    memory.delete_all(db, user)
    db.delete(user)
    db.commit()
    sessions.reset(request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE)
    response.delete_cookie(AUTH_COOKIE)
