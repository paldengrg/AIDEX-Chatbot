"""API routes for accounts and the "What I've learned about you" memory panel.

Every memory route depends on `require_user`, and every memory function is
scoped to that user, so a request can only ever see or change its own items.
"""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import memory
from app.auth import (AUTH_COOKIE, current_user, hash_password, make_token, require_user,
                      validate_credentials, verify_password)
from app.config import settings
from app.db import get_db
from app.models import User
from app.session_memory import SESSION_COOKIE, sessions

router = APIRouter(prefix="/api")


# --- Request bodies ------------------------------------------------------------------

class Credentials(BaseModel):
    username: str = Field(max_length=30)
    password: str = Field(max_length=200)


class Preferences(BaseModel):
    memory_enabled: bool


class NewMemory(BaseModel):
    rule_text: str = Field(max_length=1000)
    category: Literal["intent_calibration", "domain_knowledge"] = "intent_calibration"


class MemoryUpdate(BaseModel):
    rule_text: str | None = Field(default=None, max_length=1000)
    category: Literal["intent_calibration", "domain_knowledge"] | None = None
    active: bool | None = None


def _profile(user: User) -> dict:
    return {"username": user.username, "tier": user.tier, "memory_enabled": user.memory_enabled}


def _start_fresh_conversation(request: Request, response: Response) -> None:
    """Forget the in-RAM chat so one person's conversation never carries over
    to another account on a shared computer."""
    sessions.reset(request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE)


def _log_in(request: Request, response: Response, user: User) -> None:
    _start_fresh_conversation(request, response)
    response.set_cookie(AUTH_COOKIE, make_token(user.id), httponly=True, samesite="lax",
                        secure=settings.cookie_secure, max_age=settings.login_days * 86400)


# --- Accounts --------------------------------------------------------------------------

@router.post("/auth/register", status_code=201)
def register(body: Credentials, request: Request, response: Response, db: Session = Depends(get_db)) -> dict:
    username = body.username.strip()
    error = validate_credentials(username, body.password)
    if error:
        raise HTTPException(400, error)
    if db.scalar(select(User).where(User.username == username)):
        raise HTTPException(409, "That username is taken.")
    user = User(username=username, password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    _log_in(request, response, user)
    return _profile(user)


@router.post("/auth/login")
def login(body: Credentials, request: Request, response: Response, db: Session = Depends(get_db)) -> dict:
    user = db.scalar(select(User).where(User.username == body.username.strip()))
    # Same message for "no such user" and "wrong password", so attackers can't
    # discover which usernames exist.
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Incorrect username or password.")
    _log_in(request, response, user)
    return _profile(user)


@router.post("/auth/logout", status_code=204)
def logout(request: Request, response: Response) -> None:
    _start_fresh_conversation(request, response)
    response.delete_cookie(AUTH_COOKIE)


@router.get("/me")
def me(user: User | None = Depends(current_user)) -> dict:
    return {"user": _profile(user) if user else None}


@router.patch("/me")
def update_me(body: Preferences, user: User = Depends(require_user),
              db: Session = Depends(get_db)) -> dict:
    """Opt in or out of personal memory. Opting out keeps items but stops using them."""
    user.memory_enabled = body.memory_enabled
    db.commit()
    return _profile(user)


# --- Memory items ----------------------------------------------------------------------

@router.get("/memory")
def get_memory(user: User = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    return {"items": [i.to_dict() for i in memory.list_items(db, user)]}


@router.post("/memory", status_code=201)
def create_memory(body: NewMemory, user: User = Depends(require_user),
                  db: Session = Depends(get_db)) -> dict:
    try:
        item, created = memory.add_item(db, user, body.rule_text, body.category)
    except memory.MemoryInputError as exc:
        raise HTTPException(400, str(exc))
    return {"item": item.to_dict(), "created": created}


@router.patch("/memory/{item_id}")
def edit_memory(item_id: int, body: MemoryUpdate, user: User = Depends(require_user),
                db: Session = Depends(get_db)) -> dict:
    try:
        item = memory.update_item(db, user, item_id, body.rule_text, body.category, body.active)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except memory.MemoryInputError as exc:
        raise HTTPException(400, str(exc))
    return {"item": item.to_dict()}


@router.delete("/memory/{item_id}", status_code=204)
def remove_memory(item_id: int, user: User = Depends(require_user),
                  db: Session = Depends(get_db)) -> None:
    try:
        memory.delete_item(db, user, item_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc))


@router.delete("/memory")
def remove_all_memory(user: User = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    return {"deleted": memory.delete_all(db, user)}
