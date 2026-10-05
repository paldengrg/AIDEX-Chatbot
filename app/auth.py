"""Accounts: password hashing and login cookies.

Only the Python standard library is used:
  - Passwords are hashed with scrypt (slow on purpose, salted per user), so a
    leaked database doesn't reveal passwords.
  - The login cookie is "user_id.expiry.signature". The signature is an HMAC
    made with SECRET_KEY, so users can't forge or edit their cookie to become
    someone else. The cookie is httponly (JavaScript can't read it).
"""

import base64
import hashlib
import hmac
import logging
import re
import secrets
import time

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import User

AUTH_COOKIE = "aidx_auth"
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
MIN_PASSWORD = 8

if settings.secret_key and not settings.secret_key.startswith("change-me"):
    _SECRET = settings.secret_key.encode()
else:
    # Fine for local testing; everyone is logged out when the server restarts.
    _SECRET = secrets.token_bytes(32)
    logging.getLogger(__name__).warning("SECRET_KEY not set in .env: using a temporary key.")


# --- Passwords -----------------------------------------------------------------

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_b64, digest_b64 = stored.split("$")
        salt, expected = base64.b64decode(salt_b64), base64.b64decode(digest_b64)
    except ValueError:
        return False
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return hmac.compare_digest(digest, expected)  # constant-time comparison


def validate_credentials(username: str, password: str) -> str | None:
    """Return an error message, or None if the username/password are acceptable."""
    if not USERNAME_RE.match(username):
        return "Username must be 3–30 characters: letters, numbers, dot, dash or underscore."
    if len(password) < MIN_PASSWORD:
        return f"Password must be at least {MIN_PASSWORD} characters."
    return None


# --- Signed login tokens -----------------------------------------------------------

def _sign(payload: str) -> str:
    return hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()


def make_token(user_id: int, now: float | None = None) -> str:
    expires = int((time.time() if now is None else now) + settings.login_days * 86400)
    payload = f"{user_id}.{expires}"
    return f"{payload}.{_sign(payload)}"


def read_token(token: str | None, now: float | None = None) -> int | None:
    """Return the user id inside a valid, unexpired token, else None."""
    try:
        user_id, expires, signature = (token or "").split(".")
        payload = f"{user_id}.{expires}"
        if not hmac.compare_digest(signature, _sign(payload)):
            return None
        if int(expires) < (time.time() if now is None else now):
            return None
        return int(user_id)
    except ValueError:
        return None


# --- FastAPI dependencies ----------------------------------------------------------

def current_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    """The logged-in user, or None for anonymous visitors."""
    user_id = read_token(request.cookies.get(AUTH_COOKIE))
    return db.get(User, user_id) if user_id is not None else None


def require_user(user: User | None = Depends(current_user)) -> User:
    if user is None:
        raise HTTPException(401, "Please sign in first.")
    return user
