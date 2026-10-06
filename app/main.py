"""FastAPI web server: serves the chat page and the JSON API.

Run with:   uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import chat, guardrails, learning, memory, rag
from app.api_account import router as account_router
from app.api_admin import router as admin_router
from app.api_privacy import router as privacy_router
from app.auth import current_user
from app.config import BASE_DIR, settings
from app.db import get_db, init_db
from app.llm_client import LLMError
from app.models import ConversationLog, User
from app.session_memory import SESSION_COOKIE, sessions


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Create database tables if needed, and open the document index (built on
    # first run) so the first visitor doesn't wait for the embedding model.
    init_db()
    rag.get_collection()
    yield


app = FastAPI(title=settings.assistant_name, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/assets", StaticFiles(directory=BASE_DIR / "assets"), name="assets")
app.include_router(account_router)  # /api/auth/*, /api/me, /api/memory, /api/feedback
app.include_router(privacy_router)  # /api/research/*, /api/me/export, DELETE /api/me
app.include_router(admin_router)    # /api/admin/*


# --- Pages -------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    # The assistant's name is configurable, so we fill it into the page here.
    html = (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")
    return html.replace("{{ASSISTANT_NAME}}", settings.assistant_name)


@app.get("/admin", response_class=HTMLResponse)
def admin_page() -> str:
    # The page itself is public; all data behind it needs the admin cookie.
    html = (BASE_DIR / "static" / "admin.html").read_text(encoding="utf-8")
    return html.replace("{{ASSISTANT_NAME}}", settings.assistant_name)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(BASE_DIR / "assets" / "favicon.ico")


# --- API ---------------------------------------------------------------------

class ChatRequest(BaseModel):
    # Field limits reject empty or oversized input before it reaches the model.
    message: str = Field(min_length=1, max_length=settings.max_message_chars)


class ChatResponse(BaseModel):
    reply: str
    sources: list[dict]
    memory_used: list[dict]      # personal memory items that shaped this reply
    learned: list[dict]          # memory items learned from this message
    exchange_id: int | None      # for thumbs up/down (signed-in + memory on only)
    logged: bool = False         # true when saved to the research log (participants)
    guardrail: str | None = None # set when a fixed safety reply was used


def log_for_research(db: Session, user: User | None, exchange_id: int | None, message: str,
                     reply: str, sources: list[dict], guardrail: str | None = None) -> bool:
    """Save the full turn, but only for research participants who consented."""
    if user is None or not user.is_participant:
        return False
    db.add(ConversationLog(
        user_id=user.pseudonym_id, exchange_id=exchange_id, user_message=message,
        assistant_reply=reply, sources=", ".join(s["source"] for s in sources),
        guardrail=guardrail, synthetic=not settings.ethics_approved,
    ))
    db.commit()
    return True


@app.post("/api/chat", response_model=ChatResponse)
def post_chat(body: ChatRequest, request: Request, response: Response,
              user: User | None = Depends(current_user),
              db: Session = Depends(get_db)) -> ChatResponse:
    # A plain `def` (not async) endpoint: FastAPI runs it in a worker thread,
    # so a slow model call doesn't block other visitors.
    guardrails.enforce(guardrails.chat_limiter, guardrails.client_key(request, user))
    message = guardrails.sanitize_message(body.message)
    if not message:
        raise HTTPException(422, "Message is empty.")

    # Safety check first: injection attempts and crisis messages get a fixed
    # reply. Nothing is learned from them and the model is never called.
    guard = guardrails.check_message(message)
    if guard.blocked:
        # Out of care for the person, a message that triggered the crisis reply
        # is not stored word for word, even for research participants.
        logged_text = "[withheld: crisis support shown]" if guard.reason == "crisis" else message
        logged = log_for_research(db, user, None, logged_text, guard.reply, [], guard.reason)
        return ChatResponse(reply=guard.reply, sources=[], memory_used=[], learned=[],
                            exchange_id=None, logged=logged, guardrail=guard.reason)

    # Conversation history (short-term, in RAM) is kept for everyone. Personal
    # memory (long-term, in the database) only for signed-in users who opted in.
    session_id, session = sessions.get_or_create(request.cookies.get(SESSION_COOKIE))
    history = sessions.history(session)
    remembering = user is not None and user.memory_enabled

    learned = []
    if remembering:
        # Learn from this message *before* answering, so a new preference
        # ("use bullet points from now on") already applies to this reply.
        previous_reply = next((t["content"] for t in reversed(history)
                               if t["role"] == "assistant"), "")
        learned = learning.learn_from_message(db, user, message, previous_reply,
                                              session.last_exchange_id)

    items = [i.to_dict() for i in memory.select_for_prompt(db, user, message)]
    style = learning.clarify_style(user.clarify_score) if remembering else "neutral"
    try:
        reply, sources, used_ids = chat.answer(message, history, items, style)
    except LLMError:
        raise HTTPException(503, "The assistant is temporarily unavailable. Please try again.")

    exchange_id = None
    if remembering:
        exchange = learning.record_exchange(db, user, [i["id"] for i in items], used_ids, reply)
        exchange_id = session.last_exchange_id = exchange.id
    logged = log_for_research(db, user, exchange_id, message, reply, sources)

    sessions.add_exchange(session, message, reply)
    response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="lax",
                        max_age=settings.session_ttl_minutes * 60)
    brief = lambda i: {"id": i["id"], "rule_text": i["rule_text"], "category": i["category"]}  # noqa: E731
    return ChatResponse(
        reply=reply,
        sources=sources,
        memory_used=[brief(i) for i in items if i["id"] in used_ids],
        learned=[brief(i.to_dict()) for i in learned],
        exchange_id=exchange_id,
        logged=logged,
    )


@app.post("/api/reset", status_code=204)
def post_reset(request: Request, response: Response) -> None:
    """Start a new conversation: forget this browser's session memory."""
    sessions.reset(request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
