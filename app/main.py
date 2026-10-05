"""FastAPI web server: serves the chat page and the JSON API.

Run with:   uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import chat, memory, rag
from app.api_account import router as account_router
from app.auth import current_user
from app.config import BASE_DIR, settings
from app.db import get_db, init_db
from app.llm_client import LLMError
from app.models import User
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
app.include_router(account_router)  # /api/auth/*, /api/me, /api/memory


# --- Pages -------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    # The assistant's name is configurable, so we fill it into the page here.
    html = (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")
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
    memory_used: list[dict]  # personal memory items that were in the prompt


@app.post("/api/chat", response_model=ChatResponse)
def post_chat(body: ChatRequest, request: Request, response: Response,
              user: User | None = Depends(current_user),
              db: Session = Depends(get_db)) -> ChatResponse:
    # A plain `def` (not async) endpoint: FastAPI runs it in a worker thread,
    # so a slow model call doesn't block other visitors.
    message = body.message.strip()
    if not message:
        raise HTTPException(422, "Message is empty.")

    # Conversation history (short-term, in RAM) is kept for everyone. Personal
    # memory (long-term, in the database) only for signed-in users who opted in.
    session_id, session = sessions.get_or_create(request.cookies.get(SESSION_COOKIE))
    items = [i.to_dict() for i in memory.select_for_prompt(db, user, message)]
    try:
        reply, sources = chat.answer(message, sessions.history(session), items)
    except LLMError:
        raise HTTPException(503, "The assistant is temporarily unavailable. Please try again.")

    sessions.add_exchange(session, message, reply)
    response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="lax",
                        max_age=settings.session_ttl_minutes * 60)
    memory_used = [{"id": i["id"], "rule_text": i["rule_text"], "category": i["category"]}
                   for i in items]
    return ChatResponse(reply=reply, sources=sources, memory_used=memory_used)


@app.post("/api/reset", status_code=204)
def post_reset(request: Request, response: Response) -> None:
    """Start a new conversation: forget this browser's session memory."""
    sessions.reset(request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
