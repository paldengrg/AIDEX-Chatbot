"""Synthetic users chatting over several days, to demonstrate memory learning.

    python -m scripts.simulate                 # free: offline mock model, 7 days
    python -m scripts.simulate --days 10       # more simulated days
    python -m scripts.simulate --real          # use the real LLM from .env (costs a little)
    python -m scripts.simulate --reset         # delete earlier simulated users first

Five made-up users (usernames start with "sim_") with distinct communication
styles hold one conversation per simulated day and rate some replies. Their
data goes into your normal aidx.db, so open /admin afterwards to see the
dashboard fill up. Four of them join the research study, so their memory text
is visible to the admin; "sim_student_sam" stays a student, so his is hidden.

Honest caveats for your presentation:
  - All of this is synthetic data, and it is labelled as such.
  - Rows are back-dated so the "deactivated over time" chart spans several
    days. That is for demonstration only.
  - In mock mode the "model" always reports the first memory item in the prompt
    as used, so reuse numbers show the pipeline working, not real behaviour.
    Use --real for meaningful numbers.
"""

import argparse
import os
import random
import sys
from datetime import datetime, timedelta, timezone

PASSWORD = "sim-password-123"

# Each persona: whether they join the study, and a pool of messages per day.
# Messages are chosen so both the LLM and the offline heuristic can learn from them.
PERSONAS = {
    "sim_brief_ben": {
        "participant": True,
        "satisfaction": 0.8,   # chance of a thumbs up when memory was used
        "days": [
            ["From now on please keep answers short.", "Who leads the lab?"],
            ["Just answer, don't ask me questions. What are the research themes?"],
            ["What does AIDX stand for?", "How can I join?"],
            ["Please always use bullet points.", "What publications are there?"],
            ["Where is the lab?", "Who is Md Nazmul Hossain?"],
        ],
    },
    "sim_detail_dana": {
        "participant": True,
        "satisfaction": 0.75,
        "days": [
            ["I prefer detailed answers with an example. What is the Human Layer paper about?"],
            ["Please ask me first if you're unsure. Tell me about the research themes."],
            ["What did the IT Boot Camp cover?", "Can you explain the trustworthy AI theme?"],
            ["Please always give a real example from the lab.", "What are the lab's principles?"],
            ["How can industry partners work with the lab?"],
        ],
    },
    "sim_ambiguous_amir": {
        "participant": True,
        "satisfaction": 0.4,
        "days": [
            # Corrections follow a reply in the same conversation, so they count
            # against the memory used in that reply.
            ["Please always answer in one sentence.", "papers?", "No, that's not what I asked."],
            ["papers?", "No, I meant the publications, not the people.", "themes?", "No, wrong."],
            ["When I say papers I mean the lab's publications.", "papers from this year?",
             "No, that's not what I asked."],
            ["people?", "No, wrong."],
            ["papers?"],
        ],
    },
    "sim_facts_faye": {
        "participant": True,
        "satisfaction": 0.85,
        "days": [
            ["I'm a Bachelor of IT student at ACU.", "How can a student get involved?"],
            ["I'm interested in agentic AI and memory.", "What agentic AI research does the lab do?"],
            ["My thesis is on AI in education.", "Which publications are about education?"],
            ["What internships are there for IT students?"],
            ["Tell me more about agentic AI memory research."],
        ],
    },
    "sim_student_sam": {
        "participant": False,
        "satisfaction": 0.7,
        "days": [
            ["Please keep answers under 80 words.", "What does the lab research?"],
            ["I'm a business student.", "Who leads the lab?"],
            ["How do I contact the lab?"],
            ["From now on please use plain English.", "What is decision intelligence?"],
            ["What news does the lab have?"],
        ],
    },
}


def configure(real: bool) -> None:
    """Settings that must be in place before the app is imported."""
    os.environ["RATE_LIMIT_CHAT_PER_MINUTE"] = "100000"
    os.environ["RATE_LIMIT_AUTH_PER_15MIN"] = "100000"
    if not real:
        os.environ["LLM_PROVIDER"] = "mock"


def reset_simulated_users() -> int:
    """Delete every sim_ user and all of their data."""
    from sqlalchemy import select

    from app import memory
    from app.api_privacy import delete_research_data
    from app.db import SessionLocal, init_db
    from app.models import User

    init_db()
    with SessionLocal() as db:
        users = list(db.scalars(select(User).where(User.username.like("sim\\_%", escape="\\"))))
        for user in users:
            delete_research_data(db, user)
            memory.delete_all(db, user)
            db.delete(user)
        db.commit()
    return len(users)


def sign_in(client, username: str, participant: bool) -> None:
    res = client.post("/api/auth/register", json={"username": username, "password": PASSWORD})
    if res.status_code == 409:  # already exists from an earlier run
        client.post("/api/auth/login", json={"username": username, "password": PASSWORD})
    client.patch("/api/me", json={"memory_enabled": True})
    if participant:
        version = client.get("/api/research/info").json()["consent_version"]
        client.post("/api/research/consent", json={
            "consent_version": version, "agree_information": True,
            "agree_logging": True, "agree_withdrawal": True})


# --- Back-dating, so one run looks like several days of use ----------------------------

def _snapshot() -> dict:
    from sqlalchemy import func, select

    from app.db import SessionLocal
    from app.models import ConversationLog, Exchange, MemoryItem

    with SessionLocal() as db:
        return {table: db.scalar(select(func.coalesce(func.max(model.id), 0)))
                for table, model in (("memory_items", MemoryItem), ("exchanges", Exchange),
                                     ("conversation_logs", ConversationLog))}


def _backdate(before: dict, started: datetime, days_ago: int) -> None:
    """Move everything created or changed since `started` back by `days_ago` days."""
    if days_ago <= 0:
        return
    from sqlalchemy import text

    from app.db import engine

    shift = f"-{days_ago} days"
    since = started.strftime("%Y-%m-%d %H:%M:%S")
    with engine.begin() as conn:
        for table, last_id in before.items():
            conn.execute(text(f"UPDATE {table} SET created_at = datetime(created_at, :s) "
                              f"WHERE id > :last"), {"s": shift, "last": last_id})
        for column in ("deactivated_at", "last_used_at"):
            conn.execute(text(f"UPDATE memory_items SET {column} = datetime({column}, :s) "
                              f"WHERE {column} >= :since"), {"s": shift, "since": since})


# --- The simulation -----------------------------------------------------------------------

def simulate(days: int, seed: int) -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    rng = random.Random(seed)
    stats = {"messages": 0, "learned": 0, "thumbs_up": 0, "thumbs_down": 0}

    with TestClient(app):  # startup: tables + document index
        clients = {}
        for name, persona in PERSONAS.items():
            clients[name] = TestClient(app)
            sign_in(clients[name], name, persona["participant"])

        for day in range(days):
            started = datetime.now(timezone.utc) - timedelta(seconds=1)
            before = _snapshot()
            for name, persona in PERSONAS.items():
                client = clients[name]
                client.post("/api/reset")  # a new conversation each day
                for message in persona["days"][day % len(persona["days"])]:
                    body = client.post("/api/chat", json={"message": message}).json()
                    stats["messages"] += 1
                    stats["learned"] += len(body.get("learned", []))
                    if body.get("exchange_id") and rng.random() < 0.7:  # not every reply is rated
                        happy = rng.random() < (persona["satisfaction"] if body["memory_used"] else 0.5)
                        client.post("/api/feedback", json={"exchange_id": body["exchange_id"],
                                                           "rating": 1 if happy else -1})
                        stats["thumbs_up" if happy else "thumbs_down"] += 1
            _backdate(before, started, days_ago=days - 1 - day)
            print(f"Day {day + 1}/{days} done")

    print(f"\n{len(PERSONAS)} synthetic users, {stats['messages']} messages, "
          f"{stats['learned']} memories learned, {stats['thumbs_up']} thumbs up, "
          f"{stats['thumbs_down']} thumbs down.")
    print("Open http://127.0.0.1:8000/admin to see the results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate synthetic users (see module docstring).")
    parser.add_argument("--days", type=int, default=7, help="simulated days (default 7, max 30)")
    parser.add_argument("--real", action="store_true", help="use the real LLM from .env")
    parser.add_argument("--reset", action="store_true", help="delete earlier sim_ users first")
    parser.add_argument("--seed", type=int, default=42, help="random seed for repeatable runs")
    args = parser.parse_args()
    if not 1 <= args.days <= 30:
        sys.exit("--days must be between 1 and 30")

    configure(args.real)
    if args.reset:
        print(f"Removed {reset_simulated_users()} earlier simulated users.")
    simulate(args.days, args.seed)
