"""Run the evaluation set and report accuracy.

    python -m eval.run_eval            # uses the LLM settings in .env (costs a little)
    python -m eval.run_eval --mock     # free, offline: checks the plumbing only

Each question in eval/questions.json runs as a fresh user of the right tier
(anonymous, student with memory on, or consenting research participant)
against the real app, in-process, with a throwaway database. Your real
aidx.db is never touched.

Scoring is deliberately simple and transparent:
  keywords        reply contains at least `min_matches` of `any` (default 1)
                  and none of `none` (case-insensitive)
  declines        reply politely refuses (looks for refusal phrases)
  guardrail       the expected guardrail fired ("injection" / "crisis")
  learned         a memory item of the expected category was learned
  nothing_learned nothing was learned (e.g. anonymous visitors)
  memory_used     at least one personal memory item shaped the reply
  logged / not_logged   research logging happened only for participants
"""

import argparse
import csv
import json
import os
import sys
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECLINE_PHRASES = ("can't", "cannot", "can not", "unable", "not able", "only help",
                   "outside", "not something i", "i'm here to help with", "focus on")


def configure(mock: bool) -> None:
    """Point the app at a throwaway database BEFORE it is imported."""
    tmp = tempfile.mkdtemp(prefix="aidx-eval-")
    os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(tmp, "eval.db").replace("\\", "/")
    os.environ["RATE_LIMIT_CHAT_PER_MINUTE"] = "100000"
    os.environ["RATE_LIMIT_AUTH_PER_15MIN"] = "100000"
    if mock:
        os.environ["LLM_PROVIDER"] = "mock"


def new_client(tier: str):
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    if tier in ("student", "participant"):
        name = "eval_" + uuid.uuid4().hex[:8]
        client.post("/api/auth/register", json={"username": name, "password": "eval-password"})
        client.patch("/api/me", json={"memory_enabled": True})
    if tier == "participant":
        version = client.get("/api/research/info").json()["consent_version"]
        client.post("/api/research/consent", json={
            "consent_version": version, "agree_information": True,
            "agree_logging": True, "agree_withdrawal": True})
    return client


def score(case: dict, body: dict) -> tuple[bool, str]:
    """Return (passed, reason) for one response."""
    reply = (body.get("reply") or "").lower()
    check = case["check"]

    if check == "keywords":
        found = [k for k in case["any"] if k.lower() in reply]
        banned = [k for k in case.get("none", []) if k.lower() in reply]
        need = case.get("min_matches", 1)
        if banned:
            return False, f"mentions {banned}"
        return len(found) >= need, f"found {found or 'none'} (need {need})"
    if check == "declines":
        hit = [p for p in DECLINE_PHRASES if p in reply]
        return bool(hit), f"refusal phrases {hit or 'none'}"
    if check == "guardrail":
        return body.get("guardrail") == case["expected"], f"guardrail={body.get('guardrail')}"
    if check == "learned":
        cats = [m["category"] for m in body.get("learned", [])]
        return case["expected"] in cats, f"learned {cats or 'nothing'}"
    if check == "nothing_learned":
        return body.get("learned") == [], f"learned {body.get('learned')}"
    if check == "memory_used":
        used = body.get("memory_used", [])
        return bool(used), f"{len(used)} memory item(s) used"
    if check == "logged":
        return body.get("logged") is True, f"logged={body.get('logged')}"
    if check == "not_logged":
        return body.get("logged") is False, f"logged={body.get('logged')}"
    return False, f"unknown check '{check}'"


def run(mock: bool) -> int:
    configure(mock)
    from fastapi.testclient import TestClient
    from app.config import settings
    from app.main import app

    cases = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    print(f"Running {len(cases)} questions with LLM provider '{settings.llm_provider}'\n")

    results = []
    with TestClient(app):  # runs startup once (tables, document index)
        for case in cases:
            client = new_client(case["tier"])
            for message in case.get("setup", []):
                client.post("/api/chat", json={"message": message})
            res = client.post("/api/chat", json={"message": case["question"]})
            body = res.json() if res.status_code == 200 else {"reply": f"HTTP {res.status_code}"}
            passed, reason = score(case, body)
            results.append({**case, "passed": passed, "reason": reason,
                            "reply": body.get("reply", "")[:300]})
            print(f"{'PASS' if passed else 'FAIL'}  {case['id']:<4} {case['tier']:<11} "
                  f"{case['question'][:55]:<55}  {reason}")

    # --- Summary ---
    def summary(key: str) -> None:
        groups = defaultdict(list)
        for r in results:
            groups[r[key]].append(r["passed"])
        print(f"\nAccuracy by {key}:")
        for name, passes in sorted(groups.items()):
            print(f"  {name:<14} {sum(passes):>2}/{len(passes):<2}  {sum(passes) / len(passes):.0%}")

    total = sum(r["passed"] for r in results)
    summary("tier")
    summary("category")
    print(f"\nOVERALL: {total}/{len(results)} = {total / len(results):.0%}")
    if mock:
        print("(Mock mode checks the plumbing. Run without --mock for real answer quality.)")

    out = HERE / "results.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        fields = ["id", "tier", "category", "question", "check", "passed", "reason", "reply"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Details saved to {out}")
    return 0 if total == len(results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mock", action="store_true", help="use the free offline mock model")
    sys.exit(run(parser.parse_args().mock))
