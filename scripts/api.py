"""
api.py - the lending pipeline as a FastAPI service (same pattern as the notebook's library assistant).

    GET  /health                          liveness check (no LLM call)
    POST /policy/ask                      policy question -> answer + sources   (guardrails, like /chat)
    POST /applications                    run one application through loan_workflow
    GET  /applications/{id}               its status: route, grade, DSCR, AI review, decision
    GET  /reviews/pending                 applications waiting for an officer
    POST /applications/{id}/review        officer decides: approve / decline / refer

Run:   uvicorn api:api_app        then open http://127.0.0.1:8000/docs
"""
import os
import threading
from typing import List, Literal, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import guardrails
import loan_workflow as lw

workflow, saver = lw.open_workflow()       # the LangGraph workflow + its checkpoint DB, opened once
lock = threading.Lock()                    # one application at a time (shared DB and model quota)
API_KEY = os.getenv("LENDING_API_KEY")     # a password you choose; callers send it as the X-API-Key header


def check_key(x_api_key):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key header")


# --- request / response shapes (FastAPI returns 422 automatically when a request doesn't match) ---
class PolicyRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    sector: Optional[str] = None


class PolicyResponse(BaseModel):
    answer: str
    sources: List[str]


class ApplicationRequest(BaseModel):
    application: dict                       # one intake record, e.g. a row of intake_validated.json


class ApplicationStatus(BaseModel):
    application_id: str
    status: str                             # awaiting_review | done
    route: Optional[str] = None             # auto_approve | human_review | flag | decline
    risk_grade: Optional[str] = None
    dscr: Optional[float] = None
    ai_review: Optional[str] = None         # approved | escalated
    final_decision: Optional[dict] = None
    # deliberately NOT returning prompts, tool calls or raw records: internal, not part of the API


class ReviewRequest(BaseModel):
    decision: Literal["approve", "decline", "refer"]
    officer: str = Field(min_length=3, max_length=40)       # signatory id, e.g. ASH-U-007
    notes: str = ""


def status_of(application_id: str) -> ApplicationStatus:
    status, snap = lw.status_of(workflow, application_id)
    if status == "new":
        raise HTTPException(status_code=404, detail=f"No application {application_id}")
    v = snap.values
    a = v.get("assessment") or {}
    return ApplicationStatus(application_id=application_id, status=status, route=v.get("route"),
                             risk_grade=a.get("risk_grade"), dscr=a.get("dscr"),
                             ai_review=(v.get("committee") or {}).get("status"),
                             final_decision=v.get("final_decision"))


api_app = FastAPI(title="Ashva Capital Lending API", version="1.0")


@api_app.get("/health")
def health():
    return {"status": "ok"}


@api_app.post("/policy/ask", response_model=PolicyResponse)
def policy_ask(req: PolicyRequest, x_api_key: Optional[str] = Header(default=None)):
    check_key(x_api_key)
    guard = guardrails.check_input(req.question)                     # block prompt injection
    if not guard["passed"]:
        raise HTTPException(status_code=422, detail={"message": "guardrail check failed", "flags": guard["flags"]})
    result = lw._policy().policy_answer(guard["clean_text"], req.sector)
    guard = guardrails.check_output(result["answer"])                # block approval claims etc.
    if not guard["passed"]:
        raise HTTPException(status_code=422, detail={"message": "guardrail check failed", "flags": guard["flags"]})
    sources = sorted({f"{c['source']} p.{c['page']}" for c in result["citations"]})
    return PolicyResponse(answer=result["answer"], sources=sources)


@api_app.post("/applications", response_model=ApplicationStatus)
def submit_application(req: ApplicationRequest, x_api_key: Optional[str] = Header(default=None)):
    check_key(x_api_key)
    application_id = lw.thread_id_for(req.application)
    with lock:
        if lw.status_of(workflow, application_id)[0] != "new":
            raise HTTPException(status_code=409, detail=f"{application_id} was already submitted")
        workflow.invoke({"application_id": application_id, "raw": req.application,
                         "input_kind": "extracted", "events": []}, lw.config_for(application_id))
    return status_of(application_id)


@api_app.get("/applications/{application_id}", response_model=ApplicationStatus)
def get_application(application_id: str, x_api_key: Optional[str] = Header(default=None)):
    check_key(x_api_key)
    return status_of(application_id)


@api_app.get("/reviews/pending", response_model=List[ApplicationStatus])
def pending_reviews(x_api_key: Optional[str] = Header(default=None)):
    check_key(x_api_key)
    waiting = [tid for tid in lw.all_thread_ids(saver) if lw.status_of(workflow, tid)[0] == "awaiting_review"]
    return [status_of(tid) for tid in waiting]


@api_app.post("/applications/{application_id}/review", response_model=ApplicationStatus)
def review_application(application_id: str, req: ReviewRequest, x_api_key: Optional[str] = Header(default=None)):
    check_key(x_api_key)
    with lock:
        status, snap = lw.status_of(workflow, application_id)
        if status != "awaiting_review":
            raise HTTPException(status_code=409, detail=f"{application_id} is not waiting for review ({status})")
        outcome = lw.OFFICER_DECISIONS[req.decision]                  # approve -> approved, ...
        amount = snap.values["validated"].get("requested_amount_inr")
        problem = lw.signatory_problem(req.officer, outcome, amount)  # may this officer sign this amount?
        if problem:
            raise HTTPException(status_code=403, detail=problem)
        workflow.invoke(lw.Command(resume={"outcome": outcome, "officer": req.officer, "notes": req.notes}),
                        lw.config_for(application_id))
    return status_of(application_id)
