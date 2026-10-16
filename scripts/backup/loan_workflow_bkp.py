"""
Loan application workflow (LangGraph) with checkpointing and human review.

    intake -> verify --(valid)--> risk_score --> route --(auto_approve)--> system_approve ------------------------> record -> END
                    \\--(invalid)------------------^      |--(decline)-------> draft_decline --> prepare_review -> human_review -> record
                                                          |--(borderline)----> flag ----------> prepare_review -> human_review -> record
                                                          \\--(approve)-----------------------> prepare_review -> human_review -> record

  intake          intake_validator.extract_fields()  (LLM extraction of the 16 intake fields)
  verify          intake_validator.validate()        (Valid / Invalid + reason)
  risk_score      risk_agent.run_risk_agent()        (or run_rules(): no LLM) -> grade, decision, reason codes
  route           plain code on risk_agent's decision (rules in risk_agent.py's docstring):
                    auto_approve -> the system approves alone (approve, <= 10 lakh, grade C1 or better)
                    decline      -> clear no: notice drafted, then signed off by the approval authority
                    borderline   -> flagged for review
                    approve      -> needs sign-off by the approval authority
  draft_decline   adverse-action notice drafted from the policy PDFs (Qdrant), ready for sign-off
  prepare_review  officer's pack: applicant history (applicant_memory) + the draft notice for declines
  human_review    interrupt(): waits for the signatory (approve / decline / refer); the signatory must hold
                  the application's approval_authority (Credit Officer / Credit Committee / Board)
  record          decision log + Supermemory write-back, so the next application sees this decision

Every application is one LangGraph thread (thread_id = application_id), checkpointed in SQLite after every
node: a paused application waits for review across restarts, and a run that crashed mid-way resumes from
its last completed step.

pip install langgraph langgraph-checkpoint-sqlite

Usage:
    python loan_workflow.py run [--input FILE] [--limit N]     # .jsonl raw intake, or intake_validated.json
    python loan_workflow.py pending                            # waiting for an officer (flagged first)
    python loan_workflow.py ask ASH-L-00077 "has this applicant been declined before?"     # Supermemory
    python loan_workflow.py ask ASH-L-00077 --policy "acceptable collateral for this sector" # Qdrant
    python loan_workflow.py ask ASH-L-00077 --notice           # draft adverse-action notice for this case
    python loan_workflow.py review ASH-L-00077 --decision approve|decline|refer --officer ASH-U-007 [--notes ...]
    python loan_workflow.py show ASH-L-00077
    python loan_workflow.py graph
"""
import argparse
import hashlib
import json
import operator
import os
import sqlite3
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import intake_validator as iv
import risk_agent as ra

CHECKPOINT_DB = Path(os.getenv("WORKFLOW_CHECKPOINT_DB", "loan_workflow_checkpoints.db"))
OUTCOMES_FILE = ra.DATA_FILE.with_name("workflow_decisions.jsonl")
RISK_MODE = os.getenv("WORKFLOW_RISK_MODE", "agent")   # "agent" (LLM + tools) or "rules" (no LLM)

OFFICER_DECISIONS = {"approve": "approved", "decline": "declined", "refer": "referred_back"}
# For evaluation only: route vs a ground_truth risk_band, when the raw intake carries one.
ROUTE_TO_BAND = {"decline": "decline", "flag": "borderline", "human_review": "approve", "auto_approve": "approve"}


# ---------------------------------------------------------------------------
# Optional services: a missing Qdrant / Supermemory setup degrades the review pack, never the run
# ---------------------------------------------------------------------------
def _policy():
    import qdrant_vector
    return qdrant_vector


def _memory():
    import applicant_memory
    return applicant_memory


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _event(step: str, **detail) -> list[dict]:
    return [{"at": _now(), "step": step, **{k: v for k, v in detail.items() if v not in (None, [], "", {})}}]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class LoanState(TypedDict, total=False):
    application_id: str
    raw: dict                       # the application as received
    input_kind: str                 # "raw" (needs LLM extraction) or "extracted" (already the 16 fields)
    extracted: dict                 # 16 intake fields
    validated: dict                 # intake_validator row: 16 fields + reason_code + reason_description
    assessment: dict                # risk_agent output: grade, DSCR, decision, decision_reasons, ...
    risk_record: dict               # risk_agent.to_output_record(): application layout + risk_review block
    route: str                      # "auto_approve" | "decline" | "flag" | "human_review"
    route_reasons: list[str]
    review_pack: dict               # history + draft notice shown to the officer
    notice: dict                    # adverse-action notice (declines)
    final_decision: dict            # {outcome, by, at, notes}
    events: Annotated[list[dict], operator.add]   # audit trail: each node appends, nothing is overwritten


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def intake(state: LoanState) -> dict:
    raw = state["raw"]
    if state.get("input_kind") == "extracted":
        extracted = {k: raw.get(k) for k in iv.INTAKE_FIELDS}
        return {"extracted": extracted, "events": _event("intake", mode="already extracted")}
    # Raw application: the LLM reads it (without ground_truth, so it can't be copied).
    source = {k: v for k, v in raw.items() if k != "ground_truth"}
    try:
        extracted = iv.extract_fields(json.dumps(source, ensure_ascii=False, default=str))
        return {"extracted": extracted, "events": _event("intake", mode="LLM extraction")}
    except Exception as e:
        return {"extracted": {}, "events": _event("intake", error=f"Extraction failed: {e}")}


def verify(state: LoanState) -> dict:
    extracted = state.get("extracted") or {}
    row = iv.validate(extracted) if extracted else iv.invalid_row("Extraction failed; nothing to validate.")
    return {"validated": row,
            "events": _event("verify", result=row["reason_code"], reason=row["reason_description"])}


def after_verify(state: LoanState) -> str:
    # Failed verification is still scored when the application can be identified (a decline outranks a
    # missing figure; otherwise the missing figure makes it borderline). Unidentifiable -> straight to a person.
    v = state["validated"]
    return "risk_score" if v["reason_code"] == "Valid" or (v.get("application_id") and v.get("applicant_id")) else "route"


def risk_score(state: LoanState) -> dict:
    record = state["validated"]
    try:
        assessment = ra.run_risk_agent(record) if RISK_MODE == "agent" else ra.run_rules(record)
    except Exception as e:  # model or MCP server down: a person takes it, nothing crashes
        assessment = {"application_id": state["application_id"], "decision": "manual_review",
                      "error": str(ra._unwrap_exception_group(e)), "decision_reasons": []}
    return {
        "assessment": assessment,
        "risk_record": ra.to_output_record(record, assessment),
        "events": _event("risk_score", decision=assessment.get("decision"), risk_grade=assessment.get("risk_grade"),
                         dscr=assessment.get("dscr"), error=assessment.get("error")),
    }


def _route(route: str, reasons: list[str]) -> dict:
    return {"route": route, "route_reasons": reasons, "events": _event("route", route=route, reasons=reasons)}


def route(state: LoanState) -> dict:
    """The one routing decision. Plain code on risk_agent's output; no model call, so it's unit-testable."""
    v = state["validated"]
    a = state.get("assessment") or {}
    if not a:   # not scored: could not even be identified
        return _route("flag", [f"Failed document verification: {v['reason_description']}"])
    if a.get("error"):
        return _route("flag", [f"Risk scoring failed: {a['error']}"])

    reasons = [f"{r['code']}: {r['detail']}" for r in a.get("decision_reasons", [])] + a.get("routing_reasons", [])
    decision = a.get("decision")
    if decision == "auto_approve":   # approve, <= 10 lakh, grade C1 or better: the system signs alone
        return _route("auto_approve", reasons)
    if decision == "decline":        # DSCR < 1.0, bureau < 450, < 24 months, grade D2
        return _route("decline", reasons)
    if decision == "approve":        # meets every standard, but a signatory must approve
        return _route("human_review", reasons)
    return _route("flag", reasons)   # borderline (and anything unexpected): a person decides


def next_after_route(state: LoanState) -> str:
    """The edge only READS the route written by route()."""
    return state["route"]


def _draft_notice(assessment: dict) -> dict:
    try:
        return _policy().adverse_action_notice(assessment)
    except Exception as e:  # Qdrant / PDFs not available: say so, the decision still stands
        reasons = [f"{r['code']}: {r['detail']}" for r in assessment.get("decision_reasons", [])]
        return {"notice": "DRAFT - policy citations unavailable\n\n" + "\n".join(reasons),
                "error": str(e), "citations": []}


def system_approve(state: LoanState) -> dict:
    """The only path that finishes without a person: approve, <= 10 lakh, grade C1 or better."""
    return {"final_decision": {"outcome": "approved", "by": "system", "at": _now(), "notes": ""},
            "events": _event("system_approve")}


def draft_decline(state: LoanState) -> dict:
    """Clear no: the notice is drafted now; the decline still needs the approval authority's sign-off."""
    notice = _draft_notice(state.get("assessment") or {})
    return {"notice": notice,
            "events": _event("draft_decline", notice_citations=len(notice.get("citations", [])), error=notice.get("error"))}


def flag(state: LoanState) -> dict:
    return {"events": _event("flag", priority="high")}


def prepare_review(state: LoanState) -> dict:
    """Everything the officer needs, gathered BEFORE the pause (so it isn't re-fetched on resume)."""
    a = state.get("assessment") or {}
    applicant_id = state["validated"].get("applicant_id") or (state.get("raw") or {}).get("applicant_id")
    pack: dict = {}
    try:
        pack["history"] = _memory().history_summary(applicant_id) if applicant_id else None
    except Exception as e:
        pack["history"] = {"error": str(e)}
    # Declines arrive with the notice already drafted; for other cases the officer can ask for one (ask --notice).
    if state.get("notice"):
        pack["draft_notice"] = state["notice"]
    return {"review_pack": pack, "events": _event("prepare_review", history=bool(pack.get("history")),
                                                  draft_notice="draft_notice" in pack)}


def human_review(state: LoanState) -> dict:
    """Pauses the graph. On resume LangGraph re-runs this node and interrupt() returns the officer's answer,
    so nothing with side effects goes before interrupt()."""
    a = state.get("assessment") or {}
    rr = (state.get("risk_record") or {}).get(ra.OUTCOME_BLOCK, {})
    pack = state.get("review_pack") or {}
    answer = interrupt({
        "application_id": state["application_id"],
        "applicant_id": state["validated"].get("applicant_id"),
        "business_name": (state.get("risk_record") or {}).get("business_name"),
        "sector": a.get("sector"),
        "requested_amount_inr": state["validated"].get("requested_amount_inr"),
        "flagged": state["route"] == "flag",
        "recommendation": {"decline": "decline (confirm)", "flag": "borderline (decide)",
                           "human_review": "approve (sign off)"}.get(state["route"], state["route"]),
        "priority": "high" if state["route"] == "flag" else "normal",
        "risk_grade": a.get("risk_grade"),
        "dscr": a.get("dscr"),
        "risk_band": rr.get("risk_band"),
        "approval_authority": rr.get("approval_authority"),
        "indicative_rate_pct": a.get("indicative_rate_pct"),
        "reasons": state.get("route_reasons", []),
        "history": pack.get("history"),
        "draft_notice_ready": "draft_notice" in pack,
        "rationale": a.get("rationale"),
    })
    out = {
        "final_decision": {"outcome": answer["outcome"], "by": answer["officer"], "at": _now(),
                           "notes": answer.get("notes", "")},
        "events": _event("human_review", officer=answer["officer"], outcome=answer["outcome"]),
    }
    if answer["outcome"] == "declined":   # runs once, after the officer has answered
        out["notice"] = pack.get("draft_notice") or _draft_notice(a)
    return out


def _primary_reason(a: dict, outcome: str) -> str | None:
    codes = [r["code"] for r in a.get("decision_reasons", [])]
    if outcome == "approved":
        return "APPROVED_STANDARD" if "APPROVED_STANDARD" in codes or not codes else codes[0]
    return next((c for c in codes if c != "APPROVED_STANDARD"), codes[0] if codes else None)


def record(state: LoanState) -> dict:
    """Decision log + Supermemory write-back, in the past_decisions shape."""
    a = state.get("assessment") or {}
    v = state["validated"]
    fd = state["final_decision"]
    decision = {
        "decision_id": state["application_id"],
        "applicant_id": v.get("applicant_id") or (state.get("raw") or {}).get("applicant_id"),
        "decided_on": date.today().isoformat(),
        "outcome": fd["outcome"],
        "internal_grade": a.get("risk_grade"),
        "approved_amount_inr": v.get("requested_amount_inr") if fd["outcome"] == "approved" else 0,
        "reason_code": _primary_reason(a, fd["outcome"]),
        "human_signatory": None if fd["by"] == "system" else fd["by"],
        "notes": fd.get("notes") or None,
    }
    memory = {"stored": False, "error": "no applicant_id"}
    if decision["applicant_id"]:
        try:
            memory = _memory().record_final_decision(decision)
        except Exception as e:
            memory = {"stored": False, "error": str(e)}

    row = {**decision, "route": state.get("route"), "decided_by": fd["by"], "at": fd["at"],
           "risk_record": state.get("risk_record"),
           "notice": (state.get("notice") or {}).get("notice"), "memory": memory}
    OUTCOMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTCOMES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
    return {"events": _event("record", outcome=fd["outcome"], memory_stored=memory.get("stored"),
                             memory_error=memory.get("error"))}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def build_graph(checkpointer):
    g = StateGraph(LoanState)
    for name, fn in [("intake", intake), ("verify", verify), ("risk_score", risk_score), ("route", route),
                     ("system_approve", system_approve), ("draft_decline", draft_decline), ("flag", flag),
                     ("prepare_review", prepare_review), ("human_review", human_review), ("record", record)]:
        g.add_node(name, fn)

    g.add_edge(START, "intake")
    g.add_edge("intake", "verify")
    g.add_conditional_edges("verify", after_verify, {"risk_score": "risk_score", "route": "route"})
    g.add_edge("risk_score", "route")
    # Every value next_after_route() can return must be a key here, or LangGraph fails at run time.
    g.add_conditional_edges("route", next_after_route,
                            {"auto_approve": "system_approve", "decline": "draft_decline",
                             "flag": "flag", "human_review": "prepare_review"})
    g.add_edge("draft_decline", "prepare_review")
    g.add_edge("flag", "prepare_review")
    g.add_edge("prepare_review", "human_review")
    g.add_edge("system_approve", "record")
    g.add_edge("human_review", "record")
    g.add_edge("record", END)
    return g.compile(checkpointer=checkpointer)


def open_workflow():
    # check_same_thread=False: LangGraph may run nodes on worker threads.
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    saver = SqliteSaver(conn)
    return build_graph(saver), saver


def config_for(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def thread_id_for(record: dict) -> str:
    return (record.get("application_id")
            or "line-" + hashlib.sha1(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()[:12])


def status_of(app, thread_id: str) -> tuple[str, object]:
    """new | awaiting_review | interrupted (crashed mid-run, resumable) | done"""
    snap = app.get_state(config_for(thread_id))
    if not snap.values:
        return "new", snap
    if snap.next:
        return ("awaiting_review" if "human_review" in snap.next else "interrupted"), snap
    return "done", snap


def review_payload(snap) -> dict:
    for task in snap.tasks:
        for intr in getattr(task, "interrupts", ()) or ():
            return intr.value
    return {}


def all_thread_ids(saver) -> list[str]:
    seen = []
    for cp in saver.list(None):
        tid = cp.config["configurable"]["thread_id"]
        if tid not in seen:
            seen.append(tid)
    return seen


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_input(path: Path, limit: int | None) -> list[tuple[dict, str]]:
    """.jsonl -> raw applications (LLM extraction); .json array -> already-extracted intake records."""
    items: list[tuple[dict, str]] = []
    if path.suffix == ".json":
        items = [(r, "extracted") for r in json.loads(path.read_text(encoding="utf-8"))]
    else:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                items.append((json.loads(line), "raw"))
            except json.JSONDecodeError as e:
                print(f"Line {n}: not valid JSON, skipped ({e})")
    return items[:limit] if limit else items


def print_result(app, tid: str) -> str | None:
    status, snap = status_of(app, tid)
    v = snap.values
    a = v.get("assessment") or {}
    print(f"   intake: {v.get('validated', {}).get('reason_code')}   grade {a.get('risk_grade')}  "
          f"DSCR {a.get('dscr')}  ->  route: {v.get('route')}")
    for r in v.get("route_reasons", [])[:4]:
        print(f"     - {r}")
    if status == "awaiting_review":
        rr = (v.get("risk_record") or {}).get(ra.OUTCOME_BLOCK, {})
        print(f"   ⏸  waiting for sign-off by {rr.get('approval_authority') or 'an officer'}"
              + (" (FLAGGED)" if v.get("route") == "flag" else "")
              + (" - decline notice drafted" if v.get("notice") else ""))
    elif status == "done":
        fd = v["final_decision"]
        print(f"   ✅ {fd['outcome']} by {fd['by']}" + ("  (adverse-action notice drafted)" if v.get("notice") else ""))
    return v.get("route")


def cmd_run(app, saver, args) -> None:
    items = load_input(Path(args.input), args.limit)
    routes, agree, total = Counter(), 0, 0
    for raw, kind in items:
        tid = thread_id_for(raw)
        status, _ = status_of(app, tid)
        print(f"\n=== {tid} ===")
        if status in ("done", "awaiting_review"):
            print(f"   already {status.replace('_', ' ')}; skipped")
            continue
        if status == "interrupted":
            print("   resuming from the last checkpoint")
            app.invoke(None, config_for(tid))
        else:
            app.invoke({"application_id": tid, "raw": raw, "input_kind": kind, "events": []}, config_for(tid))
        r = print_result(app, tid)
        if r:
            routes[r] += 1
            gt = (raw.get("ground_truth") or {}).get("risk_band")
            if gt:
                total += 1
                agree += ROUTE_TO_BAND[r] == gt
    print(f"\nRoutes this run: {dict(routes)}")
    if total:
        print(f"Route vs ground-truth risk_band (evaluation only): {agree}/{total}")
    print(f"Checkpoints: {CHECKPOINT_DB}   Decisions: {OUTCOMES_FILE}")


def cmd_pending(app, saver, args) -> None:
    waiting = [(tid, snap) for tid in all_thread_ids(saver)
               for st, snap in [status_of(app, tid)] if st == "awaiting_review"]
    if not waiting:
        print("No applications waiting for review.")
        return
    rank = {"high": 0, "normal": 1, "low": 2}
    waiting.sort(key=lambda x: rank.get(review_payload(x[1]).get("priority"), 1))
    for tid, snap in waiting:
        p = review_payload(snap)
        amount = p.get("requested_amount_inr")
        print(f"\n[{p.get('priority', 'normal').upper()}] {tid} | {p.get('recommendation')} | {p.get('business_name')} | {p.get('sector')}"
              + (f" | INR {amount:,}" if amount else ""))
        print(f"   grade {p.get('risk_grade') or '-'}  DSCR {p.get('dscr') if p.get('dscr') is not None else '-'}"
              f"  band {p.get('risk_band') or '-'}  signs: {p.get('approval_authority') or '-'}"
              + (f"  rate {p['indicative_rate_pct']}%" if p.get("indicative_rate_pct") else ""))
        for r in p.get("reasons", [])[:5]:
            print(f"   - {r}")
        h = p.get("history") or {}
        if h.get("previous_decisions"):
            print(f"   history: {h['previous_decisions']} previous decision(s): " + "; ".join(h["outcomes"][-3:]))
        if p.get("draft_notice_ready"):
            print(f"   draft adverse-action notice ready ->  python loan_workflow.py ask {tid} --notice")
    print(f"\n{len(waiting)} waiting.")


def cmd_ask(app, saver, args) -> None:
    """Read-only: answers questions about a case; never changes its saved state."""
    status, snap = status_of(app, args.application_id)
    if status == "new":
        print(f"No workflow state for {args.application_id}.")
        return
    v = snap.values
    a = v.get("assessment") or {}
    applicant_id = (v.get("validated") or {}).get("applicant_id")

    if args.notice:
        notice = (v.get("notice") or (v.get("review_pack") or {}).get("draft_notice") or _draft_notice(a))
        print(notice["notice"])
        for c in notice.get("citations", []):
            print(f"  [{c['id']}] {c['source']} p.{c['page']}")
        return
    if not args.question:
        print("Give a question, or use --notice.")
        return
    if args.policy:
        result = _policy().policy_answer(args.question, a.get("sector"))
        print(result["answer"])
        for c in result["citations"]:
            print(f"  [{c['id']}] {c['source']} p.{c['page']}")
    else:
        result = _memory().ask_history(applicant_id, args.question)
        print(result["answer"])
        print(f"\n(source: {result['source']})")


AUTHORITY_RANK = {name: i for i, (_, name) in enumerate(ra.APPROVAL_AUTHORITY_TIERS)}


def signatory_problem(officer_id: str, outcome: str, amount: int | None) -> str | None:
    """signatories.json: the officer must exist and be active, and must hold the application's
    approval_authority (Credit Officer <= 10 lakh, Credit Committee <= 50 lakh, Board above) to approve or
    decline. Any active signatory may refer a case back."""
    try:
        signer = _memory().signatories().get(officer_id)
    except Exception as e:
        return f"Cannot check signatory authority: {e}"
    if signer is None:
        return f"{officer_id} is not in signatories.json."
    if signer.get("active") is False:
        return f"{officer_id} ({signer.get('name')}) is not an active signatory."
    required = ra.approval_authority(amount)
    has = AUTHORITY_RANK.get(signer.get("authority"), -1)
    limit = signer.get("approval_limit_inr")   # null = no limit
    if outcome in ("approved", "declined") and required and (has < AUTHORITY_RANK[required]
                                                             or (limit is not None and amount > limit)):
        return (f"INR {amount:,} needs {required} sign-off; {officer_id} ({signer.get('name')}) is "
                f"{signer.get('authority')}" + (f" with a limit of INR {limit:,}." if limit is not None else "."))
    return None


def cmd_review(app, saver, args) -> None:
    status, snap = status_of(app, args.application_id)
    if status != "awaiting_review":
        print(f"{args.application_id} is not waiting for review (status: {status}).")
        return
    amount = (snap.values.get("validated") or {}).get("requested_amount_inr")
    problem = signatory_problem(args.officer, OFFICER_DECISIONS[args.decision], amount)
    if problem:   # checked BEFORE resuming, so the case stays paused and nothing is recorded
        print(f"Not recorded: {problem}")
        return
    app.invoke(Command(resume={"outcome": OFFICER_DECISIONS[args.decision], "officer": args.officer,
                               "notes": args.notes or ""}),
               config_for(args.application_id))
    _, snap = status_of(app, args.application_id)
    fd = snap.values["final_decision"]
    print(f"{args.application_id}: {fd['outcome']} by {fd['by']}. Logged to {OUTCOMES_FILE} and applicant memory.")
    if snap.values.get("notice"):
        print(f"Adverse-action notice drafted ->  python loan_workflow.py ask {args.application_id} --notice")


def cmd_show(app, saver, args) -> None:
    status, snap = status_of(app, args.application_id)
    if status == "new":
        print(f"No workflow state for {args.application_id}.")
        return
    v = snap.values
    print(json.dumps({
        "status": status,
        "next": list(snap.next),
        "validated": {k: (v.get("validated") or {}).get(k) for k in ("reason_code", "reason_description")},
        "route": v.get("route"),
        "route_reasons": v.get("route_reasons"),
        "risk_record": v.get("risk_record"),
        "review_pack": {k: x for k, x in (v.get("review_pack") or {}).items() if k != "draft_notice"},
        "final_decision": v.get("final_decision"),
        "notice": (v.get("notice") or {}).get("notice"),
        "audit_trail": v.get("events"),
    }, indent=2, default=str, ensure_ascii=False))


def cmd_graph(app, saver, args) -> None:
    print(app.get_graph().draw_mermaid())


def main() -> None:
    parser = argparse.ArgumentParser(description="Loan application workflow with checkpointing")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="process applications")
    p.add_argument("--input", default=str(iv.INTAKE_FILE),
                   help=".jsonl raw intake (LLM extraction) or intake_validated.json (already extracted)")
    p.add_argument("--limit", type=int)

    sub.add_parser("pending", help="applications waiting for officer review")

    p = sub.add_parser("ask", help="read-only questions about a case")
    p.add_argument("application_id")
    p.add_argument("question", nargs="?")
    p.add_argument("--policy", action="store_true", help="ask the credit policy (Qdrant) instead of history")
    p.add_argument("--notice", action="store_true", help="show the draft adverse-action notice")

    p = sub.add_parser("review", help="record the signatory's decision and resume the workflow")
    p.add_argument("application_id")
    p.add_argument("--decision", required=True, choices=sorted(OFFICER_DECISIONS))
    p.add_argument("--officer", required=True, help="signatory id, e.g. ASH-U-007")
    p.add_argument("--notes")

    p = sub.add_parser("show", help="state and audit trail for one application")
    p.add_argument("application_id")

    sub.add_parser("graph", help="print the workflow as a Mermaid diagram")

    args = parser.parse_args()
    app, saver = open_workflow()
    {"run": cmd_run, "pending": cmd_pending, "ask": cmd_ask, "review": cmd_review,
     "show": cmd_show, "graph": cmd_graph}[args.command](app, saver, args)


if __name__ == "__main__":
    main()
