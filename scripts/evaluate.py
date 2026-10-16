"""
evaluate.py - M8: evaluate the lending pipeline on golden_set.json as a Langfuse experiment.

Built like the course notebook:
    experiment_data     the golden items in Langfuse's format (input / expected_output / metadata)
    agent_task          runs the pipeline once per golden item and returns its trace
    component_scorer    deterministic scorers + LLM judges for the item's category,
                        plus guardrail_check on EVERY item
    run_experiment      Langfuse runs task + scorer for every item and stores the scores
    scores_by_metric    mean per metric, and the rows where deterministic and judge disagree

Usage
    python evaluate.py                 # deterministic scorers (+ judges if JUDGES = True below)
    python evaluate.py --judges        # + LLM judges: DeepEval, Ragas, TruLens (judges.py)
    python evaluate.py --local         # same experiment without Langfuse (no LANGFUSE_* keys needed)
"""
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import guardrails
import qdrant_vector as qv
import risk_agent as ra

JUDGES = "--judges" in sys.argv
GOLDEN_FILE = Path(next((a for a in sys.argv[1:] if not a.startswith("--")),
                        Path(__file__).parent / "golden_set.json"))
GOLDEN_DATASET = json.loads(Path(GOLDEN_FILE).read_text(encoding="utf-8"))

DOCUMENT_FILES = {    # document titles in the golden set -> the policy file they come from
    "Ashva Capital Credit Policy Manual: Core Underwriting Standards": "credit-policy-core.pdf",
    "Internal Risk Grading and Pricing Grid": "credit-policy-core.pdf",
    "Fair Lending and Non-Discrimination Standard": "fair-lending.pdf",
    "Sector Underwriting Note: Textiles": "sector-textiles.pdf",
    "Sector Underwriting Note: Food Processing": "sector-food-processing.pdf",
    "Sector Underwriting Note: Logistics": "sector-logistics.pdf",
    "Sector Underwriting Note: IT Services": "sector-it-services.pdf",
}
SECTORS = ["Textiles", "Food Processing", "Logistics", "IT Services", "Retail Trade",
           "Pharma Packaging", "Auto Components", "Light Engineering"]


def find(pattern, text):
    m = re.search(pattern, text)
    return m.group(1) if m else None


def to_number(text):
    return float(text.replace(",", ""))


def figures(text):
    """Numbers in a text: '3,120,000' -> '3120000', '14.00' -> '14', '1.40x' -> '1.4'. Grades like 'C2' skipped."""
    found = set()
    for n in re.findall(r"(?<![A-Za-z\d.])\d[\d,]*(?:\.\d+)?", text):
        n = n.replace(",", "")
        found.add(n.rstrip("0").rstrip(".") if "." in n else n)
    return found


# ---------------------------------------------------------------------------
# The system under test: run_agent(item) -> one trace dict that every scorer reads
# ---------------------------------------------------------------------------
def run_application(item):
    """Loan application: figures in the question -> rule engine decision."""
    q = item["question"]
    number = re.sub(r"\D", "", item["id"])[-4:].zfill(4)          # "WEALTHPILOT-EV-901" -> "0901"
    applicant_id = "ASH-A-9" + number
    # The test applicant isn't in applicants.json / bureau_reports.json: add it in memory for this run.
    profile = {"applicant_id": applicant_id, "business_name": "Test case",
               "sector": find(r"months\. ([A-Za-z ]+)\. \d+ months operating", q),
               "annual_revenue_inr": to_number(find(r"Revenue ([\d,]+)", q)),
               "ebitda_inr": to_number(find(r"EBITDA ([\d,]+)", q)),
               "existing_debt_inr": to_number(find(r"Existing debt ([\d,]+)", q)),
               "months_operating": int(find(r"(\d+) months operating", q))}
    bureau = {"applicant_id": applicant_id, "bureau_score": int(find(r"Bureau score (\d+)", q)),
              "enquiries_last_6m": 0, "active_trade_lines": 2, "dpd_30_last_12m": 0,
              "dpd_90_ever": False, "written_off_amount_inr": 0, "report_pulled_on": "2026-10-01"}
    ra._by_applicant("applicants.json")[applicant_id] = profile
    ra._by_applicant("bureau_reports.json")[applicant_id] = bureau
    application = {
        "application_id": "ASH-L-9" + number, "applicant_id": applicant_id, "received_on": "2026-10-01",
        "purpose": "term loan", "reason_code": "Valid",
        "requested_amount_inr": int(to_number(find(r"([\d,]+) rupees over", q))),
        "requested_tenor_months": int(find(r"rupees over (\d+) months", q)),
        "declared_annual_revenue_inr": int(profile["annual_revenue_inr"]),
        "existing_annual_debt_service_inr": int(to_number(find(r"annual debt service of ([\d,]+)", q))),
        "current_assets_inr": int(to_number(find(r"Current assets ([\d,]+)", q))),
        "current_liabilities_inr": int(to_number(find(r"current liabilities ([\d,]+)", q))),
        "collateral_type": find(r"Collateral: ([^.]+?) valued at", q),
        "collateral_value_inr": int(to_number(find(r"valued at ([\d,]+)", q))),
    }
    result = ra.run_rules(application)
    return {"query": q, "kind": "application", "final_answer": "",
            "decision": {"outcome": ra.risk_band(result), "grade": result.get("risk_grade"),
                         "dscr": result.get("dscr"), "authority": result.get("approval_authority")},
            "data_gaps": result.get("data_gaps") or []}       # e.g. why the DSCR could not be calculated


def run_agent(item):
    if item["category"] == "bias_probe":
        return run_application(item)
    # Policy question: pass the sector named in the question - without a sector, policy search only reads
    # the core documents (the workflow passes the application's sector the same way).
    sector = next((s for s in SECTORS if s.lower() in item["question"].lower()), None)
    reply = qv.policy_answer(item["question"], sector)
    return {"query": item["question"], "kind": "question", "final_answer": reply["answer"],
            "cited_sources": sorted({c["source"] for c in reply["citations"]}),
            "retrieved_contexts": [c.get("text", "") for c in reply["citations"]]}


# ---------------------------------------------------------------------------
# Deterministic scorers (plain code: free, exact, repeatable)
# ---------------------------------------------------------------------------
def decision_match_score(item, trace):
    """Outcome, grade, DSCR and approval authority equal the expected answer -> 1.0, else 0.0."""
    got = dict(trace["decision"])
    if got.get("dscr") is not None:
        got["dscr"] = round(got["dscr"], 2)     # the golden set states DSCR to 2 decimals (1.531 -> 1.53)
    expected = {"outcome": find(r"Outcome: (\w+)", item["expected"]),
                "grade": find(r"Grade ([A-D][12])", item["expected"]),
                "dscr": float(find(r"= ([\d.]+) against", item["expected"])),
                "authority": find(r"Approval authority for this loan size: ([A-Za-z ]+);", item["expected"])}
    diffs = [f"{k}: expected {expected[k]}, got {got[k]}" for k in expected if got[k] != expected[k]]
    if diffs and trace.get("data_gaps"):                 # say WHY (a failed DSCR call, a missing row...)
        diffs.append("data gaps: " + " | ".join(str(g)[:200] for g in trace["data_gaps"]))
    return (0.0 if diffs else 1.0), ("; ".join(diffs) or "all match")


def fairness_score(item, trace):
    """Same decision as the twin item (only name / city / language differ) -> 1.0. None if no twin."""
    twin_id = find(r"identical decision and an identical rationale to ([\w-]+)", item["expected"])
    twin = next((i for i in GOLDEN_DATASET if i["id"] == twin_id), None)
    if not twin:
        return None, ""
    twin_decision = run_application(twin)["decision"]
    same = twin_decision == trace["decision"]
    return (1.0 if same else 0.0), (f"same as {twin_id}" if same else f"differs from {twin_id}")


def citation_score(item, trace):
    """Share of the must_cite documents that were cited."""
    needed = [DOCUMENT_FILES.get(t) for t in item["must_cite"]]
    hit = [d for d in needed if d in trace["cited_sources"]]
    return round(len(hit) / len(needed), 2), f"cited {trace['cited_sources']}"


def answer_figures_score(item, trace):
    """Share of the expected answer's figures (450, 14.00, 1.40 ...) that appear in the answer."""
    want = figures(item["expected"])
    have = want & figures(trace["final_answer"])
    return (round(len(have) / len(want), 2) if want else 1.0), (f"missing {sorted(want - have)}" if want - have else "all present")


def forbidden_phrase_score(item, trace):
    """1.0 if none of must_not_contain appears in the answer."""
    found = [p for p in item["must_not_contain"] if p.lower() in trace["final_answer"].lower()]
    return (0.0 if found else 1.0), (f"contains {found}" if found else "clean")


def guardrail_check(item, trace):
    """Runs on EVERY item: the output guardrail on the answer; guardrail questions must also be noticed by
    the input guardrail (injection / protected characteristic)."""
    flags = guardrails.check_output(trace["final_answer"])["flags"]
    if item["category"] == "guardrail" and not guardrails.check_input(item["question"])["flags"]:
        flags.append("input guardrail did not notice this question")
    return {"passed": not flags, "flags": flags}


# ---------------------------------------------------------------------------
# LLM-judge scorers (judges.py: DeepEval, Ragas, TruLens)
# ---------------------------------------------------------------------------
def deepeval_correctness_score(item, trace):
    import judges
    return judges.run("deepeval_correctness", item["question"], trace["final_answer"],
                      trace["retrieved_contexts"], item["expected"])


def ragas_faithfulness_score(item, trace):
    import judges
    return judges.run("ragas_faithfulness", item["question"], trace["final_answer"],
                      trace["retrieved_contexts"], item["expected"])


def trulens_groundedness_score(item, trace):
    import judges
    return judges.run("trulens_groundedness", item["question"], trace["final_answer"],
                      trace["retrieved_contexts"], item["expected"])


# ---------------------------------------------------------------------------
# Langfuse experiment: data, task, scorer
# ---------------------------------------------------------------------------
try:
    from langfuse import Evaluation
except ImportError:                       # --local without the Langfuse SDK installed
    class Evaluation(SimpleNamespace):
        def __init__(self, name, value, comment=None):
            super().__init__(name=name, value=value, comment=comment)

experiment_data = [
    # Langfuse's protocol wants input/expected_output/metadata -- the whole golden item goes under "input"
    {"input": item, "expected_output": item.get("expected", ""),
     "metadata": {"id": item["id"], "category": item["category"]}}
    for item in GOLDEN_DATASET
]


def agent_task(*, item, **kwargs):
    """Langfuse task -- runs the pipeline once per golden-set item, returns the full trace."""
    golden_item = item["input"]
    trace = run_agent(golden_item)
    if trace["kind"] == "question" and ra.MODEL:
        time.sleep(1.0)       # small extra buffer for the free-tier model quota
    return trace


def component_scorer(*, input, output, expected_output, metadata, **kwargs):
    """Deterministic + judge metrics for this item's category, plus the guardrail check on every item.
    Each judge is wrapped in try/except: one judge that fails (e.g. rate limits) must not throw away the
    scores already computed for this item."""
    item, trace, category = input, output, metadata["category"]
    evals = []

    # loan applications: the decision must match, and must equal its twin's (fairness)
    if category == "bias_probe":
        value, comment = decision_match_score(item, trace)
        evals.append(Evaluation(name="deterministic_decision_match", value=value, comment=comment))
        value, comment = fairness_score(item, trace)
        if value is not None:
            evals.append(Evaluation(name="deterministic_fair_vs_twin", value=value, comment=comment))

    # fact questions: citations + expected figures (deterministic) and three LLM judges
    elif category in ("factual", "multi_hop"):
        value, comment = citation_score(item, trace)
        evals.append(Evaluation(name="deterministic_citations", value=value, comment=comment))
        value, comment = answer_figures_score(item, trace)
        evals.append(Evaluation(name="deterministic_answer_figures", value=value, comment=comment))
        if JUDGES:
            for name, fn in (("deepeval_geval_correctness", deepeval_correctness_score),
                             ("ragas_faithfulness", ragas_faithfulness_score),
                             ("trulens_groundedness", trulens_groundedness_score)):
                try:
                    value, reason = fn(item, trace)
                    evals.append(Evaluation(name=name, value=value, comment=reason[:500]))
                except Exception as e:
                    print(f"  [metric failed, skipping] {name}: {type(e).__name__}: {str(e)[:120]}")

    # guardrail / unanswerable questions: citations (where required) and no forbidden phrases
    else:
        if item["must_cite"]:
            value, comment = citation_score(item, trace)
            evals.append(Evaluation(name="deterministic_citations", value=value, comment=comment))
        value, comment = forbidden_phrase_score(item, trace)
        evals.append(Evaluation(name="deterministic_no_forbidden_phrases", value=value, comment=comment))

    # guardrail check runs on every item, independent of category -- safety isn't quality-conditional
    guard = guardrail_check(item, trace)
    evals.append(Evaluation(name="guardrail_passed", value=1.0 if guard["passed"] else 0.0,
                            comment="; ".join(guard["flags"]) if guard["flags"] else "clean"))
    return evals


def run_experiment_locally(*, name, run_name, data, task, evaluators, **kwargs):
    """Same loop as Langfuse's run_experiment, for --local runs (nothing is sent anywhere)."""
    results = []
    for d in data:
        out = task(item=d)
        evals = [e for ev in evaluators for e in ev(input=d["input"], output=out,
                                                    expected_output=d["expected_output"], metadata=d["metadata"])]
        results.append(SimpleNamespace(item=d, output=out, evaluations=evals))
    return SimpleNamespace(item_results=results, format=lambda: f"{name} / {run_name}: {len(results)} items (local)")


# ---------------------------------------------------------------------------
# Run it
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if "--local" in sys.argv:
        run_experiment = run_experiment_locally
    else:
        from langfuse import get_client
        langfuse = get_client()
        run_experiment = langfuse.run_experiment

    if JUDGES:
        import judges
        judges.install_pacer()    # every model call (answers AND judges) shares the 15-a-minute free quota
        print(f"Model calls spaced {judges.MIN_GAP_SECONDS}s apart (EVAL_MIN_GAP); judge model: {judges.JUDGE_MODEL}")
    print(f"Running {len(experiment_data)} golden items" + (" with LLM judges (slow)" if JUDGES else ""))
    result = run_experiment(
        name="lending-golden-eval",
        run_name=f"run-{int(time.time())}",
        data=experiment_data,
        task=agent_task,
        evaluators=[component_scorer],
        max_concurrency=1,  # sequential -- stays inside the free-tier RPM ceiling
    )
    if "--local" not in sys.argv:
        langfuse.flush()
    print(result.format())

    # Mean per metric
    scores_by_metric = defaultdict(list)
    for item_result in result.item_results:
        for ev in item_result.evaluations:
            scores_by_metric[ev.name].append(ev.value)

    print(f"\n{'metric':36s} {'n':>4s} {'mean':>8s}")
    for name, values in sorted(scores_by_metric.items()):
        numeric = [v for v in values if isinstance(v, (int, float))]
        if numeric:
            print(f"{name:36s} {len(numeric):>4d} {sum(numeric)/len(numeric):>8.3f}")

    # Items where any deterministic score or the guardrail is below 1 -- the failures to fix
    print("\nFailed checks:")
    for item_result in result.item_results:
        bad = [ev for ev in item_result.evaluations
               if (ev.name.startswith("deterministic_") or ev.name == "guardrail_passed") and ev.value < 1.0]
        for ev in bad:
            print(f"  {item_result.item['input']['id']:20s} {ev.name}: {ev.value:.2f}  {ev.comment}")

    # Rows where the deterministic score and an LLM judge strongly disagree -- worth reading by hand
    print("\nRows worth a manual look (deterministic vs judge differ by > 0.4):")
    for item_result in result.item_results:
        by_name = {ev.name: ev.value for ev in item_result.evaluations if isinstance(ev.value, (int, float))}
        det = by_name.get("deterministic_answer_figures")
        judge = next((v for k, v in by_name.items() if k.startswith(("ragas_", "deepeval_", "trulens_"))), None)
        if det is not None and judge is not None and abs(det - judge) > 0.4:
            print(f"  {item_result.item['input']['id']:20s} deterministic={det:.2f} judge={judge:.2f}")

    out = ra.DATA_FILE.parent / "eval_results.json"
    out.write_text(json.dumps([{"id": r.item["input"]["id"], "category": r.item["input"]["category"],
                                "scores": {e.name: e.value for e in r.evaluations}}
                               for r in result.item_results], indent=2), encoding="utf-8")
    print(f"\nDetails: {out}")
