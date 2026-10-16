"""lending_mcp_server.py - Ashva Capital's lending lookups (the "mock APIs") as one read-only MCP server.

Tools (the MODEL decides to call these):
  get_credit_bureau(applicant_id)         bureau_reports.json      -> risk_agent.tool_get_credit_bureau
  get_bank_statement(applicant_id)        bank_statements.json     -> risk_agent.tool_get_bank_statement
  get_sector_policy(sector)               sector_policies.json     -> risk_agent.tool_get_sector_policy
  search_credit_policy(question, sector)  policy PDFs in Qdrant    -> qdrant_vector.policy_search

Resources (the APPLICATION decides to load these; no model involved):
  policy://limits         policy_limits.json   bank-wide minimums
  policy://reason-codes   reason_codes.json    the codes a decision may cite
  policy://risk-grades    risk_grades.json     grade table (DSCR / bureau minimums, rate, tenor, collateral)

Design rules (Lab B):
  * Read-only. There is no write, delete or "run anything" tool: a capability that is never exposed can't
    be misused, whatever a prompt says.
  * The same functions risk_agent uses, so the numbers can never drift between the rule engine and MCP.
  * Inputs are validated (applicant id format, k capped) and errors are RAISED: MCP turns them into a tool
    error the client can read instead of a silent empty result.
  * No personal fields: none of these sources returns the proprietor's name, city or language, and there
    is no applicant-profile tool. Fair lending is enforced here, in the server, for every client.
  * stdout is the JSON-RPC channel, so everything that prints is redirected to stderr.

Clients: underwriting_committee.py (one session per application), the notebook's MultiServerMCPClient,
Claude Desktop, or plain Python via call_lending_tool() at the bottom.

Run standalone:  python lending_mcp_server.py   (one line on stderr, then silence = healthy)
"""
import contextlib
import json
import re
import sys
from pathlib import Path

from fastmcp import FastMCP

SCRIPT_DIR = Path(__file__).resolve().parent          # not os.getcwd(): we run as a subprocess
sys.path.insert(0, str(SCRIPT_DIR))

with contextlib.redirect_stdout(sys.stderr):            # risk_agent prints ".env not found" etc. on import
    import risk_agent as ra

APPLICANT_ID = re.compile(r"^ASH-A-\d{5}$")
MAX_K = 8
MAX_QUESTION_CHARS = 500

mcp = FastMCP("ashva-lending")


def _check_applicant(applicant_id: str) -> str:
    applicant_id = (applicant_id or "").strip()
    if not APPLICANT_ID.match(applicant_id):
        raise ValueError(f"applicant_id must look like ASH-A-00123, got {applicant_id!r}")
    return applicant_id


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool()
def get_credit_bureau(applicant_id: str) -> dict:
    """Credit-bureau report for an applicant.

    Args:
        applicant_id: the applicant's ID, e.g. "ASH-A-00147".

    Returns:
        bureau_score, enquiries_last_6m, active_trade_lines, dpd_30_last_12m (30+ days-past-due count in
        the last 12 months), dpd_90_ever, written_off_amount_inr and report_pulled_on.
        Raises if no report is on file.
    """
    with contextlib.redirect_stdout(sys.stderr):
        return ra.tool_get_credit_bureau(_check_applicant(applicant_id))


@mcp.tool()
def get_bank_statement(applicant_id: str) -> dict:
    """Analyse an applicant's monthly bank statements.

    Args:
        applicant_id: the applicant's ID, e.g. "ASH-A-00147".

    Returns:
        months_covered, period, total/average inflow and outflow, net cash flow, closing and average daily
        balances, bounced instruments, months with a negative closing balance, and red_flags (plain-English
        warnings). months_covered is 0 when nothing is on file.
    """
    with contextlib.redirect_stdout(sys.stderr):
        return ra.tool_get_bank_statement(_check_applicant(applicant_id))


@mcp.tool()
def get_sector_policy(sector: str) -> dict:
    """Ashva Capital's lending rules for one sector.

    Args:
        sector: sector name as in sector_policies.json, e.g. "Textiles", "Light Engineering".

    Returns:
        dscr_floor, max_tenor_months, max_exposure_pct, current_exposure_pct, at_sector_cap,
        working_capital_cycle_days and seasonal_peak. Raises with the known sectors if not recognised.
    """
    with contextlib.redirect_stdout(sys.stderr):
        return ra.tool_get_sector_policy(sector)


@mcp.tool()
def search_credit_policy(question: str, sector: str | None = None, k: int = 4) -> dict:
    """Search Ashva Capital's credit policy and regulatory documents.

    Args:
        question: what to look up, e.g. "collateral cover required for grade C1".
        sector: optional sector, to include that sector's policy note in the search.
        k: number of clauses to return (1-8).

    Returns:
        {"clauses": [...]}: the most relevant clauses, each with id, source (PDF file name), page and text.
    """
    question = (question or "").strip()[:MAX_QUESTION_CHARS]
    if not question:
        raise ValueError("question must not be empty")
    k = max(1, min(int(k), MAX_K))
    with contextlib.redirect_stdout(sys.stderr):     # the first call may build the Qdrant index and print
        import qdrant_vector as qv                    # lazy: bureau/bank tools work even without Qdrant
        return {"clauses": qv.policy_search(question, sector=sector, k=k)}


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------
@mcp.resource("policy://limits")
def policy_limits() -> str:
    """Bank-wide minimums (DSCR, bureau score, months operating, leverage, liquidity, approval limits)."""
    with contextlib.redirect_stdout(sys.stderr):
        return json.dumps(ra.policy_limits(), indent=2)


@mcp.resource("policy://reason-codes")
def reason_codes() -> str:
    """Every reason code a credit decision may cite, with its meaning and whether it is a decline reason."""
    with contextlib.redirect_stdout(sys.stderr):
        return json.dumps(ra.reason_catalogue(), indent=2)


@mcp.resource("policy://risk-grades")
def risk_grades() -> str:
    """Risk-grade table: minimum DSCR and bureau score, rate, maximum tenor and collateral cover per grade."""
    with contextlib.redirect_stdout(sys.stderr):
        return json.dumps(ra.risk_grades(), indent=2)


# ---------------------------------------------------------------------------
# Synchronous client helper, for plain-Python callers (one server process per call)
# ---------------------------------------------------------------------------
def call_lending_tool(tool_name: str, arguments: dict):
    """Call one tool on this server over MCP from synchronous code. Returns the parsed result."""
    text = ra.call_mcp_tool(ra.PYTHON_BIN, [str(Path(__file__).resolve())], tool_name, arguments)
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text


if __name__ == "__main__":
    print("[lending_mcp_server] starting on stdio", file=sys.stderr)
    mcp.run(transport="stdio", show_banner=False)
