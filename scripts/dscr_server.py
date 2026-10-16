from pathlib import Path

DSCR_SERVER_CODE = '''
"""A minimal MCP server exposing calculate_dscr."""
from fastmcp import FastMCP
# Create your own MCP server
mcp = FastMCP("My DSCR Calculator")

@mcp.tool()
async def calculate_dscr(
    ebitda_inr: float,
    requested_amount_inr: float,
    requested_tenor_months: int,
    existing_annual_debt_service_inr: float,
    assessment_rate_pct: float
) -> dict:
    """Calculate DSCR for a loan application.
    Args:
        ebitda_inr: Annual EBITDA in INR.
        requested_amount_inr: Requested loan amount in INR.
        requested_tenor_months: Requested loan tenor in months.
        existing_annual_debt_service_inr:
            Existing annual debt repayments in INR.
        assessment_rate_pct:
            assessment_rate_pct in INR,
            assessed at the applicable interest rate.
    """

    if ebitda_inr < 0:
        raise ValueError("EBITDA cannot be negative.")

    if existing_annual_debt_service_inr < 0:
        raise ValueError("Existing debt service cannot be negative.")

    monthly_interest_rate=assessment_rate_pct/12/100


    emi = (
            requested_amount_inr * monthly_interest_rate
            / (
                1 - (1 + monthly_interest_rate)
                    ** (-requested_tenor_months)
                 )
            )
    total_debt_service=existing_annual_debt_service_inr + 12 * emi
    
    dscr = ebitda_inr / total_debt_service

    return {
        "dscr": round(dscr, 3),
        "total_annual_debt_service_inr": total_debt_service,
        "coverage_status": (
            "adequate" if dscr >= 1.0 else "insufficient"
        ),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
'''
USE_MCP = True  # Set to False to skip writing dscr_mcp_server.py

if USE_MCP:
    Path(__file__).with_name("dscr_mcp_server.py").write_text(DSCR_SERVER_CODE)
else:
    print("USE_MCP is False -- skipping server file creation. Set USE_MCP = True above to build it.")
