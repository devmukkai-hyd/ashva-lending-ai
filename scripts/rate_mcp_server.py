
"""A minimal MCP server exposing get_rate."""
import json
from pathlib import Path
 
from fastmcp import FastMCP
 
mcp = FastMCP("WealthPilot Rates Server")
 
# Full path to the reference rates file. Adjust if your project moves.
RATES_FILE = Path(
    "/Users/devarajan/Devmukkai-git/dev-repo/data-toolkit/capstone-data-toolkit"
    "/data/wealthpilot/mock_api/reference_rates.json"
)
 
 
def _load_rates() -> list[dict]:
    with open(RATES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
 
 
@mcp.tool()
def get_rate(rate_name: str) -> dict:
    """Look up a reference rate by name.
 
    Args:
        rate_name: Name of the rate, e.g. "assessment_rate_pct".
 
    Returns:
        {"rate": <name>, "value": <value>}
    """
    rates = _load_rates()
    for item in rates:
        if item.get("rate") == rate_name:
            return {"rate": rate_name, "value": item["value"]}
 
    available = [item.get("rate") for item in rates]
    raise ValueError(f"Rate not found: {rate_name!r}. Available rates: {available}")
 
 
if __name__ == "__main__":
    mcp.run(transport="stdio")
