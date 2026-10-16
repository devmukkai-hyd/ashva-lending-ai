"""
intake_validator.py - extract and validate loan applications (new 16-field intake schema).

For each line of the intake JSONL file:
  1. the LLM extracts exactly the 16 intake fields (EXTRACTION_PROMPT),
  2. the result is validated against LoanIntake (types, required fields, business rules),
  3. every record is written to the output file with
        reason_code        "Valid" | "Invalid"
        reason_description None when valid, otherwise why it was rejected.
Only the 16 intake fields plus the two reason fields are written (no ground_truth, no raw text).

Usage:
    python intake_validator.py                       # default intake file
    python intake_validator.py path/to/intake.jsonl
"""
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
#env_path = Path("/Users/devarajan/Devmukkai-git/dev-repo/data-toolkit/capstone-data-toolkit/.env")
#if env_path.exists():
#    load_dotenv(dotenv_path=env_path)
#else:
#    print(f"No .env file found at {env_path}. If the key is set in your shell, this script will still proceed.")

load_dotenv(Path(__file__).resolve().parent / ".env")   # the .env file in the scripts folder


API_KEY = os.getenv("DATAGEN_GEMINI_API_KEY")
LAB_MODEL = os.getenv("LAB_MODEL")

DATA_DIR = Path(__file__).resolve().parent.parent      # data/wealthpilot (the folder above scripts/)
INTAKE_FILE = DATA_DIR / "intake" / "TestIntake.jsonl"
OUTPUT_FILE = DATA_DIR / "data_output" / "intake_validated.json"

# ID formats seen in the data (ASH-L-00147 / ASH-A-00147). Adjust if other formats are legitimate.
APPLICATION_ID_PATTERN = r"^ASH-L-\d{5}$"
APPLICANT_ID_PATTERN = r"^ASH-A-\d{5}$"

# ---------------------------------------------------------------------------
# Extraction prompt: exactly the 16 intake fields
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = """Extract the loan application in the text below into a JSON object with exactly these fields:
application_id (string or null), applicant_id (string or null), received_on (date as YYYY-MM-DD or null),
requested_amount_inr (number or null), requested_tenor_months (number or null), purpose (string or null),
declared_annual_revenue_inr (number or null), existing_annual_debt_service_inr (number or null),
current_assets_inr (number or null), current_liabilities_inr (number or null),
 collateral_value_inr (number or null), export_receivables_usd (number or 0),
proprietor_name (string or null), city (string or null), application_language (string or English).
Rules:
- Write amounts as plain numbers with no commas or currency symbols, and convert lakh/crore to full
  numbers (e.g. "4.5 lakh" -> 450000).
- Use only information present in the text; do not guess or calculate values.
- If a field is missing or empty, set it to null.
- Do not add any other fields (no ground_truth, scores, sector or raw text).
Return ONLY valid JSON, no markdown fences, no commentary.
Text:
{text}"""

INTAKE_FIELDS = [
    "application_id", "applicant_id", "received_on", "requested_amount_inr", "requested_tenor_months",
    "purpose", "declared_annual_revenue_inr", "existing_annual_debt_service_inr", "current_assets_inr",
    "current_liabilities_inr", "collateral_value_inr", "export_receivables_usd",
    "proprietor_name", "city", "application_language",
]


def extract_fields(text: str) -> dict:
    """LLM extraction -> dict restricted to the 16 intake fields (anything else the model adds is dropped)."""
    resp = completion(
        model=LAB_MODEL,
        api_key=API_KEY,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}],
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    if not isinstance(data, dict):
        raise ValueError("model did not return a JSON object")
    return {k: data.get(k) for k in INTAKE_FIELDS}


# ---------------------------------------------------------------------------
# Pydantic model for the new prompt
# ---------------------------------------------------------------------------
class LoanIntake(BaseModel):
    # Required: without these the application can't be identified, dated or assessed (DSCR / liquidity).
    application_id: str = Field(pattern=APPLICATION_ID_PATTERN)
    applicant_id: str = Field(pattern=APPLICANT_ID_PATTERN)
    received_on: date
    requested_amount_inr: int = Field(gt=0)
    requested_tenor_months: int = Field(gt=0)
    declared_annual_revenue_inr: int = Field(gt=0)
    existing_annual_debt_service_inr: int = Field(ge=0)
    current_assets_inr: int = Field(ge=0)
    current_liabilities_inr: int = Field(ge=0)

    # Optional: recorded when present, but their absence doesn't reject the application.
    purpose: str | None = None
    collateral_value_inr: int | None = Field(default=None, ge=0)
    export_receivables_usd: float | None = Field(default=None, ge=0)
    proprietor_name: str | None = None
    city: str | None = None
    application_language: str | None = None

    @field_validator("received_on", mode="before")
    @classmethod
    def _date_only(cls, v):
        # Accept "2026-07-03T11:56:00" as well as "2026-07-03".
        return v[:10] if isinstance(v, str) else v

    @field_validator("requested_amount_inr", "requested_tenor_months", "declared_annual_revenue_inr",
                     "existing_annual_debt_service_inr", "current_assets_inr", "current_liabilities_inr",
                     "collateral_value_inr", "export_receivables_usd", mode="before")
    @classmethod
    def _plain_number(cls, v):
        # Tolerate "4,50,000" / "INR 450000" if the model ignores the formatting rule.
        if isinstance(v, str):
            if not v.strip():
                return None
            cleaned = re.sub(r"[^\d.\-]", "", v)
            return cleaned or v   # nothing numeric left -> keep the text so it fails as "not a number"
        return v

    @field_validator("purpose",  "proprietor_name", "city", "application_language",
                     mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        return v.strip() or None if isinstance(v, str) else v

    @model_validator(mode="after")
    def _business_rules(self):
        if self.received_on > date.today():
            raise ValueError(f"received_on {self.received_on} is in the future")
        return self


def describe_errors(e: ValidationError) -> str:
    """Pydantic errors -> one readable line, e.g. 'received_on: missing; requested_amount_inr: must be > 0'."""
    parts = []
    for err in e.errors():
        field = ".".join(str(x) for x in err["loc"])
        if err["type"] == "missing" or (err["type"].endswith("_type") and err.get("input") is None):
            msg = "missing"
        else:
            msg = err["msg"].removeprefix("Value error, ")
        parts.append(f"{field}: {msg}" if field else msg)   # business rules have no single field
    return "; ".join(parts)


def validate(extracted: dict) -> dict:
    """Return the output row: the 16 fields + reason_code + reason_description."""
    try:
        record = LoanIntake(**extracted)
    except ValidationError as e:
        return {**{k: extracted.get(k) for k in INTAKE_FIELDS},
                "reason_code": "Invalid", "reason_description": describe_errors(e)}
    return {**record.model_dump(mode="json"), "reason_code": "Valid", "reason_description": None}


def invalid_row(description: str, extracted: dict | None = None) -> dict:
    extracted = extracted or {}
    return {**{k: extracted.get(k) for k in INTAKE_FIELDS},
            "reason_code": "Invalid", "reason_description": description}


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------
def main() -> None:
    if not API_KEY or not LAB_MODEL:
        sys.exit("DATAGEN_GEMINI_API_KEY and LAB_MODEL must be set.")

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else INTAKE_FILE
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                row = invalid_row(f"Line {line_number} is not valid JSON: {e}")
            else:
                # Send the record as JSON text, without ground_truth, so the model can't copy it.
                source = {k: v for k, v in data.items() if k != "ground_truth"} if isinstance(data, dict) else data
                try:
                    row = validate(extract_fields(json.dumps(source, ensure_ascii=False, default=str)))
                except Exception as e:  # rate limit, bad JSON from the model, network
                    row = invalid_row(f"Extraction failed: {e}")
            rows.append(row)
#            print(f"Line {line_number}: {row['application_id'] or '?'} -> {row['reason_code']}"
#                  + (f" ({row['reason_description']})" if row["reason_description"] else ""))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    valid = sum(r["reason_code"] == "Valid" for r in rows)
    print(f"\n{valid} valid, {len(rows) - valid} invalid -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()