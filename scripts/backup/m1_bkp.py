import os, json, time
import json
from pydantic import ValidationError
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, Field
from litellm import completion
from pydantic import BaseModel
from datetime import datetime



EXTRACTION_PROMPT = """Extract loan data from the text below into JSON with exactly these fields:
application_id (String or null), applicant_id (String or null), loan_tenor (number or null), received_on (datetime or null),
business_name (String or null), sector (String or null), requested_amount_inr (number or null), requested_tenor_months (number or null), 
declared_financials (object with annual_revenue_inr, ebitda_inr, existing_debt_inr, existing_annual_debt_service_inr, current_assets_inr, 
current_liabilities_inr, months_operating, collateral_value_inr), bureau_score (number or null), 
ground_truth (object with dscr, meets_dscr_floor, risk_band, requires_human_signoff, missing_fields), record_id (String or null). Dont add raw data to the output. 
If a field is missing, set it to null. If a field is present but empty, set it to null.
Return ONLY valid JSON, no markdown fences, no commentary.
Text:
{text}"""


def extract_raw(model: str,required_keys: str,text: str) -> str:
    # Implement extraction from raw data
    resp = completion(
        model=model,
        api_key=required_keys,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}],
        response_format={"type": "json_object"},
    )

    return resp.choices[0].message.content






class DeclaredFinancials(BaseModel):
    annual_revenue_inr: int
    ebitda_inr: int
    existing_debt_inr: int
    existing_annual_debt_service_inr: int
    current_assets_inr: int
    current_liabilities_inr: int
    months_operating: int
    collateral_value_inr: int


class GroundTruth(BaseModel):
    dscr: float
    meets_dscr_floor: bool
    risk_band: str
    requires_human_signoff: bool
    missing_fields: list[str]


class LoanApplication(BaseModel):
    application_id: str
    applicant_id: str
    channel: str
    received_at: datetime
    business_name: str
    sector: str
    requested_amount_inr: int
    requested_tenor_months: int
    declared_financials: DeclaredFinancials
    bureau_score: int
    ground_truth: GroundTruth
    record_id: str


# Need to set the path to your .env file. Adjust this path as necessary.

env_path = Path("/Users/devarajan/Devmukkai-git/dev-repo/data-toolkit/capstone-data-toolkit/.env")

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    print(f"No .env file found at {env_path}. Copy .env.template to .env and add your API key.")
    print("If you already set the key in your shell, this cell will still proceed.")

required_keys = os.getenv("DATAGEN_GEMINI_API_KEY")
lab_model = os.getenv("LAB_MODEL")


if not required_keys:
    print("Error: required_keys is empty or None")
else:
    print("Success: required_keys is available")

valid_records = []
invalid_records = []
filename="/Users/devarajan/Devmukkai-git/dev-repo/data-toolkit/capstone-data-toolkit/data/wealthpilot/intake/TestIntake.jsonl"
with open(filename, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

                             
            try:
                # Step 1: JSON string -> Python dictionary
                data = json.loads(line)
                raw = extract_raw(lab_model,required_keys, data)
                appended_data = {**data, "extracted_raw": raw}
                    # Step 2: Validate the dictionary against the Pydantic model
                validated_record = LoanApplication(**appended_data)
                valid_records.append(validated_record)
            except json.JSONDecodeError as e:
                print(f"Line {line_number}: JSON decoding error: {e}")
                invalid_records.append({"line_number": line_number, "error": str(e), "line": line})
            except ValidationError as e:
                    print(f"Line {line_number}: Validation error: {e}")
                    invalid_records.append({"line_number": line_number, "error": str(e), "line": line})  
                       
with open("/Users/devarajan/Devmukkai-git/dev-repo/data-toolkit/capstone-data-toolkit/data/wealthpilot/data_output/valid_records.json", "w", encoding="utf-8") as f:
    json.dump(
        [record.model_dump(mode="json") for record in valid_records],
        f,
        indent=2,
        ensure_ascii=False
    )

print("Records saved to valid_records.json")




