"""
Conversation memory classifier -> Supermemory
LLM: Gemini via LiteLLM (no OpenAI dependency)

Each record's "raw_narrative" is split into sentences, every sentence gets its own
memory tier (Working / Episodic / Semantic / Procedural), and each persistable
sentence is stored as a separate Supermemory document under the record's "applicant_id".

pip install litellm python-dotenv pydantic supermemory

.env
    GOOGLE_API_KEY       -> required: powers the LLM (mirrored into GEMINI_API_KEY for LiteLLM)
    SUPERMEMORY_API_KEY  -> required: the long-term memory store (get one at console.supermemory.ai)
    LLM_MODEL            -> optional: defaults to gemini/gemini-flash-latest
    RECORDS_PATH         -> optional: defaults to records.jsonl

Usage:
    python memory_classifier.py [records.jsonl]
"""
import json
import os
import re
import sys
import time
from enum import Enum
from pathlib  import Path
import supermemory, time
from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
load_dotenv(".env", override=True)

env_path = Path("/Users/devarajan/Devmukkai-git/dev-repo/data-toolkit/capstone-data-toolkit/.env")

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    print(f"No .env file found at {env_path}. Copy .env.template to .env and add your API key.")
    print("If you already set the key in your shell, this script will still proceed.")

GEMINI_API_KEY = os.getenv("DATAGEN_GEMINI_API_KEY")
print(GEMINI_API_KEY)
LLM_MODEL = os.getenv("LAB_MODEL")
if not GEMINI_API_KEY:
    print("Warning: DATAGEN_GEMINI_API_KEY is not set.")
if not LLM_MODEL:
    print("Warning: LAB_MODEL is not set.")




# Each record's applicant_id is used as its Supermemory container tag (multi-tenant isolation).
RECORDS_PATH = os.getenv("RECORDS_PATH", "records.jsonl")
TEXT_KEY = "raw_narrative"
USER_ID_KEY = "applicant_id"

SUPERMEMORY_API_KEY = os.getenv("SUPERMEMORY_API_KEY", "")
mem = supermemory.Supermemory(api_key=SUPERMEMORY_API_KEY)



# ---------------------------------------------------------------------------
# 1. Memory taxonomy + structured response schema
# ---------------------------------------------------------------------------
class MemoryType(str, Enum):
    WORKING = "Working"
    EPISODIC = "Episodic"
    SEMANTIC = "Semantic"
    PROCEDURAL = "Procedural"


class SentenceClassification(BaseModel):
    index: int = Field(description="The sentence number exactly as given in the input list.")
    category: MemoryType = Field(description="The memory classification for this sentence.")
    justification: str = Field(description="Brief explanation of why it fits this specific tier.")
    should_persist: bool = Field(
        description="True if it represents long-term data. False if it's transient working memory."
    )


class NarrativeClassification(BaseModel):
    sentences: list[SentenceClassification] = Field(
        description="One entry per input sentence, in the same order."
    )


# ---------------------------------------------------------------------------
# 2. Sentence splitting
# ---------------------------------------------------------------------------
# Split after . ! ? when followed by whitespace and an upper-case letter or quote.
# Safe for this dataset: no abbreviations like "Mr." and no decimals ("1.5") in the narratives.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'A-Z])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]


# ---------------------------------------------------------------------------
# 3. Router: Gemini (via LiteLLM) classifies all sentences of a narrative in one call
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = "You are a memory sorting mechanism for an advanced AI agent architecture."

CLASSIFY_PROMPT = """
Below is a loan applicant's narrative, split into numbered sentences. Classify EACH sentence
into one of four memory categories. Use the other sentences only as context; judge each one on its own.

1. Working: Immediate, session-specific instructions, corrections, or references to the 'here and now'
   (e.g., 'Change this sentence right now', 'Look at the code above'). Greetings, sign-offs and
   requests to process this application now also belong here. These shouldn't be stored long-term.
2. Episodic: Real-world personal events, past experiences, or timelines tied to a specific occurrence
   (e.g., 'I presented a project yesterday', 'We had a crash last week').
3. Semantic: Fact-based, stable knowledge, settings, structural rules, or long-term preferences
   (e.g., 'I am allergic to peanuts', 'Python doesn't need semicolons').
4. Procedural: Guidelines, automation criteria, if/then multi-step workflows, or methodologies on how
   tasks must be executed moving forward (e.g., 'Always summarize before writing code').

Return JSON with a "sentences" list containing exactly one entry per numbered sentence, in order,
each with fields: index, category, justification, should_persist.

Sentences:
{numbered}
"""


def classify_sentences(sentences: list[str]) -> list[SentenceClassification]:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
    response = completion(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": CLASSIFY_PROMPT.format(numbered=numbered)},
        ],
        api_key=GEMINI_API_KEY,
        response_format=NarrativeClassification,  # LiteLLM converts this to Gemini's JSON schema
        temperature=0,
    )
    parsed = NarrativeClassification.model_validate_json(response.choices[0].message.content)

    # Map results back by index so a reordered or missing entry can't attach a label to the wrong sentence.
    by_index = {item.index: item for item in parsed.sentences}
    missing = [i for i in range(len(sentences)) if i not in by_index]
    if missing:
        raise ValueError(f"LLM returned no classification for sentence(s) {missing}")

    results = []
    for i in range(len(sentences)):
        item = by_index[i]
        # Guardrail: working memory is never persisted, whatever the model says.
        if item.category == MemoryType.WORKING:
            item.should_persist = False
        results.append(item)
    return results


# ---------------------------------------------------------------------------
# 4. Storage + recall: Supermemory SDK, scoped by container tag (the applicant_id)
# ---------------------------------------------------------------------------
def remember(text: str, kind: str, user_id: str, **extra):
    """Write one memory into the store (stored as a document, tagged with its kind).

    Supermemory SDK v5: the first argument is the namespace, which scopes memories per applicant.
    """
    # Supermemory metadata values must be str/number/bool, so drop empty fields.
    metadata = {"type": kind, "source": "conversation_classifier",
                **{k: v for k, v in extra.items() if v is not None}}
    return mem.add(user_id, content=text, metadata=metadata)


def store_in_supermemory(
    text: str, category: str, justification: str, user_id: str, extra_metadata: dict | None = None
) -> None:
    # Self-contained document so Supermemory can index/graph it on its own.
    content = f"[{category} Memory] User Statement: {text}\nContext/Justification: {justification}"
    try:
        ref = remember(content, category, user_id, **(extra_metadata or {}))
        print(f"   ✅ Saved [{user_id}] as '{category}'. id={ref.id} status={ref.status}")
    except Exception as e:  # SDK raises ApiError subclasses; keep the batch going
        print(f"   ❌ Supermemory error: {e}")


def recall(query: str, user_id: str, k: int = 3) -> list[tuple[str, str, float]]:
    """Recall an applicant's memories by meaning -> [(type, text, score)].

    Supermemory SDK v5 has a single search() call. Its default "hybrid" mode returns learned
    memories (`.memory`) alongside source passages (`.chunk`), so either field can hold the text.
    """
    out = []
    for r in mem.search(user_id, query=query, limit=k).results:
        text = r.memory or r.chunk
        if text:
            out.append(((r.metadata or {}).get("type", "?"), text, float(r.similarity or 0.0)))
    return out


# ---------------------------------------------------------------------------
# 5. Pipeline
# ---------------------------------------------------------------------------
def process_record(row: dict) -> None:
    user_id = row[USER_ID_KEY]
    sentences = split_sentences(row[TEXT_KEY])
    print(f"\n[{user_id}] {len(sentences)} sentences")

    try:
        results = classify_sentences(sentences)
    except Exception as e:  # one bad LLM response shouldn't stop the whole batch
        print(f"   ❌ Classification failed, record skipped: {e}")
        return

    for i, (sentence, analysis) in enumerate(zip(sentences, results)):
        preview = sentence if len(sentence) <= 100 else sentence[:100] + "..."
        print(f" {i}. {analysis.category.value:<10} | {preview}")

        if analysis.should_persist:
            store_in_supermemory(
                sentence,
                analysis.category.value,
                analysis.justification,
                user_id,
                extra_metadata={
                    "application_id": row.get("application_id"),
                    "record_id": row.get("record_id"),
                    "sentence_index": i,
                },
            )
        else:
            print("   ⏳ Skipped: working memory stays in the active session.")


def load_records(path: str) -> list[dict]:
    """Read JSONL records, skipping blank lines and rows missing the narrative or applicant id."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get(TEXT_KEY) or not row.get(USER_ID_KEY):
                print(f"⚠️  Line {line_no}: missing '{TEXT_KEY}' or '{USER_ID_KEY}', skipped.")
                continue
            records.append(row)
    return records


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else RECORDS_PATH
    records = load_records(path)
    print(f"\nLoaded {len(records)} records from {path}")

    for row in records:
        process_record(row)

    # Quick recall check on the first applicant. Supermemory extracts memories asynchronously,
    # so give ingestion a moment before searching.
    if records:
        demo_user = records[0][USER_ID_KEY]
        time.sleep(int(os.getenv("RECALL_WAIT_SECONDS", "10")))
        print(f"\nRecall for {demo_user}: 'loan amount and purpose'")
        for kind, text, score in recall("loan amount and purpose", demo_user):
            print(f"  [{kind}] ({score:.2f}) {text[:120]}")