
import argparse
import json
import os
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent

env_path = Path("/Users/devarajan/Devmukkai-git/dev-repo/data-toolkit/capstone-data-toolkit/.env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

GEMINI_API_KEY = os.getenv("DATAGEN_GEMINI_API_KEY")
LLM_MODEL = os.getenv("LAB_MODEL")
print(f"Loaded .env from {env_path}, GEMINI_API_KEY={GEMINI_API_KEY}, LLM_MODEL={LLM_MODEL}")
