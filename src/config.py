from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_GOLDEN = DATA_DIR / "golden"
DATA_SAMPLE = DATA_DIR / "sample"

RESULTS_DIR = PROJECT_ROOT / "results"

for _dir in (DATA_RAW, DATA_PROCESSED, DATA_GOLDEN, DATA_SAMPLE, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

INSPECTION_SUMMARY_PATH = DATA_PROCESSED / "dataset_inspection.json"
BRAND_FREQUENCIES_PATH = DATA_PROCESSED / "brand_frequencies.csv"
SELECTED_BRAND_PATH = DATA_PROCESSED / "selected_brand.json"
CLEANED_DATA_PATH = DATA_PROCESSED / "cleaned.csv"


def conversations_path_for(brand: str) -> Path:
    safe_name = "".join(c if c.isalnum() else "_" for c in brand.lower())
    return DATA_PROCESSED / f"conversations_{safe_name}.csv"


@dataclass
class Settings:
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "none").lower())

    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY") or None)
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))

    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY") or None)
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    gemini_api_key: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY") or None)
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))

    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

    min_intent_confidence: float = field(
        default_factory=lambda: float(os.getenv("MIN_INTENT_CONFIDENCE", "0.05"))
    )
    min_retrieval_similarity: float = field(
        default_factory=lambda: float(os.getenv("MIN_RETRIEVAL_SIMILARITY", "0.45"))
    )

    def llm_configured(self) -> bool:
        if self.llm_provider == "groq":
            return bool(self.groq_api_key)
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        if self.llm_provider == "gemini":
            return bool(self.gemini_api_key)
        return False


SETTINGS = Settings()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger