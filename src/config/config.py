"""Project-relative paths and environment-controlled settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VECTORSTORE_PATH = str(BASE_DIR / "vectorstore")
PDF_FOLDER = str(BASE_DIR / "data" / "pdfs")
COLLECTION_NAME = "insurance"
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0"))
