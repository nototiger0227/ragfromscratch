"""
Central config. Keeping these in one place means when you tune the
pipeline later (bigger chunks? different model?) you change one file,
not five.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads a .env file in this folder into environment variables

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or ""

# Which Gemini models we use, and why:
EMBEDDING_MODEL = "models/gemini-embedding-001"  # current embedding model, free tier
CHAT_MODEL = "gemini-2.5-flash"                  # free tier, fast, good enough to start

# Chunking knobs — the two most important RAG tuning parameters
CHUNK_SIZE = 800       # characters per chunk (~150-200 tokens)
CHUNK_OVERLAP = 150    # characters shared between consecutive chunks

# Retrieval knobs
TOP_K = 5              # how many chunks to retrieve per query

# Where the persistent vector DB lives on disk
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "documents"
