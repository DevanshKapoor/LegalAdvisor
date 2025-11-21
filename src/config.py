import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_PATH = os.path.join(BASE_DIR, "database")
CHARTS_OUTPUT_PATH = os.path.join(BASE_DIR, "performance_metrics.png")

# Models
RETRIEVER_MODEL_ID = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
RERANKER_MODEL_ID = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
LLM_MODEL_ID = "google/gemma-2b-it"

# Generation Settings
MAX_NEW_TOKENS = 200

# Language Mapping
LANGUAGE_OPTIONS = {
    "en": "English",
    "hi": "Hindi",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali"
}