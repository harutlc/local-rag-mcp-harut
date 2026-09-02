# Configuration for Company Knowledge Base Assistant

# Document directory - update this to point to your company documentation
DOCUMENTS_DIR = "./docs"

# Chunking configuration
# taking account that token max size for all-MiniLM-L6-v2 is 256 and tokenization differs across different models and the difference is about 5-10% chunk size is set 220 so +10% is less than 250
CHUNK_SIZE = 220
CHUNK_OVERLAP = 40

# Embedding model
# max token size is 256
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# 512 tokens max
# EMBEDDING_MODEL = "bge-small-en-v1.5"
# 8192 tokes max
# EMBEDDING_MODEL = "nomic-embed-text"

# Storage paths (relative to src directory)
# FAISS holds the vectors; SQLite holds the chunk text and the FTS index
FAISS_INDEX_PATH = "data/index.faiss"
CHUNKS_DB_PATH = "data/chunks.db"

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"

# RAG retrieval configuration
TOP_K = 5
# Full-text search over-fetches relative to the vector side so that rank fusion
# has enough candidates to work with
FTS_TOP_K = 10

# Fallback message node: used when query expansion produces nothing usable
FALLBACK_MODEL = "qwen2.5:3b"
FALLBACK_NUM_SUGGESTIONS = 3
FALLBACK_TEMPERATURE = 0.4

# Use the full hybrid pipeline (query expansion -> parallel vector + FTS ->
# pool/dedupe -> RRF) instead of a single vector search. Costs one extra LLM
# call per question for the expansion; set False to fall back to vector-only.
USE_HYBRID_RETRIEVAL = True

# Reciprocal Rank Fusion: damping constant. Higher values flatten the advantage
# of top positions, so agreement across searches matters more than any single
# search's first place. 60 is the value from the original RRF paper.
RRF_K = 60

# Query expansion configuration
# Base host for the `ollama` python package client (OLLAMA_URL above is the raw
# /api/generate endpoint still used by rag/query.py)
OLLAMA_HOST = "http://localhost:11434"
# Model used to generate alternative queries and keywords. Can be swapped for a
# smaller/faster model, e.g. qwen3:1.7b
EXPANSION_MODEL = "qwen2.5:3b"
EXPANSION_NUM_ALTERNATIVES = 3
EXPANSION_NUM_KEYWORDS = 6
EXPANSION_TEMPERATURE = 0.1
# seconds
EXPANSION_TIMEOUT = 30
