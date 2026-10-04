import os
import sys
import ctypes
import warnings
import sqlite3
import hashlib
import json
import numpy as np
from datetime import datetime

# Windows PyTorch / OpenMP DLL pre-loading setup to fix WinError 1114
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib):
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(torch_lib)
                except Exception:
                    pass
            for dll_name in ["libiomp5md.dll", "c10.dll", "torch_cpu.dll"]:
                dll_path = os.path.join(torch_lib, dll_name)
                if os.path.exists(dll_path):
                    try:
                        ctypes.CDLL(dll_path)
                    except Exception:
                        pass
    except Exception:
        pass

# Global model cache to avoid re-loading within same process
_MODEL_CACHE = {}
_EMBEDDING_DIM = {}  # Cache actual embedding dimension per model name

# Model priority: lightweight first (always works), heavier models as upgrades
MODEL_NAME_PRIMARY = "all-MiniLM-L6-v2"       # 384-dim, fast, reliable
MODEL_NAME_FALLBACK = "BAAI/bge-small-en-v1.5" # 384-dim
MODEL_NAME_LIGHTWEIGHT = "BAAI/bge-m3"         # 1024-dim, heavy — optional upgrade

DEFAULT_EMBEDDING_DIM = 384  # Matches the primary model

def init_embedding_db(db_path="Resume.db"):
    """Initialize vector storage table in SQLite if not existing."""
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS resume_embeddings (
            text_hash TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            vector_blob BLOB NOT NULL,
            dim INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Warning: Could not initialize embedding DB: {e}")

def load_bge_m3_model(preferred_model=MODEL_NAME_PRIMARY):
    """
    Loads embedding model using sentence-transformers.
    Tries preferred_model first, then falls back through alternatives.
    Primary is all-MiniLM-L6-v2 (fast, reliable, 384-dim).
    """
    global _MODEL_CACHE, _EMBEDDING_DIM
    if preferred_model in _MODEL_CACHE:
        return _MODEL_CACHE[preferred_model], preferred_model

    # Try preferred, then fallbacks in order
    candidates = [preferred_model] + [
        m for m in [MODEL_NAME_PRIMARY, MODEL_NAME_FALLBACK, MODEL_NAME_LIGHTWEIGHT]
        if m != preferred_model
    ]

    for model_name in candidates:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {model_name}...")
            model = SentenceTransformer(model_name)
            # Probe actual dimension
            test_vec = model.encode("probe", normalize_embeddings=True)
            _EMBEDDING_DIM[model_name] = len(test_vec)
            _MODEL_CACHE[model_name] = model
            print(f"✅ Loaded '{model_name}' (dim={_EMBEDDING_DIM[model_name]})")
            return model, model_name
        except Exception as e:
            print(f"⚠️  Could not load '{model_name}': {e}")
            continue

    print("❌ All embedding models failed. Scoring will use zero vectors.")
    return None, "none"

def compute_text_hash(text):
    """Generate SHA256 hash for exact text content matching."""
    return hashlib.sha256(text.strip().encode('utf-8')).hexdigest()

def get_cached_embedding(text_hash, model_name, db_path="Resume.db"):
    """Retrieve embedding vector from SQLite cache."""
    try:
        init_embedding_db(db_path)
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT vector_blob, dim FROM resume_embeddings WHERE text_hash = ? AND model_name = ?", (text_hash, model_name))
        row = c.fetchone()
        conn.close()
        if row:
            blob, dim = row
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.shape[0] == dim:
                return vec
    except Exception as e:
        print(f"Cache lookup error: {e}")
    return None

def store_cached_embedding(text_hash, model_name, vector, db_path="Resume.db"):
    """Store embedding vector into SQLite cache."""
    try:
        init_embedding_db(db_path)
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        blob = vector.astype(np.float32).tobytes()
        dim = int(vector.shape[0])
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute(
            "INSERT OR REPLACE INTO resume_embeddings (text_hash, model_name, vector_blob, dim, created_at) VALUES (?, ?, ?, ?, ?)",
            (text_hash, model_name, blob, dim, now)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Cache store error: {e}")

def _get_zero_vector(model_name=MODEL_NAME_PRIMARY):
    """Return a zero vector with the correct dimension for the given model."""
    dim = _EMBEDDING_DIM.get(model_name, DEFAULT_EMBEDDING_DIM)
    return np.zeros(dim, dtype=np.float32)


def get_embedding(text, model=None, model_name=MODEL_NAME_PRIMARY, use_cache=True, db_path="Resume.db"):
    """
    Get embedding for input text.
    Uses cached SQLite storage if available; otherwise computes and caches embedding.
    Zero-vector fallback dimension matches the loaded model to avoid shape mismatches.
    """
    if not text or not text.strip():
        return _get_zero_vector(model_name)

    text_hash = compute_text_hash(text)

    if use_cache:
        cached = get_cached_embedding(text_hash, model_name, db_path)
        if cached is not None:
            return cached

    if model is None:
        model, actual_model_name = load_bge_m3_model(model_name)
    else:
        actual_model_name = model_name

    if model is not None:
        try:
            vec = model.encode(text, normalize_embeddings=True)
            vec = np.array(vec, dtype=np.float32)
            # Update cached dimension
            _EMBEDDING_DIM[actual_model_name] = len(vec)
            if use_cache:
                store_cached_embedding(text_hash, actual_model_name, vec, db_path)
            return vec
        except Exception as e:
            print(f"Embedding encoding error: {e}")

    # Fallback to correctly-sized zero vector
    return _get_zero_vector(model_name)

def compute_cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two 1D numpy vectors."""
    if vec1 is None or vec2 is None:
        return 0.0
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    sim = np.dot(vec1, vec2) / (norm1 * norm2)
    return float(np.clip(sim, 0.0, 1.0))
