"""
config.py — Declarative model routing and pipeline defaults.

All hardware-specific tuning (which models, which thresholds, VRAM caps)
lives here as plain data.  Nothing in this file imports anything else
from the package.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Ollama connection ────────────────────────────────────────────────────────
OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# ── Model tiers ──────────────────────────────────────────────────────────────
# Keys are semantic roles; values are Ollama model tags.
# Add / swap models here — nothing else in the codebase hard-codes a model name.
MODEL_TIERS: dict[str, str] = {
    "default":   "gemma4:26b-a4b-it-q4_K_M",   # MoE ~17 GB — fast, lighter
    "reasoning": "qwen3.6:35b",                 # MoE ~22 GB — stronger reasoning
}

# Page-count → tier mapping, evaluated top-to-bottom (first match wins).
TIER_THRESHOLDS: list[tuple[int, str]] = [
    (35,  "default"),       # short-to-standard papers
    (200, "reasoning"),     # long papers benefit from stronger reasoning
    (999, "default"),       # very large docs → lighter model, stays GPU-resident
]

# Models that cannot coexist in VRAM simultaneously.
# The VRAM manager evicts any loaded model in this set before loading another.
EXCLUSIVE_MODELS: set[str] = set(MODEL_TIERS.values())

# Per-model GPU-layer caps (Ollama num_gpu option).
# Empty = let Ollama auto-fill.  Add {"model_tag": N} to cap a specific model.
MODEL_GPU_LAYERS: dict[str, int] = {}

# ── Generation defaults ──────────────────────────────────────────────────────
DEFAULT_CTX_TOKENS: int = 32768
TEMPERATURE: float = 0.20
TOP_P: float = 0.90
REPEAT_PENALTY: float = 1.10

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_WINDOW: int = 12       # pages per chunk
CHUNK_OVERLAP: int = 2       # overlap between consecutive chunks
CHUNK_SUMMARY_CTX: int = 32768   # evidence mapping and reduction context
CHUNK_INPUT_CHAR_CAP: int = 48_000
CONTEXT_CAP: int = 45_000    # max chars sent to the model (~11k tokens)
DIAGRAM_CONTEXT_CAP: int = 30_000

# ── Hierarchical evidence synthesis ──────────────────────────────────────
EVIDENCE_SCHEMA_VERSION: int = 1
EVIDENCE_REDUCE_GROUP: int = 4
EVIDENCE_MAX_SELECTED: int = 16
EVIDENCE_MAP_RETRIES: int = 1
EVIDENCE_REDUCE_RETRIES: int = 1
EVIDENCE_SUMMARY_MAX_CHARS: int = 3_000

# ── OCR defaults ─────────────────────────────────────────────────────────────
DEFAULT_OCR_MAX_PAGES: int = 40

# ── Processing leases ─────────────────────────────────────────────────
LEASE_STALE_SECONDS: int = 4 * 3600
LEASE_HEARTBEAT_SECONDS: int = 60

# ── Storage ──────────────────────────────────────────────────────────────────
DEFAULT_DB_PATH: Path = Path("/mnt/nvme_staging/paper_processor_data/papers-v2.db")
DB_PATH_ENV_VAR: str = "PAPER_PROCESSOR_DB"
DEFAULT_PAPERS_DIR: Path = Path.home() / "Documents" / "AI-ML_Papers"

# ── Timeouts (seconds) ──────────────────────────────────────────────────────
GENERATE_TIMEOUT: int = 1200      # per-call Ollama timeout
EVICT_TIMEOUT: int = 20
SERVICE_RESTART_TIMEOUT: int = 45
HEALTH_CHECK_TIMEOUT: int = 5


def select_model(page_count: int, override: str | None = None) -> str:
    """Pick the model tier for a document by page count, or return the override."""
    if override:
        return override
    for threshold, tier_key in TIER_THRESHOLDS:
        if page_count <= threshold:
            return MODEL_TIERS[tier_key]
    return MODEL_TIERS["default"]
