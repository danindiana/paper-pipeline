"""
diagrams.py — Parse, style-enforce, and render Graphviz DOT diagrams.

Handles the full lifecycle of diagram output from the LLM:
  1. Parse delimited DOT blocks (or fenced fallback) from raw text
  2. Inject neon-on-black styling if the model forgot
  3. Render DOT → SVG via the system `dot` binary
"""

from __future__ import annotations

import re
import subprocess
from typing import Optional

# ── Parsing ──────────────────────────────────────────────────────────────────

_DELIM_RE = re.compile(
    r"===DIAGRAM_START:\s*(.+?)===\s*(.*?)===DIAGRAM_END===",
    re.DOTALL | re.IGNORECASE,
)

_FENCE_RE = re.compile(
    r"```(?:dot|graphviz)?\s*\n?((?:digraph|graph)\b.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def parse_diagrams(raw: str) -> list[tuple[str, str]]:
    """Extract (title, dot_source) pairs from LLM output.

    Tries the ===DIAGRAM_START/END=== delimiters first, falls back to
    fenced code blocks containing digraph/graph keywords.
    """
    results: list[tuple[str, str]] = []

    for m in _DELIM_RE.finditer(raw):
        title = m.group(1).strip()
        dot = m.group(2).strip()
        if dot and ("digraph" in dot.lower() or "graph" in dot.lower()):
            results.append((title, dot))

    if not results:
        for idx, m in enumerate(_FENCE_RE.finditer(raw), 1):
            results.append((f"diagram_{idx:02d}", m.group(1).strip()))

    return results


# ── Style enforcement ────────────────────────────────────────────────────────

def ensure_neon_black(dot_src: str) -> str:
    """Inject bgcolor=black and default neon node/edge styles if absent."""
    if "bgcolor" not in dot_src:
        dot_src = re.sub(
            r"((?:di)?graph\s+\w*\s*\{)",
            r'\1\n  graph [bgcolor="black" fontcolor="#00FF41" fontname="Courier New"];'
            r'\n  node  [style=filled fillcolor="#0a0a0a" color="#00FF41" fontcolor="#00FF41" fontname="Courier New"];'
            r'\n  edge  [color="#FF00FF" penwidth=2.0];',
            dot_src,
            count=1,
        )
    return dot_src


# ── Rendering ────────────────────────────────────────────────────────────────

def render_dot(dot_src: str) -> Optional[str]:
    """Render DOT source to SVG string via the system `dot` binary.

    Returns None (not raises) on any failure — one bad diagram should
    never abort a whole paper run.
    """
    try:
        r = subprocess.run(
            ["dot", "-Tsvg"],
            input=dot_src,
            text=True,
            capture_output=True,
            timeout=30,
        )
        return r.stdout if r.returncode == 0 else None
    except FileNotFoundError:
        print("      ⚠️  graphviz `dot` not found — SVGs will not be rendered")
        print("          Fix: sudo apt install graphviz")
        return None
    except Exception:
        return None
