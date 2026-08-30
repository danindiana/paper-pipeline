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


# ── Repair ───────────────────────────────────────────────────────────────────

_QUOTE_WRAPPED_LABEL_RE = re.compile(r"\[\s*\"label='(.*?)'\"?\s*\]")


def repair_dot_syntax(dot_src: str) -> str:
    """Repair a known invalid attribute-list pattern from the LLM.

    Observed live across a 308-paper batch: the model occasionally wraps an
    entire `label='...'` assignment in an extra, incorrect outer pair of
    double quotes — `Node ["label='text'"];` — instead of valid
    `Node [label="text"];`, sometimes also dropping the outer quote's closer
    (`Node ["label='text'];`). Both variants parse as a single bare quoted
    string with no attribute name, which `dot` rejects outright.

    This rewrites either variant to valid `[label="..."]` syntax, escaping
    the label text for its new quoting context (backslashes doubled before
    quotes are escaped) so the original characters render unchanged rather
    than being reinterpreted as Graphviz's own \\n/\\l/\\r label escapes.

    Confirmed against a real batch: fixes 5 of 55 diagrams that failed to
    render for other, more varied reasons (truncated generation, invented
    attribute names, control characters) — those are a materially different,
    much less tractable problem and are not addressed here. Confirmed to be
    a no-op on 1,111 diagrams that already rendered successfully (zero
    false-positive matches on valid syntax).
    """
    def repl(match: re.Match) -> str:
        inner = match.group(1)
        inner = inner.replace("\\", "\\\\").replace('"', '\\"')
        return f'[label="{inner}"]'

    return _QUOTE_WRAPPED_LABEL_RE.sub(repl, dot_src)


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
