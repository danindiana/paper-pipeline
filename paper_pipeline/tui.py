"""
tui.py — read-only curses dashboard: batch progress, database entities,
and system/visualization status, all in one place.

This is the recommended entry point for *watching and understanding* an
ongoing run. It never writes to the database and never starts, stops, or
reprocesses anything -- use the `paper-pipeline` command itself for that.

Every view is split into a pure data layer (no curses, fully unit-testable)
and a pure render layer (turns that data into plain text lines, also no
curses). The curses driver itself is a thin loop that polls both and blits
the result to the screen -- deliberately not unit tested, matching this
project's existing precedent of shellcheck + manual smoke test for
scripts/*.sh rather than forcing tests onto glue code with no real logic.

`--once` prints all three views as plain text and exits without touching
curses at all -- useful for scripting/logging/piping, and how this module
was verified against the real live database (a real interactive curses
screen can't be screenshotted in an unattended environment).
"""

from __future__ import annotations

import argparse
import sys
import json
import os
import socket
import sqlite3
import subprocess
import textwrap
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config, store

# Must match scripts/graph_viz.sh's COSMOS_PORT.
GRAPH_VIZ_PORT = 8687
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


# ══════════════════════════════════════════════════════════════════════════
# Data layer — pure, no curses, fully unit-testable
# ══════════════════════════════════════════════════════════════════════════

def _readonly_connect(db_path: Path) -> sqlite3.Connection:
    """Open the database strictly read-only -- this tool never writes."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@dataclass
class PaperStatusRow:
    name: str
    status: str  # "complete" | "partial" | "not started"
    missing: tuple[str, ...]
    in_progress: bool


@dataclass
class BatchProgress:
    papers_dir: str
    total: int = 0
    complete: int = 0
    partial: int = 0
    not_started: int = 0
    rows: list[PaperStatusRow] = field(default_factory=list)
    error: Optional[str] = None


def get_batch_progress(db_path: Path, papers_dir: Path) -> BatchProgress:
    if not papers_dir.is_dir():
        return BatchProgress(str(papers_dir), error=f"not a directory: {papers_dir}")
    if not db_path.is_file():
        return BatchProgress(str(papers_dir), error=f"database not found: {db_path}")

    try:
        conn = _readonly_connect(db_path)
    except sqlite3.OperationalError as exc:
        return BatchProgress(str(papers_dir), error=str(exc))

    try:
        # A lease is only trustworthy evidence of "being processed right
        # now" if it's both active *and* recently renewed -- a crashed
        # worker can leave an active=1 row behind that heartbeats no longer
        # touch, and that's a stale lease, not live work. Leases are keyed
        # by content hash ("sha256:<paper_hash>"), not by path.
        cutoff = time.time() - 2 * config.LEASE_HEARTBEAT_SECONDS
        active_lease_keys = {
            r["resource_key"]
            for r in conn.execute(
                "SELECT resource_key, renewed_at FROM processing_leases WHERE active=1"
            ).fetchall()
            if r["renewed_at"] >= cutoff
        }

        pdfs = sorted(p for p in papers_dir.rglob("*.pdf") if "_processed" not in p.parts)

        # First pass: papers that already have a metadata row give us their
        # paper_hash directly -- resolve their lease membership for free.
        records: dict[Path, Optional[store.PaperRecord]] = {}
        unmatched_leases = set(active_lease_keys)
        for pdf in pdfs:
            record = store.load_paper_by_pdf_path(conn, str(pdf))
            records[pdf] = record
            if record is not None:
                unmatched_leases.discard(f"sha256:{record.paper_hash}")

        # Second pass, only if needed: a paper can be claimed and mid-way
        # through evidence synthesis *before* its first metadata row is ever
        # written, so an active lease can exist with nothing in `papers` yet
        # to match it against. Hashing a PDF is a full-file streaming SHA-256
        # (see reader.hash_file) -- too expensive to do for every not-started
        # PDF on every refresh, so only do it, and only for "not started"
        # papers, and stop as soon as every remaining lease is accounted for.
        in_progress_paths: set[Path] = set()
        if unmatched_leases:
            from .reader import hash_file
            for pdf, record in records.items():
                if not unmatched_leases:
                    break
                if record is not None:
                    continue
                key = f"sha256:{hash_file(pdf)}"
                if key in unmatched_leases:
                    in_progress_paths.add(pdf)
                    unmatched_leases.discard(key)

        rows: list[PaperStatusRow] = []
        n_complete = n_partial = n_not_started = 0
        for pdf in pdfs:
            record = records[pdf]
            in_progress = pdf in in_progress_paths or (
                record is not None and f"sha256:{record.paper_hash}" in active_lease_keys
            )
            if record is None:
                status, missing = "not started", ()
                n_not_started += 1
            elif store.ALL_SECTIONS.issubset(set(record.sections_completed)):
                status, missing = "complete", ()
                n_complete += 1
            else:
                missing = tuple(sorted(store.ALL_SECTIONS - set(record.sections_completed)))
                status = "partial"
                n_partial += 1
            rows.append(PaperStatusRow(pdf.name, status, missing, in_progress))

        return BatchProgress(
            str(papers_dir), len(pdfs), n_complete, n_partial, n_not_started, rows
        )
    finally:
        conn.close()


@dataclass
class EntityCard:
    table: str
    description: str
    stats: list[str]


def get_db_entity_report(db_path: Path) -> tuple[list[EntityCard], Optional[str]]:
    if not db_path.is_file():
        return [], f"database not found: {db_path}"
    try:
        conn = _readonly_connect(db_path)
    except sqlite3.OperationalError as exc:
        return [], str(exc)

    try:
        cards: list[EntityCard] = []

        n_papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        n_complete = 0
        for row in conn.execute("SELECT sections_completed FROM papers"):
            sections = set(json.loads(row["sections_completed"] or "[]"))
            if store.ALL_SECTIONS.issubset(sections):
                n_complete += 1
        cards.append(EntityCard(
            "papers",
            "One row per PDF ever processed. Tracks its content-hash identity, "
            "which model processed it, and the five generated output sections "
            "(summary, symbolic logic, C++ examples, diagrams, extras) plus the "
            "evidence corpus they were built from.",
            [f"{n_papers} total", f"{n_complete} fully complete"],
        ))

        n_diagrams = conn.execute("SELECT COUNT(*) FROM diagrams").fetchone()[0]
        n_rendered = conn.execute(
            "SELECT COUNT(*) FROM diagrams WHERE svg_content IS NOT NULL"
        ).fetchone()[0]
        cards.append(EntityCard(
            "diagrams",
            "Up to 6 Graphviz diagrams per paper. Each row holds the raw DOT "
            "source the model generated and the rendered SVG, when rendering "
            "succeeded (some fail -- see diagrams/05_future_directions in the "
            "repo for the known reasons why).",
            [f"{n_diagrams} total", f"{n_rendered} rendered",
             f"{n_diagrams - n_rendered} failed to render"],
        ))

        n_ocr = conn.execute("SELECT COUNT(*) FROM ocr_cache").fetchone()[0]
        cards.append(EntityCard(
            "ocr_cache",
            "Per-page OCR text, cached by (paper, page, OCR settings) so "
            "re-running with the same DPI/language doesn't re-run Tesseract "
            "unnecessarily.",
            [f"{n_ocr} cached page(s)"],
        ))

        n_leases = conn.execute("SELECT COUNT(*) FROM processing_leases").fetchone()[0]
        n_active = conn.execute(
            "SELECT COUNT(*) FROM processing_leases WHERE active=1"
        ).fetchone()[0]
        cards.append(EntityCard(
            "processing_leases",
            "Which paper (if any) each worker currently owns, with a "
            "generation counter that prevents a stale or crashed worker from "
            "overwriting a newer one's work. An active, recently-renewed "
            "lease means that paper is being processed right now -- see the "
            "Batch Progress view.",
            [f"{n_leases} total (history)", f"{n_active} currently marked active"],
        ))

        schema_version = conn.execute("PRAGMA user_version").fetchone()[0]
        identity_row = conn.execute(
            "SELECT value FROM schema_meta WHERE key='identity_scheme'"
        ).fetchone()
        cards.append(EntityCard(
            "schema_meta",
            "Bookkeeping that confirms this database uses the current "
            "paper-identity scheme (full SHA-256 hashes), so the pipeline can "
            "trust it wasn't left over from an incompatible older version.",
            [f"schema version {schema_version}",
             f"identity scheme: {identity_row['value'] if identity_row else 'unknown'}"],
        ))

        return cards, None
    finally:
        conn.close()


@dataclass
class SystemStatus:
    ollama_model: Optional[str] = None
    ollama_vram_gb: Optional[float] = None
    gpu_lines: list[str] = field(default_factory=list)
    graph_viz_neo4j: str = "not running"
    graph_viz_viewer: str = "not running"


def _http_get_json(url: str, timeout: float = 3.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read())
    except Exception:
        return None


def get_system_status() -> SystemStatus:
    status = SystemStatus()

    ps = _http_get_json(f"{OLLAMA_URL}/api/ps")
    if ps and ps.get("models"):
        m = ps["models"][0]
        status.ollama_model = m.get("name")
        status.ollama_vram_gb = m.get("size_vram", 0) / 1_073_741_824

    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0:
            status.gpu_lines = [ln.strip() for ln in out.stdout.strip().splitlines() if ln.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        out = subprocess.run(
            ["docker", "ps", "--filter", "name=paper-pipeline-neo4j", "--format", "{{.Status}}"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0 and out.stdout.strip():
            status.graph_viz_neo4j = out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        with socket.create_connection(("127.0.0.1", GRAPH_VIZ_PORT), timeout=1):
            status.graph_viz_viewer = f"http://localhost:{GRAPH_VIZ_PORT}/"
    except OSError:
        pass

    return status


# ══════════════════════════════════════════════════════════════════════════
# Render layer — pure, produces list[str] lines, no curses
# ══════════════════════════════════════════════════════════════════════════

_STATUS_ICON = {"complete": "✅", "partial": "⚠️ ", "not started": "⬜"}


def render_batch_view(progress: BatchProgress) -> list[str]:
    lines = [f"Batch Progress — {progress.papers_dir}", ""]
    if progress.error:
        lines.append(f"  ! {progress.error}")
        return lines
    lines.append(
        f"  {progress.complete} complete / {progress.partial} partial / "
        f"{progress.not_started} not started  (of {progress.total} total)"
    )
    lines.append("")
    for row in progress.rows:
        marker = "▶ " if row.in_progress else "  "
        icon = _STATUS_ICON[row.status]
        detail = f"  (missing: {', '.join(row.missing)})" if row.missing else ""
        lines.append(f"{marker}{icon} {row.name}{detail}")
    return lines


def render_db_entities_view(cards: list[EntityCard], error: Optional[str]) -> list[str]:
    lines = ["Database Entities", ""]
    if error:
        lines.append(f"  ! {error}")
        return lines
    for card in cards:
        header = f"── {card.table} "
        lines.append(header + "─" * max(0, 60 - len(header)))
        for chunk in textwrap.wrap(card.description, 76) or [""]:
            lines.append(f"  {chunk}")
        lines.append("  " + " · ".join(card.stats))
        lines.append("")
    return lines


def render_system_view(status: SystemStatus) -> list[str]:
    lines = ["System Status", ""]
    model_line = f"  Ollama model : {status.ollama_model or '(none loaded)'}"
    if status.ollama_vram_gb:
        model_line += f"  ({status.ollama_vram_gb:.1f} GB VRAM)"
    lines.append(model_line)
    lines.append("  GPU          :")
    if status.gpu_lines:
        for gpu_line in status.gpu_lines:
            lines.append(f"    {gpu_line}")
    else:
        lines.append("    nvidia-smi not available")
    lines.append("")
    lines.append(f"  Graph viz (neo4j)  : {status.graph_viz_neo4j}")
    lines.append(f"  Graph viz (viewer) : {status.graph_viz_viewer}")
    return lines


# ══════════════════════════════════════════════════════════════════════════
# Curses driver — thin, not unit tested
# ══════════════════════════════════════════════════════════════════════════

_VIEW_KEYS = {"1": "batch", "2": "entities", "3": "system"}


def _compute_view(view: str, db_path: Path, papers_dir: Path) -> list[str]:
    if view == "batch":
        return render_batch_view(get_batch_progress(db_path, papers_dir))
    if view == "entities":
        cards, error = get_db_entity_report(db_path)
        return render_db_entities_view(cards, error)
    return render_system_view(get_system_status())


def _run(stdscr, db_path: Path, papers_dir: Path, refresh_interval: float) -> None:
    import curses  # local import: only the curses driver needs the module

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(200)

    view = "batch"
    last_refresh = 0.0
    lines: list[str] = []

    while True:
        now = time.time()
        if now - last_refresh >= refresh_interval:
            lines = _compute_view(view, db_path, papers_dir)
            last_refresh = now

        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()
        header = (
            " paper-pipeline-tui | [1] Batch  [2] DB Entities  [3] System "
            "| [r]efresh [q]uit "
        )
        stdscr.addnstr(0, 0, header.ljust(max_x), max_x, curses.A_REVERSE)
        for i, line in enumerate(lines[: max_y - 2], start=2):
            try:
                stdscr.addnstr(i, 0, line, max_x)
            except curses.error:
                pass  # last line/column edge case -- not worth crashing over
        stdscr.refresh()

        try:
            key = stdscr.getkey()
        except curses.error:
            continue

        if key == "q":
            return
        if key in _VIEW_KEYS:
            view = _VIEW_KEYS[key]
            last_refresh = 0.0
        elif key == "r":
            last_refresh = 0.0


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="paper-pipeline-tui",
        description=(
            "Read-only dashboard for paper-pipeline: batch progress for a "
            "chosen folder, what the database tables mean and hold, and "
            "system/graph-viz status. Never writes to the database or "
            "starts/stops/reprocesses anything -- use `paper-pipeline` "
            "for that."
        ),
    )
    ap.add_argument(
        "papers_dir", nargs="?", default=None,
        help=f"Directory containing PDF papers [default: {config.DEFAULT_PAPERS_DIR}]",
    )
    ap.add_argument("--db-path", default=None, metavar="PATH",
                     help=f"SQLite database path [default: {config.DEFAULT_DB_PATH}]")
    ap.add_argument("--refresh-interval", type=float, default=5.0, metavar="SECONDS")
    ap.add_argument(
        "--once", action="store_true",
        help="Print all three views as plain text and exit (no curses)",
    )
    args = ap.parse_args()

    papers_dir = Path(
        os.path.expandvars(args.papers_dir or str(config.DEFAULT_PAPERS_DIR))
    ).expanduser()
    db_path = store.resolve_db_path(args.db_path)

    if args.once:
        for view in ("batch", "entities", "system"):
            print("=" * 70)
            print("\n".join(_compute_view(view, db_path, papers_dir)))
        return

    import curses
    try:
        curses.wrapper(_run, db_path, papers_dir, args.refresh_interval)
    except curses.error as exc:
        sys.exit(
            f"ERROR: could not start the interactive dashboard ({exc}).\n"
            "This usually means stdout/stdin isn't a real terminal (e.g. "
            "piped output, or run from a script). Run it directly in a "
            "terminal, or use --once for a plain-text snapshot instead."
        )


if __name__ == "__main__":
    main()
