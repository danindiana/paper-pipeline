"""
store.py — SQLite-backed storage with fenced lease writes.

Lease protocol:
  - generation advances only on acquisition or stale reclamation.
  - Heartbeats update renewed_at without changing generation.
  - Release marks the lease inactive — the row persists.
  - Every protected write runs inside BEGIN IMMEDIATE and verifies
    (resource_key, owner_id, generation, active=1, not expired)
    in the same transaction.  No check-to-write gap.

Write functions have two forms:
  - Public committing versions (e.g. write_section) for unfenced callers.
  - Internal non-committing versions (e.g. _write_section_sql) for use
    inside typed fenced operations.  These MUST NOT commit — transaction control
    belongs exclusively to the fenced layer.
"""

import json
import os
import socket
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from . import config
from .errors import LeaseLostError
from .migrations import apply_migrations

DEFAULT_DB_PATH = config.DEFAULT_DB_PATH
DB_PATH_ENV_VAR = config.DB_PATH_ENV_VAR
STALE_LOCK_SECONDS = config.LEASE_STALE_SECONDS

ALL_SECTIONS = {"summary", "logic", "cpp", "diagrams", "extras"}
_SECTION_COLUMNS = {
    "summary": "summary_md",
    "logic": "symbolic_logic_md",
    "cpp": "cpp_examples_md",
    "extras": "extras_md",
}


# ── data records ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PaperRecord:
    paper_hash: str
    paper_name: str
    pdf_path: str
    page_count: Optional[int]
    chunk_strategy: Optional[str]
    model_used: Optional[str]
    code_model: Optional[str]
    processed_at: Optional[str]
    sections_completed: List[str]
    summary_md: Optional[str]
    symbolic_logic_md: Optional[str]
    cpp_examples_md: Optional[str]
    extras_md: Optional[str]
    diagrams_raw_output: Optional[str]
    source_corpus: Optional[str]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class DiagramRecord:
    paper_hash: str
    idx: int
    title: str
    dot_src: str
    svg_content: Optional[str]


@dataclass(frozen=True)
class ClaimResult:
    claimed: bool
    age_seconds: Optional[float] = None
    reclaimed: bool = False
    owner_id: str = ""
    generation: int = 0


# ── connection ───────────────────────────────────────────────────────────────

def resolve_db_path(cli_value: Optional[str] = None) -> Path:
    if cli_value:
        return Path(os.path.expandvars(cli_value)).expanduser()
    env = os.environ.get(DB_PATH_ENV_VAR)
    if env:
        return Path(os.path.expandvars(env)).expanduser()
    return DEFAULT_DB_PATH


def connect(db_path: Path, *, run_migrations: bool = True) -> sqlite3.Connection:
    """Open a connection with WAL mode and busy_timeout.

    When ``run_migrations`` is True (the default), pending schema migrations
    are applied.  Pass False for connections that know the schema is already
    current (e.g. heartbeat threads after init_db has run).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    # busy_timeout MUST be set before journal_mode — WAL activation can
    # contend with other connections on a fresh database, and without
    # busy_timeout the PRAGMA will raise "database is locked" instead of
    # retrying.
    conn.execute("PRAGMA busy_timeout=30000")
    # WAL activation can still fail under extreme contention (e.g. library
    # use without init_db pre-initialization).  Retry lock/busy errors only.
    for attempt in range(3):
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            break
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                conn.close()
                raise
            if attempt == 2:
                conn.close()
                raise
            time.sleep(0.1 * (attempt + 1))
    try:
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        if run_migrations:
            apply_migrations(conn)
    except Exception:
        conn.close()
        raise
    return conn


def init_db(db_path: Path) -> None:
    """Ensure the database exists and is at the latest schema version.

    Call this once in main() before spawning workers, so that concurrent
    worker connections never race on migration writes.
    """
    conn = connect(db_path, run_migrations=True)
    conn.close()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _row_to_paper_record(row: sqlite3.Row) -> PaperRecord:
    return PaperRecord(
        paper_hash=row["paper_hash"],
        paper_name=row["paper_name"],
        pdf_path=row["pdf_path"],
        page_count=row["page_count"],
        chunk_strategy=row["chunk_strategy"],
        model_used=row["model_used"],
        code_model=row["code_model"],
        processed_at=row["processed_at"],
        sections_completed=json.loads(row["sections_completed"] or "[]"),
        summary_md=row["summary_md"],
        symbolic_logic_md=row["symbolic_logic_md"],
        cpp_examples_md=row["cpp_examples_md"],
        extras_md=row["extras_md"],
        diagrams_raw_output=row["diagrams_raw_output"],
        source_corpus=row["source_corpus"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# FENCED LEASES
# ══════════════════════════════════════════════════════════════════════════════


def make_owner_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"


def try_claim(
    conn: sqlite3.Connection,
    resource_key: str,
    claimed_by: Optional[str] = None,
    stale_after: float = STALE_LOCK_SECONDS,
) -> ClaimResult:
    """Atomically acquire or reclaim a fenced lease.

    The read and write are inside BEGIN IMMEDIATE so two concurrent first
    claimants cannot both see "no row" and race on INSERT.
    """
    if stale_after <= 0:
        raise ValueError("stale_after must be positive")
    owner_id = claimed_by or make_owner_id()
    now = time.time()

    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT owner_id, generation, renewed_at, active "
            "FROM processing_leases WHERE resource_key = ?",
            (resource_key,),
        ).fetchone()

        if row is None:
            conn.execute(
                "INSERT INTO processing_leases "
                "(resource_key, owner_id, generation, renewed_at, active) "
                "VALUES (?, ?, 1, ?, 1)",
                (resource_key, owner_id, now),
            )
            conn.commit()
            return ClaimResult(claimed=True, owner_id=owner_id, generation=1)

        if row["active"] and (now - row["renewed_at"]) < stale_after:
            conn.commit()
            return ClaimResult(
                claimed=False,
                age_seconds=now - row["renewed_at"],
                owner_id=owner_id,
            )

        # Inactive or stale — reclaim by incrementing generation.
        old_gen = row["generation"]
        new_gen = old_gen + 1
        conn.execute(
            "UPDATE processing_leases SET "
            "  owner_id = ?, generation = ?, renewed_at = ?, active = 1 "
            "WHERE resource_key = ? AND generation = ?",
            (owner_id, new_gen, now, resource_key, old_gen),
        )
        conn.commit()
        return ClaimResult(
            claimed=True,
            age_seconds=now - row["renewed_at"],
            reclaimed=bool(row["active"]),
            owner_id=owner_id,
            generation=new_gen,
        )
    except Exception:
        conn.rollback()
        raise


def refresh_claim(
    conn: sqlite3.Connection,
    resource_key: str,
    owner_id: str,
    generation: int,
    stale_after: float = STALE_LOCK_SECONDS,
) -> bool:
    """Renew a lease only if owner+generation match and the lease has not expired.

    An expired lease cannot be refreshed — reacquisition must advance generation.
    """
    now = time.time()
    cur = conn.execute(
        "UPDATE processing_leases SET renewed_at = ? "
        "WHERE resource_key = ? AND owner_id = ? AND generation = ? "
        "AND active = 1 AND (? - renewed_at) < ?",
        (now, resource_key, owner_id, generation, now, stale_after),
    )
    conn.commit()
    return cur.rowcount == 1


def claim_owned(
    conn: sqlite3.Connection,
    resource_key: str,
    owner_id: str,
    generation: int,
) -> bool:
    """Return whether owner+generation is the current active, non-expired lease."""
    now = time.time()
    row = conn.execute(
        "SELECT 1 FROM processing_leases "
        "WHERE resource_key = ? AND owner_id = ? AND generation = ? "
        "AND active = 1 AND (? - renewed_at) < ?",
        (resource_key, owner_id, generation, now, STALE_LOCK_SECONDS),
    ).fetchone()
    return row is not None


def release_claim(
    conn: sqlite3.Connection,
    resource_key: str,
    owner_id: str,
    generation: int,
) -> bool:
    """Mark a lease inactive only if owner+generation match."""
    cur = conn.execute(
        "UPDATE processing_leases SET active = 0 "
        "WHERE resource_key = ? AND owner_id = ? AND generation = ? AND active = 1",
        (resource_key, owner_id, generation),
    )
    conn.commit()
    return cur.rowcount == 1


def assert_fenced(
    conn: sqlite3.Connection,
    resource_key: str,
    owner_id: str,
    generation: int,
    stale_after: float = STALE_LOCK_SECONDS,
) -> None:
    """Verify lease ownership + freshness inside an open BEGIN IMMEDIATE.

    Raises LeaseLostError if the lease is not active, not owned by this
    owner+generation, or has expired.
    """
    now = time.time()
    row = conn.execute(
        "SELECT 1 FROM processing_leases "
        "WHERE resource_key = ? AND owner_id = ? AND generation = ? "
        "AND active = 1 AND (? - renewed_at) < ?",
        (resource_key, owner_id, generation, now, stale_after),
    ).fetchone()
    if row is None:
        raise LeaseLostError(
            f"fenced write rejected: lease {resource_key} "
            f"gen={generation} owner={owner_id}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# FENCED WRITE OPERATIONS — typed, trusted, no arbitrary callbacks
# ══════════════════════════════════════════════════════════════════════════════
#
# Each operation runs BEGIN IMMEDIATE → assert_fenced → trusted _sql → COMMIT.
# The
# transaction boundary is never exposed to caller code.


def _fenced_begin(conn, resource_key, owner_id, generation, stale_after=STALE_LOCK_SECONDS):
    """Acquire write lock and verify lease.  Caller MUST call _fenced_end."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        assert_fenced(conn, resource_key, owner_id, generation, stale_after)
    except Exception:
        conn.rollback()
        raise


def _fenced_end(conn):
    conn.commit()


def _fenced_abort(conn):
    if conn.in_transaction:
        conn.rollback()


def write_section_fenced(
    conn, resource_key, owner_id, generation,
    paper_hash, section, content,
    stale_after=STALE_LOCK_SECONDS,
):
    """Fenced: write section content + mark complete in one transaction."""
    _fenced_begin(conn, resource_key, owner_id, generation, stale_after)
    try:
        _write_section_sql(conn, paper_hash, section, content)
        _fenced_end(conn)
    except Exception:
        _fenced_abort(conn)
        raise


def replace_diagrams_fenced(
    conn, resource_key, owner_id, generation,
    paper_hash, diagrams,
    stale_after=STALE_LOCK_SECONDS,
):
    """Fenced: replace diagram rows + mark complete in one transaction."""
    _fenced_begin(conn, resource_key, owner_id, generation, stale_after)
    try:
        _replace_diagrams_sql(conn, paper_hash, diagrams)
        _mark_section_complete_sql(conn, paper_hash, "diagrams")
        _fenced_end(conn)
    except Exception:
        _fenced_abort(conn)
        raise


def write_diagrams_raw_fenced(
    conn, resource_key, owner_id, generation,
    paper_hash, raw_text,
    stale_after=STALE_LOCK_SECONDS,
):
    """Fenced: save unparseable diagram output (section stays incomplete)."""
    _fenced_begin(conn, resource_key, owner_id, generation, stale_after)
    try:
        _write_diagrams_raw_sql(conn, paper_hash, raw_text)
        _fenced_end(conn)
    except Exception:
        _fenced_abort(conn)
        raise


def upsert_meta_and_corpus_fenced(
    conn, resource_key, owner_id, generation,
    paper_hash, paper_name, pdf_path, page_count,
    chunk_strategy, model_used, code_model, corpus_json,
    stale_after=STALE_LOCK_SECONDS,
):
    """Fenced: upsert paper metadata + evidence corpus in one transaction."""
    _fenced_begin(conn, resource_key, owner_id, generation, stale_after)
    try:
        _upsert_paper_meta_sql(
            conn, paper_hash, paper_name, pdf_path, page_count,
            chunk_strategy, model_used, code_model,
        )
        _write_source_corpus_sql(conn, paper_hash, corpus_json)
        _fenced_end(conn)
    except Exception:
        _fenced_abort(conn)
        raise


def clear_section_fenced(
    conn, resource_key, owner_id, generation,
    paper_hash, section,
    stale_after=STALE_LOCK_SECONDS,
):
    """Fenced: clear a section for reprocessing."""
    _fenced_begin(conn, resource_key, owner_id, generation, stale_after)
    try:
        _clear_section_sql(conn, paper_hash, section)
        _fenced_end(conn)
    except Exception:
        _fenced_abort(conn)
        raise


class LeaseHeartbeat:
    """Renew a processing lease on a dedicated SQLite connection."""

    def __init__(
        self,
        db_path: Path,
        resource_key: str,
        owner_id: str,
        generation: int,
        *,
        interval: float = config.LEASE_HEARTBEAT_SECONDS,
        stale_after: float = STALE_LOCK_SECONDS,
    ):
        if interval <= 0 or stale_after <= 0:
            raise ValueError("lease heartbeat timings must be positive")
        if interval >= stale_after / 2:
            raise ValueError("heartbeat interval must be less than half the stale timeout")
        self.db_path = Path(db_path)
        self.resource_key = resource_key
        self.owner_id = owner_id
        self.generation = generation
        self.interval = interval
        self.stale_after = stale_after
        self._stop = Event()
        self._state_lock = Lock()
        self._last_success = time.monotonic()
        self._last_error: Optional[str] = None
        self._lost_reason: Optional[str] = None
        self._thread = Thread(
            target=self._run,
            name=f"lease-heartbeat-{owner_id.rsplit(':', 1)[-1][:8]}",
            daemon=True,
        )
        self._started = False

    def start(self) -> "LeaseHeartbeat":
        if self._started:
            raise RuntimeError("lease heartbeat already started")
        self._started = True
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._started:
            self._thread.join(timeout=max(2.0, self.interval + 1.0))
            if self._thread.is_alive():
                raise RuntimeError("lease heartbeat thread did not stop")

    def assert_healthy(self) -> None:
        with self._state_lock:
            lost_reason = self._lost_reason
            last_error = self._last_error
            elapsed = time.monotonic() - self._last_success
        if lost_reason:
            raise LeaseLostError(lost_reason)
        if last_error and elapsed >= self.stale_after / 2:
            raise LeaseLostError(
                f"lease renewal unhealthy for {elapsed:.1f}s: {last_error}"
            )

    def _run(self) -> None:
        conn: Optional[sqlite3.Connection] = None
        try:
            while not self._stop.wait(self.interval):
                try:
                    if conn is None:
                        conn = connect(self.db_path, run_migrations=False)
                    if not refresh_claim(
                        conn, self.resource_key, self.owner_id,
                        self.generation, self.stale_after,
                    ):
                        with self._state_lock:
                            self._lost_reason = (
                                f"processing lease lost for {self.resource_key}"
                            )
                        return
                    with self._state_lock:
                        self._last_success = time.monotonic()
                        self._last_error = None
                except Exception as exc:
                    if conn is not None:
                        conn.close()
                        conn = None
                    with self._state_lock:
                        self._last_error = f"{type(exc).__name__}: {exc}"
        finally:
            if conn is not None:
                conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# NON-COMMITTING WRITE INTERNALS (for use inside typed fenced operations only)
# ══════════════════════════════════════════════════════════════════════════════

def _upsert_paper_meta_sql(
    conn: sqlite3.Connection,
    paper_hash: str, paper_name: str, pdf_path: str,
    page_count: int, chunk_strategy: str,
    model_used: str, code_model: str,
    source_corpus: Optional[str] = None,
    processed_at: Optional[str] = None,
) -> None:
    now = _now_iso()
    processed_at = processed_at or now
    conn.execute(
        """INSERT INTO papers (
            paper_hash, paper_name, pdf_path, page_count, chunk_strategy,
            model_used, code_model, processed_at, sections_completed,
            source_corpus, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '[]', ?, ?, ?)
        ON CONFLICT(paper_hash) DO UPDATE SET
            paper_name=excluded.paper_name, pdf_path=excluded.pdf_path,
            page_count=excluded.page_count, chunk_strategy=excluded.chunk_strategy,
            model_used=excluded.model_used, code_model=excluded.code_model,
            processed_at=excluded.processed_at,
            source_corpus=COALESCE(papers.source_corpus, excluded.source_corpus),
            updated_at=excluded.updated_at""",
        (paper_hash, paper_name, pdf_path, page_count, chunk_strategy,
         model_used, code_model, processed_at, source_corpus, now, now),
    )


def _mark_section_complete_sql(conn: sqlite3.Connection, paper_hash: str, section: str) -> None:
    row = conn.execute(
        "SELECT sections_completed FROM papers WHERE paper_hash = ?", (paper_hash,)
    ).fetchone()
    if row is None:
        return
    completed = json.loads(row["sections_completed"] or "[]")
    if section not in completed:
        completed.append(section)
        conn.execute(
            "UPDATE papers SET sections_completed = ?, updated_at = ? WHERE paper_hash = ?",
            (json.dumps(completed), _now_iso(), paper_hash),
        )


def _write_section_sql(conn: sqlite3.Connection, paper_hash: str, section: str, content: str) -> None:
    if section not in _SECTION_COLUMNS:
        raise ValueError(f"Unknown markdown section: {section!r}")
    column = _SECTION_COLUMNS[section]
    conn.execute(
        f"UPDATE papers SET {column} = ?, updated_at = ? WHERE paper_hash = ?",
        (content, _now_iso(), paper_hash),
    )
    _mark_section_complete_sql(conn, paper_hash, section)


def _replace_diagrams_sql(
    conn: sqlite3.Connection, paper_hash: str,
    diagrams: List[Tuple[str, str, Optional[str]]],
) -> None:
    conn.execute("DELETE FROM diagrams WHERE paper_hash = ?", (paper_hash,))
    conn.executemany(
        "INSERT INTO diagrams (paper_hash, idx, title, dot_src, svg_content) "
        "VALUES (?, ?, ?, ?, ?)",
        [(paper_hash, idx, t, d, s) for idx, (t, d, s) in enumerate(diagrams, 1)],
    )
    conn.execute(
        "UPDATE papers SET diagrams_raw_output = NULL, updated_at = ? WHERE paper_hash = ?",
        (_now_iso(), paper_hash),
    )


def _write_diagrams_raw_sql(conn: sqlite3.Connection, paper_hash: str, raw_text: str) -> None:
    conn.execute(
        "UPDATE papers SET diagrams_raw_output = ?, updated_at = ? WHERE paper_hash = ?",
        (raw_text, _now_iso(), paper_hash),
    )


def _write_source_corpus_sql(conn: sqlite3.Connection, paper_hash: str, corpus: str) -> None:
    conn.execute(
        "UPDATE papers SET source_corpus = ?, updated_at = ? WHERE paper_hash = ?",
        (corpus, _now_iso(), paper_hash),
    )


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC COMMITTING WRAPPERS (for unfenced callers and backward compat)
# ══════════════════════════════════════════════════════════════════════════════

def upsert_paper_meta(conn, paper_hash, paper_name, pdf_path, page_count,
                      chunk_strategy, model_used, code_model,
                      source_corpus=None, processed_at=None):
    _upsert_paper_meta_sql(conn, paper_hash, paper_name, pdf_path, page_count,
                           chunk_strategy, model_used, code_model,
                           source_corpus, processed_at)
    conn.commit()

def mark_section_complete(conn, paper_hash, section):
    _mark_section_complete_sql(conn, paper_hash, section)
    conn.commit()

def write_section(conn, paper_hash, section, content):
    _write_section_sql(conn, paper_hash, section, content)
    conn.commit()

def replace_diagrams(conn, paper_hash, diagrams):
    _replace_diagrams_sql(conn, paper_hash, diagrams)
    conn.commit()

def write_diagrams_raw_output(conn, paper_hash, raw_text):
    _write_diagrams_raw_sql(conn, paper_hash, raw_text)
    conn.commit()

def write_source_corpus(conn, paper_hash, source_corpus):
    _write_source_corpus_sql(conn, paper_hash, source_corpus)
    conn.commit()


def _clear_section_sql(conn: sqlite3.Connection, paper_hash: str, section: str) -> None:
    """Non-committing clear — for use inside typed fenced operations."""
    row = conn.execute(
        "SELECT sections_completed FROM papers WHERE paper_hash = ?", (paper_hash,)
    ).fetchone()
    if row is None:
        return
    now = _now_iso()
    if section == "all":
        conn.execute(
            """UPDATE papers SET summary_md=NULL, symbolic_logic_md=NULL,
            cpp_examples_md=NULL, extras_md=NULL, diagrams_raw_output=NULL,
            sections_completed='[]', updated_at=? WHERE paper_hash=?""",
            (now, paper_hash),
        )
        conn.execute("DELETE FROM diagrams WHERE paper_hash = ?", (paper_hash,))
        return
    completed = json.loads(row["sections_completed"] or "[]")
    if section in completed:
        completed.remove(section)
    if section == "diagrams":
        conn.execute("DELETE FROM diagrams WHERE paper_hash = ?", (paper_hash,))
        conn.execute(
            "UPDATE papers SET diagrams_raw_output=NULL, sections_completed=?, "
            "updated_at=? WHERE paper_hash=?",
            (json.dumps(completed), now, paper_hash),
        )
    elif section in _SECTION_COLUMNS:
        col = _SECTION_COLUMNS[section]
        conn.execute(
            f"UPDATE papers SET {col}=NULL, sections_completed=?, updated_at=? "
            f"WHERE paper_hash=?",
            (json.dumps(completed), now, paper_hash),
        )
    else:
        raise ValueError(f"Unknown section: {section!r}")


def clear_section(conn: sqlite3.Connection, paper_hash: str, section: str) -> None:
    """Committing wrapper for unfenced callers."""
    _clear_section_sql(conn, paper_hash, section)
    conn.commit()


# ── read ─────────────────────────────────────────────────────────────────────

def load_paper(conn, paper_hash):
    row = conn.execute("SELECT * FROM papers WHERE paper_hash=?", (paper_hash,)).fetchone()
    return _row_to_paper_record(row) if row else None

def load_paper_by_pdf_path(conn, pdf_path):
    row = conn.execute(
        "SELECT * FROM papers WHERE pdf_path=? ORDER BY updated_at DESC LIMIT 1",
        (pdf_path,),
    ).fetchone()
    return _row_to_paper_record(row) if row else None

# ── OCR cache ────────────────────────────────────────────────────────────────

# ── OCR cache (explicitly unprotected — idempotent, content-addressed) ────
# OCR page text is cached keyed by (paper_hash, page_idx, ocr_profile).
# The profile string captures DPI, language, and backend versions so that
# configuration changes naturally invalidate stale entries.  These writes
# happen during extraction, before the pipeline's fenced write phase.

def get_cached_ocr_page(conn, paper_hash, page_idx, ocr_profile):
    row = conn.execute(
        "SELECT text FROM ocr_cache WHERE paper_hash=? AND page_idx=? AND ocr_profile=?",
        (paper_hash, page_idx, ocr_profile),
    ).fetchone()
    return row["text"] if row else None

def put_cached_ocr_page(conn, paper_hash, page_idx, ocr_profile, text):
    conn.execute(
        "INSERT INTO ocr_cache (paper_hash, page_idx, ocr_profile, text) "
        "VALUES (?,?,?,?) ON CONFLICT(paper_hash, page_idx, ocr_profile) DO NOTHING",
        (paper_hash, page_idx, ocr_profile, text),
    )
    conn.commit()

# ── bulk read ────────────────────────────────────────────────────────────────

def iter_papers_for_sync(conn, since_hash_map=None):
    for row in conn.execute("SELECT * FROM papers ORDER BY paper_hash"):
        rec = _row_to_paper_record(row)
        if since_hash_map and since_hash_map.get(rec.paper_hash) == rec.paper_hash:
            continue
        yield rec

def load_diagrams(conn, paper_hash):
    return [
        DiagramRecord(r["paper_hash"], r["idx"], r["title"], r["dot_src"], r["svg_content"])
        for r in conn.execute("SELECT * FROM diagrams WHERE paper_hash=? ORDER BY idx", (paper_hash,))
    ]

def get_diagram_svg(conn, paper_hash, idx):
    row = conn.execute(
        "SELECT title, svg_content FROM diagrams WHERE paper_hash=? AND idx=?",
        (paper_hash, idx),
    ).fetchone()
    if row is None or row["svg_content"] is None:
        return None
    return row["title"], row["svg_content"]
