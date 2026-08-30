"""
migrations.py — Transactional schema evolution using PRAGMA user_version.

Each migration is a (version, label, statements) entry.  Statements are
individual SQL strings executed one at a time inside a single BEGIN IMMEDIATE
transaction — never via executescript(), which implicitly commits any open
transaction and breaks atomicity.

Design constraints:
  - Migrations must be additive wherever possible.
  - PRAGMA user_version persists in the database header.
  - A migration that fails rolls back and leaves the prior version usable.
  - Concurrent startup is safe: the version is read inside BEGIN IMMEDIATE,
    so only one connection can advance the schema at a time.
  - Databases newer than the code's latest migration are rejected.
  - Non-empty databases with truncated paper_hash identities (pre-v0.4.2)
    are rejected before any mutation occurs.
"""

from __future__ import annotations

import sqlite3
from typing import NamedTuple


class Migration(NamedTuple):
    version: int
    label: str
    statements: list[str]


class LegacyDatabaseError(RuntimeError):
    """Raised when a database contains pre-v0.4.2 truncated hash identities."""


MIGRATIONS: list[Migration] = [
    Migration(1, "baseline schema", [
        """CREATE TABLE IF NOT EXISTS papers (
            paper_hash          TEXT PRIMARY KEY,
            paper_name          TEXT NOT NULL,
            pdf_path            TEXT NOT NULL,
            page_count          INTEGER,
            chunk_strategy      TEXT,
            model_used          TEXT,
            code_model          TEXT,
            processed_at        TEXT,
            sections_completed  TEXT NOT NULL DEFAULT '[]',
            summary_md          TEXT,
            symbolic_logic_md   TEXT,
            cpp_examples_md     TEXT,
            extras_md           TEXT,
            diagrams_raw_output TEXT,
            source_corpus       TEXT,
            created_at          TEXT NOT NULL DEFAULT '',
            updated_at          TEXT NOT NULL DEFAULT ''
        )""",
        """CREATE TABLE IF NOT EXISTS diagrams (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_hash  TEXT NOT NULL REFERENCES papers(paper_hash) ON DELETE CASCADE,
            idx         INTEGER NOT NULL,
            title       TEXT NOT NULL,
            dot_src     TEXT NOT NULL,
            svg_content TEXT,
            UNIQUE(paper_hash, idx)
        )""",
        """CREATE TABLE IF NOT EXISTS ocr_cache (
            paper_hash  TEXT NOT NULL,
            page_idx    INTEGER NOT NULL,
            text        TEXT NOT NULL,
            PRIMARY KEY (paper_hash, page_idx)
        )""",
        """CREATE TABLE IF NOT EXISTS processing_locks (
            pdf_path    TEXT PRIMARY KEY,
            claimed_at  REAL NOT NULL,
            claimed_by  TEXT NOT NULL
        )""",
        "CREATE INDEX IF NOT EXISTS idx_papers_pdf_path ON papers(pdf_path)",
        "CREATE INDEX IF NOT EXISTS idx_papers_name     ON papers(paper_name)",
        "CREATE INDEX IF NOT EXISTS idx_diagrams_hash   ON diagrams(paper_hash)",
    ]),

    Migration(2, "fenced leases", [
        """CREATE TABLE IF NOT EXISTS processing_leases (
            resource_key  TEXT PRIMARY KEY,
            owner_id      TEXT NOT NULL,
            generation    INTEGER NOT NULL DEFAULT 1,
            renewed_at    REAL NOT NULL,
            active        INTEGER NOT NULL DEFAULT 1
        )""",
        """INSERT INTO processing_leases
            (resource_key, owner_id, generation, renewed_at, active)
        SELECT pdf_path, claimed_by, 1, claimed_at, 1
        FROM processing_locks
        WHERE claimed_by IS NOT NULL""",
        """INSERT INTO processing_leases
            (resource_key, owner_id, generation, renewed_at, active)
        SELECT pdf_path, 'legacy-null-owner', 1, claimed_at, 0
        FROM processing_locks
        WHERE claimed_by IS NULL""",
        "DROP TABLE IF EXISTS processing_locks",
    ]),

    Migration(3, "ocr profile cache", [
        "DROP TABLE IF EXISTS ocr_cache",
        """CREATE TABLE ocr_cache (
            paper_hash  TEXT NOT NULL,
            page_idx    INTEGER NOT NULL,
            ocr_profile TEXT NOT NULL,
            text        TEXT NOT NULL,
            PRIMARY KEY (paper_hash, page_idx, ocr_profile)
        )""",
    ]),

    # ── Version 4: identity scheme marker ────────────────────────────────
    # Marks the database as using full 64-character SHA-256 paper_hash
    # identities.  The check in _reject_legacy_identities runs BEFORE
    # this migration, so a non-empty database with truncated hashes is
    # rejected before any mutation occurs.
    Migration(4, "full sha256 identity scheme", [
        """CREATE TABLE IF NOT EXISTS schema_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )""",
        # Use ON CONFLICT to overwrite any preexisting wrong marker from a
        # tampered v3 database.  INSERT OR IGNORE would silently retain it.
        """INSERT INTO schema_meta (key, value) VALUES ('identity_scheme', 'sha256-full-v1')
        ON CONFLICT(key) DO UPDATE SET value = 'sha256-full-v1'""",
        # Write-time enforcement: reject any paper_hash that is not exactly
        # 64 lowercase hex characters.
        """CREATE TRIGGER IF NOT EXISTS enforce_paper_hash_insert
        BEFORE INSERT ON papers
        BEGIN
            SELECT RAISE(ABORT, 'paper_hash must be 64 lowercase hex chars')
            WHERE NEW.paper_hash IS NULL
               OR length(NEW.paper_hash) != 64
               OR NEW.paper_hash != lower(NEW.paper_hash)
               OR NEW.paper_hash GLOB '*[^0-9a-f]*';
        END""",
        """CREATE TRIGGER IF NOT EXISTS enforce_paper_hash_update
        BEFORE UPDATE OF paper_hash ON papers
        BEGIN
            SELECT RAISE(ABORT, 'paper_hash must be 64 lowercase hex chars')
            WHERE NEW.paper_hash IS NULL
               OR length(NEW.paper_hash) != 64
               OR NEW.paper_hash != lower(NEW.paper_hash)
               OR NEW.paper_hash GLOB '*[^0-9a-f]*';
        END""",
    ]),
]


LATEST_VERSION: int = MIGRATIONS[-1].version if MIGRATIONS else 0


def get_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def _reject_legacy_identities(conn: sqlite3.Connection) -> None:
    """Reject non-empty databases containing invalid paper_hash values.

    A valid paper_hash is exactly 64 lowercase hexadecimal characters.
    NULL, uppercase, truncated, or non-hex hashes are all rejected.
    """
    has_papers = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='papers'"
    ).fetchone()
    if not has_papers:
        return

    legacy = conn.execute(
        "SELECT paper_hash FROM papers "
        "WHERE paper_hash IS NULL "
        "   OR length(paper_hash) != 64 "
        "   OR paper_hash != lower(paper_hash) "
        "   OR paper_hash GLOB '*[^0-9a-f]*' "
        "LIMIT 1"
    ).fetchone()
    if legacy is not None:
        display = repr(legacy[0]) if legacy[0] is not None else "NULL"
        raise LegacyDatabaseError(
            f"This database contains invalid paper identities "
            f"(found: {display}).  v0.5.0+ requires full 64-character "
            f"lowercase SHA-256 hashes.\n\n"
            f"Options:\n"
            f"  1. Use a new database path (recommended):\n"
            f"       --db-path /path/to/new/papers-v2.db\n"
            f"  2. Move or rename the old database and reprocess all papers.\n"
        )


def _validate_identity_scheme(conn: sqlite3.Connection) -> None:
    """Verify the identity_scheme marker if the schema_meta table exists.

    For version >= 4 databases, the marker is mandatory.  For earlier
    versions, a schema_meta table should not exist at all — if it does
    and contains a wrong marker, that indicates tampering.
    """
    version = get_version(conn)

    has_meta = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_meta'"
    ).fetchone()

    if version >= 4:
        # Marker is mandatory at version 4+
        if not has_meta:
            raise LegacyDatabaseError(
                "Database is at schema version >= 4 but missing the schema_meta table. "
                "This database may be corrupt or was modified by an incompatible tool."
            )
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'identity_scheme'"
        ).fetchone()
        if row is None or row[0] != "sha256-full-v1":
            found = repr(row[0]) if row else "missing"
            raise LegacyDatabaseError(
                f"Database identity_scheme is {found}, expected 'sha256-full-v1'. "
                f"This database is not compatible with this version of the pipeline."
            )
    elif has_meta:
        # Pre-v4 database should not have schema_meta — if it does,
        # check that the marker is not wrong (it may be absent, which is fine).
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'identity_scheme'"
        ).fetchone()
        if row is not None and row[0] != "sha256-full-v1":
            raise LegacyDatabaseError(
                f"Database has a preexisting identity_scheme marker "
                f"({row[0]!r}) that is not 'sha256-full-v1'. "
                f"This database may have been modified by an incompatible tool."
            )


def apply_migrations(conn: sqlite3.Connection) -> int:
    """Apply all pending migrations and return the final schema version.

    Validates identity constraints both before and after migrations, so a
    v3 database with a preexisting wrong marker is caught on first connect
    (not deferred to the second).
    """
    current = get_version(conn)

    if current > LATEST_VERSION:
        raise RuntimeError(
            f"database is at schema version {current}, "
            f"but this code only supports up to {LATEST_VERSION}"
        )

    # Pre-migration validation — catches truncated hashes and wrong markers
    # on databases already at version >= 4.
    _reject_legacy_identities(conn)
    _validate_identity_scheme(conn)

    if current == LATEST_VERSION:
        return current

    for migration in MIGRATIONS:
        conn.execute("BEGIN IMMEDIATE")
        try:
            current = conn.execute("PRAGMA user_version").fetchone()[0]

            if current > LATEST_VERSION:
                conn.rollback()
                raise RuntimeError(
                    f"database is at schema version {current}, "
                    f"but this code only supports up to {LATEST_VERSION}"
                )

            if migration.version <= current:
                conn.rollback()
                continue

            if migration.version != current + 1:
                conn.rollback()
                raise RuntimeError(
                    f"migration gap: database is at version {current}, "
                    f"next migration is version {migration.version}"
                )

            # Migration 4 installs identity triggers.  Recheck under the
            # write lock so a concurrent legacy writer cannot insert a
            # truncated hash between the pre-migration check and trigger
            # installation.  If an invalid row appeared, rollback — the
            # database stays at version 3.
            if migration.version == 4:
                _reject_legacy_identities(conn)
                _validate_identity_scheme(conn)

            for sql in migration.statements:
                conn.execute(sql)

            conn.execute(f"PRAGMA user_version = {migration.version}")
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # Post-migration defense-in-depth — the transactional check inside
    # migration 4 is the primary guard; this catches anything else.
    _reject_legacy_identities(conn)
    _validate_identity_scheme(conn)

    return get_version(conn)
