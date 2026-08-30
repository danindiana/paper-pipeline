import concurrent.futures
import sqlite3
import tempfile
import unittest
from pathlib import Path

from paper_pipeline import store
from paper_pipeline.errors import LeaseLostError
from paper_pipeline.migrations import MIGRATIONS, LegacyDatabaseError, apply_migrations, get_version


class MigrationTests(unittest.TestCase):
    def test_fresh_database_reaches_latest_version(self):
        with tempfile.TemporaryDirectory() as d:
            conn = store.connect(Path(d) / "papers.db")
            self.assertEqual(get_version(conn), MIGRATIONS[-1].version)
            conn.close()

    def test_existing_v0_database_migrates_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "papers.db"
            conn = sqlite3.connect(str(db))
            conn.executescript("""
                CREATE TABLE papers (
                    paper_hash TEXT PRIMARY KEY, paper_name TEXT NOT NULL,
                    pdf_path TEXT NOT NULL, page_count INTEGER,
                    chunk_strategy TEXT, model_used TEXT, code_model TEXT,
                    processed_at TEXT, sections_completed TEXT NOT NULL DEFAULT '[]',
                    summary_md TEXT, symbolic_logic_md TEXT, cpp_examples_md TEXT,
                    extras_md TEXT, diagrams_raw_output TEXT, source_corpus TEXT,
                    created_at TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE processing_locks (
                    pdf_path TEXT PRIMARY KEY, claimed_at REAL NOT NULL,
                    claimed_by TEXT NOT NULL
                );
                INSERT INTO processing_locks VALUES ('test.pdf', 1000000.0, 'old-owner');
            """)
            conn.commit()
            conn.close()
            # Empty papers table — no truncated hashes, should migrate fine
            conn = store.connect(db)
            self.assertEqual(get_version(conn), MIGRATIONS[-1].version)
            conn.close()

    def test_v3_database_with_truncated_hash_is_rejected(self):
        """A version-3 database containing a 16-char hash must be rejected."""
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "papers.db"
            conn = sqlite3.connect(str(db))
            conn.executescript("""
                CREATE TABLE papers (
                    paper_hash TEXT PRIMARY KEY, paper_name TEXT NOT NULL,
                    pdf_path TEXT NOT NULL, sections_completed TEXT NOT NULL DEFAULT '[]',
                    page_count INTEGER, chunk_strategy TEXT, model_used TEXT,
                    code_model TEXT, processed_at TEXT, summary_md TEXT,
                    symbolic_logic_md TEXT, cpp_examples_md TEXT, extras_md TEXT,
                    diagrams_raw_output TEXT, source_corpus TEXT,
                    created_at TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE processing_leases (
                    resource_key TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
                    generation INTEGER NOT NULL DEFAULT 1,
                    renewed_at REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE ocr_cache (
                    paper_hash TEXT NOT NULL, page_idx INTEGER NOT NULL,
                    ocr_profile TEXT NOT NULL, text TEXT NOT NULL,
                    PRIMARY KEY (paper_hash, page_idx, ocr_profile)
                );
                INSERT INTO papers (paper_hash, paper_name, pdf_path,
                    sections_completed, created_at, updated_at)
                VALUES ('abcdef0123456789', 'old.pdf', '/old.pdf', '[]', '', '');
            """)
            conn.execute("PRAGMA user_version = 3")
            conn.commit()
            conn.close()

            with self.assertRaises(LegacyDatabaseError) as ctx:
                store.connect(db)
            self.assertIn("abcdef0123456789", str(ctx.exception))

    def test_v3_database_with_full_hashes_accepted(self):
        """A version-3 database with only 64-char hashes upgrades to v4."""
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "papers.db"
            full_hash = "a" * 64
            conn = sqlite3.connect(str(db))
            conn.executescript(f"""
                CREATE TABLE papers (
                    paper_hash TEXT PRIMARY KEY, paper_name TEXT NOT NULL,
                    pdf_path TEXT NOT NULL, sections_completed TEXT NOT NULL DEFAULT '[]',
                    page_count INTEGER, chunk_strategy TEXT, model_used TEXT,
                    code_model TEXT, processed_at TEXT, summary_md TEXT,
                    symbolic_logic_md TEXT, cpp_examples_md TEXT, extras_md TEXT,
                    diagrams_raw_output TEXT, source_corpus TEXT,
                    created_at TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE processing_leases (
                    resource_key TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
                    generation INTEGER NOT NULL DEFAULT 1,
                    renewed_at REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE ocr_cache (
                    paper_hash TEXT NOT NULL, page_idx INTEGER NOT NULL,
                    ocr_profile TEXT NOT NULL, text TEXT NOT NULL,
                    PRIMARY KEY (paper_hash, page_idx, ocr_profile)
                );
                INSERT INTO papers (paper_hash, paper_name, pdf_path,
                    sections_completed, created_at, updated_at)
                VALUES ('{full_hash}', 'new.pdf', '/new.pdf', '[]', '', '');
            """)
            conn.execute("PRAGMA user_version = 3")
            conn.commit()
            conn.close()

            conn = store.connect(db)
            self.assertEqual(get_version(conn), MIGRATIONS[-1].version)
            # Verify the identity scheme marker was written
            row = conn.execute(
                "SELECT value FROM schema_meta WHERE key='identity_scheme'"
            ).fetchone()
            self.assertEqual(row["value"], "sha256-full-v1")
            conn.close()

    def test_empty_legacy_database_initializes(self):
        """An empty database (no papers rows) should initialize cleanly."""
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "papers.db"
            conn = sqlite3.connect(str(db))
            conn.executescript("""
                CREATE TABLE papers (
                    paper_hash TEXT PRIMARY KEY, paper_name TEXT NOT NULL,
                    pdf_path TEXT NOT NULL, sections_completed TEXT NOT NULL DEFAULT '[]',
                    page_count INTEGER, chunk_strategy TEXT, model_used TEXT,
                    code_model TEXT, processed_at TEXT, summary_md TEXT,
                    symbolic_logic_md TEXT, cpp_examples_md TEXT, extras_md TEXT,
                    diagrams_raw_output TEXT, source_corpus TEXT,
                    created_at TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL DEFAULT ''
                );
            """)
            conn.commit()
            conn.close()

            conn = store.connect(db)
            self.assertEqual(get_version(conn), MIGRATIONS[-1].version)
            conn.close()

    def test_null_claimed_by_migrates_as_inactive(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "papers.db"
            conn = sqlite3.connect(str(db))
            conn.executescript("""
                CREATE TABLE papers (
                    paper_hash TEXT PRIMARY KEY, paper_name TEXT NOT NULL,
                    pdf_path TEXT NOT NULL, sections_completed TEXT NOT NULL DEFAULT '[]',
                    page_count INTEGER, chunk_strategy TEXT, model_used TEXT,
                    code_model TEXT, processed_at TEXT, summary_md TEXT,
                    symbolic_logic_md TEXT, cpp_examples_md TEXT, extras_md TEXT,
                    diagrams_raw_output TEXT, source_corpus TEXT,
                    created_at TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE processing_locks (
                    pdf_path TEXT PRIMARY KEY, claimed_at REAL NOT NULL,
                    claimed_by TEXT
                );
                INSERT INTO processing_locks VALUES ('null.pdf', 1000000.0, NULL);
                INSERT INTO processing_locks VALUES ('ok.pdf', 1000000.0, 'real-owner');
            """)
            conn.commit()
            conn.close()

            conn = store.connect(db)
            null_row = conn.execute(
                "SELECT * FROM processing_leases WHERE resource_key='null.pdf'"
            ).fetchone()
            self.assertIsNotNone(null_row)
            self.assertEqual(null_row["active"], 0)
            ok_row = conn.execute(
                "SELECT * FROM processing_leases WHERE resource_key='ok.pdf'"
            ).fetchone()
            self.assertEqual(ok_row["active"], 1)
            conn.close()

    def test_idempotent_on_current_version(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "papers.db"
            c1 = store.connect(db)
            v1 = get_version(c1)
            c1.close()
            c2 = store.connect(db)
            v2 = get_version(c2)
            c2.close()
            self.assertEqual(v1, v2)

    def test_rejects_future_version(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "papers.db"
            conn = store.connect(db)
            conn.execute(f"PRAGMA user_version = 999")
            conn.close()
            with self.assertRaises(RuntimeError) as ctx:
                store.connect(db)
            self.assertIn("999", str(ctx.exception))

    def test_uppercase_hash_rejected(self):
        """A 64-char uppercase hash is rejected — it would create duplicates."""
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "papers.db"
            conn = sqlite3.connect(str(db))
            conn.executescript("""
                CREATE TABLE papers (
                    paper_hash TEXT PRIMARY KEY, paper_name TEXT NOT NULL,
                    pdf_path TEXT NOT NULL, sections_completed TEXT NOT NULL DEFAULT '[]',
                    page_count INTEGER, chunk_strategy TEXT, model_used TEXT,
                    code_model TEXT, processed_at TEXT, summary_md TEXT,
                    symbolic_logic_md TEXT, cpp_examples_md TEXT, extras_md TEXT,
                    diagrams_raw_output TEXT, source_corpus TEXT,
                    created_at TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL DEFAULT ''
                );
                INSERT INTO papers (paper_hash, paper_name, pdf_path,
                    sections_completed, created_at, updated_at)
                VALUES ('ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789',
                        'upper.pdf', '/upper.pdf', '[]', '', '');
            """)
            conn.execute("PRAGMA user_version = 3")
            conn.commit()
            conn.close()
            with self.assertRaises(LegacyDatabaseError):
                store.connect(db)

    def test_null_hash_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "papers.db"
            conn = sqlite3.connect(str(db))
            conn.executescript("""
                CREATE TABLE papers (
                    paper_hash TEXT PRIMARY KEY, paper_name TEXT NOT NULL,
                    pdf_path TEXT NOT NULL, sections_completed TEXT NOT NULL DEFAULT '[]',
                    page_count INTEGER, chunk_strategy TEXT, model_used TEXT,
                    code_model TEXT, processed_at TEXT, summary_md TEXT,
                    symbolic_logic_md TEXT, cpp_examples_md TEXT, extras_md TEXT,
                    diagrams_raw_output TEXT, source_corpus TEXT,
                    created_at TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL DEFAULT ''
                );
                INSERT INTO papers (paper_hash, paper_name, pdf_path,
                    sections_completed, created_at, updated_at)
                VALUES (NULL, 'null.pdf', '/null.pdf', '[]', '', '');
            """)
            conn.execute("PRAGMA user_version = 3")
            conn.commit()
            conn.close()
            with self.assertRaises(LegacyDatabaseError):
                store.connect(db)

    def test_deleted_identity_marker_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "papers.db"
            conn = store.connect(db)
            conn.execute("DELETE FROM schema_meta WHERE key='identity_scheme'")
            conn.commit()
            conn.close()
            with self.assertRaises(LegacyDatabaseError):
                store.connect(db)

    def test_wrong_identity_marker_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "papers.db"
            conn = store.connect(db)
            conn.execute(
                "UPDATE schema_meta SET value='wrong' WHERE key='identity_scheme'"
            )
            conn.commit()
            conn.close()
            with self.assertRaises(LegacyDatabaseError):
                store.connect(db)

    def test_trigger_rejects_short_hash_insert(self):
        with tempfile.TemporaryDirectory() as d:
            conn = store.connect(Path(d) / "papers.db")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO papers (paper_hash, paper_name, pdf_path, "
                    "sections_completed, created_at, updated_at) "
                    "VALUES ('short', 'bad.pdf', '/bad.pdf', '[]', '', '')"
                )
            conn.close()

    def test_trigger_rejects_uppercase_hash_insert(self):
        with tempfile.TemporaryDirectory() as d:
            conn = store.connect(Path(d) / "papers.db")
            upper_hash = "A" * 64
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO papers (paper_hash, paper_name, pdf_path, "
                    "sections_completed, created_at, updated_at) "
                    f"VALUES ('{upper_hash}', 'bad.pdf', '/bad.pdf', '[]', '', '')"
                )
            conn.close()

    def test_trigger_accepts_valid_hash(self):
        with tempfile.TemporaryDirectory() as d:
            conn = store.connect(Path(d) / "papers.db")
            valid = "d" * 64
            conn.execute(
                "INSERT INTO papers (paper_hash, paper_name, pdf_path, "
                "sections_completed, created_at, updated_at) "
                f"VALUES ('{valid}', 'good.pdf', '/good.pdf', '[]', '', '')"
            )
            conn.commit()
            row = conn.execute(
                f"SELECT paper_name FROM papers WHERE paper_hash='{valid}'"
            ).fetchone()
            self.assertEqual(row["paper_name"], "good.pdf")
            conn.close()

    def test_v3_database_with_preexisting_wrong_marker_rejected(self):
        """A v3 database with schema_meta.identity_scheme='wrong' is caught on first connect."""
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "papers.db"
            conn = sqlite3.connect(str(db))
            conn.executescript("""
                CREATE TABLE papers (
                    paper_hash TEXT PRIMARY KEY, paper_name TEXT NOT NULL,
                    pdf_path TEXT NOT NULL, sections_completed TEXT NOT NULL DEFAULT '[]',
                    page_count INTEGER, chunk_strategy TEXT, model_used TEXT,
                    code_model TEXT, processed_at TEXT, summary_md TEXT,
                    symbolic_logic_md TEXT, cpp_examples_md TEXT, extras_md TEXT,
                    diagrams_raw_output TEXT, source_corpus TEXT,
                    created_at TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE processing_leases (
                    resource_key TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
                    generation INTEGER NOT NULL DEFAULT 1,
                    renewed_at REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE ocr_cache (
                    paper_hash TEXT NOT NULL, page_idx INTEGER NOT NULL,
                    ocr_profile TEXT NOT NULL, text TEXT NOT NULL,
                    PRIMARY KEY (paper_hash, page_idx, ocr_profile)
                );
                CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO schema_meta VALUES ('identity_scheme', 'wrong');
            """)
            conn.execute("PRAGMA user_version = 3")
            conn.commit()
            conn.close()

            # First connect must reject — not silently upgrade and defer to second
            with self.assertRaises(LegacyDatabaseError) as ctx:
                store.connect(db)
            self.assertIn("wrong", str(ctx.exception))

    def test_trigger_rejects_short_hash_update(self):
        with tempfile.TemporaryDirectory() as d:
            conn = store.connect(Path(d) / "papers.db")
            valid = "e" * 64
            conn.execute(
                "INSERT INTO papers (paper_hash, paper_name, pdf_path, "
                "sections_completed, created_at, updated_at) "
                f"VALUES ('{valid}', 'test.pdf', '/test.pdf', '[]', '', '')"
            )
            conn.commit()
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    f"UPDATE papers SET paper_hash = 'short' WHERE paper_hash = '{valid}'"
                )
            conn.close()

    def test_trigger_rejects_uppercase_hash_update(self):
        with tempfile.TemporaryDirectory() as d:
            conn = store.connect(Path(d) / "papers.db")
            valid = "e" * 64
            upper = "A" * 64
            conn.execute(
                "INSERT INTO papers (paper_hash, paper_name, pdf_path, "
                "sections_completed, created_at, updated_at) "
                f"VALUES ('{valid}', 'test.pdf', '/test.pdf', '[]', '', '')"
            )
            conn.commit()
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    f"UPDATE papers SET paper_hash = '{upper}' WHERE paper_hash = '{valid}'"
                )
            conn.close()

    def test_trigger_rejects_null_hash_update(self):
        with tempfile.TemporaryDirectory() as d:
            conn = store.connect(Path(d) / "papers.db")
            valid = "e" * 64
            conn.execute(
                "INSERT INTO papers (paper_hash, paper_name, pdf_path, "
                "sections_completed, created_at, updated_at) "
                f"VALUES ('{valid}', 'test.pdf', '/test.pdf', '[]', '', '')"
            )
            conn.commit()
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    f"UPDATE papers SET paper_hash = NULL WHERE paper_hash = '{valid}'"
                )
            conn.close()


class FencedWriteTests(unittest.TestCase):
    def _setup_db(self):
        d = tempfile.mkdtemp()
        db = Path(d) / "papers.db"
        conn = store.connect(db)
        claim = store.try_claim(conn, "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        conn.execute(
            "INSERT INTO papers (paper_hash, paper_name, pdf_path, "
            "sections_completed, created_at, updated_at) "
            "VALUES ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'test.pdf', '/test.pdf', '[]', '', '')"
        )
        conn.commit()
        return d, conn, claim

    def test_write_section_fenced_succeeds(self):
        d, conn, claim = self._setup_db()
        store.write_section_fenced(
            conn, "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", claim.owner_id, claim.generation,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "summary", "fenced content",
        )
        row = conn.execute("SELECT summary_md FROM papers WHERE paper_hash='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'").fetchone()
        self.assertEqual(row["summary_md"], "fenced content")
        store.release_claim(conn, "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", claim.owner_id, claim.generation)
        conn.close()

    def test_fenced_write_rejects_wrong_generation(self):
        d, conn, claim = self._setup_db()
        with self.assertRaises(LeaseLostError):
            store.write_section_fenced(
                conn, "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", claim.owner_id, claim.generation + 99,
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "summary", "stale",
            )
        store.release_claim(conn, "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", claim.owner_id, claim.generation)
        conn.close()

    def test_fenced_write_rejects_released_lease(self):
        d, conn, claim = self._setup_db()
        store.release_claim(conn, "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", claim.owner_id, claim.generation)
        with self.assertRaises(LeaseLostError):
            store.write_section_fenced(
                conn, "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", claim.owner_id, claim.generation,
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "summary", "stale",
            )
        conn.close()

    def test_generation_increments_on_reclaim_not_refresh(self):
        d, conn, claim = self._setup_db()
        self.assertEqual(claim.generation, 1)
        store.refresh_claim(conn, "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", claim.owner_id, claim.generation)
        row = conn.execute(
            "SELECT generation FROM processing_leases WHERE resource_key='sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'"
        ).fetchone()
        self.assertEqual(row["generation"], 1)
        store.release_claim(conn, "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", claim.owner_id, claim.generation)
        second = store.try_claim(conn, "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        self.assertEqual(second.generation, 2)
        store.release_claim(conn, "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", second.owner_id, second.generation)
        conn.close()

    def test_clear_section_fenced(self):
        d, conn, claim = self._setup_db()
        store.write_section_fenced(
            conn, "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", claim.owner_id, claim.generation,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "summary", "content",
        )
        store.clear_section_fenced(
            conn, "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", claim.owner_id, claim.generation,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "summary",
        )
        row = conn.execute("SELECT summary_md FROM papers WHERE paper_hash='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'").fetchone()
        self.assertIsNone(row["summary_md"])
        store.release_claim(conn, "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", claim.owner_id, claim.generation)
        conn.close()

    def test_concurrent_first_claim_serialized(self):
        """Two actual concurrent claimants — one wins, neither gets IntegrityError."""
        import threading

        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "papers.db"
            store.init_db(db)
            results = [None, None]
            barrier = threading.Barrier(2)

            def claim(idx):
                conn = store.connect(db)
                barrier.wait()
                results[idx] = store.try_claim(conn, "sha256:race", claimed_by=f"w{idx}")
                conn.close()

            threads = [threading.Thread(target=claim, args=(i,)) for i in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

            claimed = [r.claimed for r in results if r is not None]
            self.assertEqual(len(claimed), 2)
            self.assertEqual(claimed.count(True), 1)
            self.assertEqual(claimed.count(False), 1)


class ConcurrentStartupTests(unittest.TestCase):
    def test_init_then_concurrent_workers(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "papers.db"
            store.init_db(db)

            def worker(_):
                conn = store.connect(db)
                v = conn.execute("PRAGMA user_version").fetchone()[0]
                conn.close()
                return v

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(worker, range(20)))

            self.assertTrue(all(v == MIGRATIONS[-1].version for v in results))


if __name__ == "__main__":
    unittest.main()
