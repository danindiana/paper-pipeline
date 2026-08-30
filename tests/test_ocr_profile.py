import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper_pipeline import store
from paper_pipeline.migrations import MIGRATIONS
from paper_pipeline.ocr import ocr_profile_key


class OcrProfileTests(unittest.TestCase):
    def test_different_dpi_produces_different_profile(self):
        a = ocr_profile_key(dpi=300, lang="eng")
        b = ocr_profile_key(dpi=600, lang="eng")
        self.assertNotEqual(a, b)

    def test_different_lang_produces_different_profile(self):
        a = ocr_profile_key(dpi=300, lang="eng")
        b = ocr_profile_key(dpi=300, lang="eng+deu")
        self.assertNotEqual(a, b)

    def test_same_config_produces_same_profile(self):
        a = ocr_profile_key(dpi=300, lang="eng")
        b = ocr_profile_key(dpi=300, lang="eng")
        self.assertEqual(a, b)

    def test_profiles_coexist_in_cache(self):
        with tempfile.TemporaryDirectory() as d:
            conn = store.connect(Path(d) / "papers.db")
            store.put_cached_ocr_page(conn, "hash1", 0, "profile-a", "text from profile A")
            store.put_cached_ocr_page(conn, "hash1", 0, "profile-b", "text from profile B")

            a = store.get_cached_ocr_page(conn, "hash1", 0, "profile-a")
            b = store.get_cached_ocr_page(conn, "hash1", 0, "profile-b")
            self.assertEqual(a, "text from profile A")
            self.assertEqual(b, "text from profile B")
            conn.close()

    def test_profile_miss_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            conn = store.connect(Path(d) / "papers.db")
            store.put_cached_ocr_page(conn, "hash1", 0, "profile-a", "text A")

            miss = store.get_cached_ocr_page(conn, "hash1", 0, "profile-unknown")
            self.assertIsNone(miss)
            conn.close()

    def test_insert_or_ignore_preserves_first_result(self):
        with tempfile.TemporaryDirectory() as d:
            conn = store.connect(Path(d) / "papers.db")
            store.put_cached_ocr_page(conn, "hash1", 0, "profile-a", "first")
            store.put_cached_ocr_page(conn, "hash1", 0, "profile-a", "second")

            result = store.get_cached_ocr_page(conn, "hash1", 0, "profile-a")
            self.assertEqual(result, "first")
            conn.close()

    @patch("paper_pipeline.ocr._tesseract_version", return_value="5.4.0")
    def test_profile_includes_tesseract_version(self, mock_ver):
        a = ocr_profile_key(dpi=300, lang="eng")
        mock_ver.return_value = "5.5.0"
        from paper_pipeline.ocr import _tesseract_version
        _tesseract_version.cache_clear()
        b = ocr_profile_key(dpi=300, lang="eng")
        _tesseract_version.cache_clear()
        self.assertNotEqual(a, b)

    @patch("paper_pipeline.ocr._traineddata_fingerprint")
    def test_traineddata_change_invalidates_profile(self, mock_fp):
        mock_fp.return_value = "fingerprint-v1"
        a = ocr_profile_key(dpi=300, lang="eng")
        mock_fp.return_value = "fingerprint-v2"
        b = ocr_profile_key(dpi=300, lang="eng")
        self.assertNotEqual(a, b)

    @patch("paper_pipeline.ocr._OCR_PROFILE_SCHEMA", 1)
    def test_schema_version_change_invalidates_profile(self):
        import paper_pipeline.ocr as ocr_mod
        a = ocr_profile_key(dpi=300, lang="eng")
        original = ocr_mod._OCR_PROFILE_SCHEMA
        ocr_mod._OCR_PROFILE_SCHEMA = 2
        try:
            b = ocr_profile_key(dpi=300, lang="eng")
            self.assertNotEqual(a, b)
        finally:
            ocr_mod._OCR_PROFILE_SCHEMA = original


class MigrationV3Tests(unittest.TestCase):
    def test_v2_to_v3_purges_unprofiled_cache(self):
        """Migration 3 intentionally drops the old unprofiled ocr_cache."""
        import sqlite3 as raw_sqlite
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "papers.db"
            # Create a v2 database with old-style ocr_cache
            conn = raw_sqlite.connect(str(db))
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
                    text TEXT NOT NULL, PRIMARY KEY (paper_hash, page_idx)
                );
                INSERT INTO ocr_cache VALUES ('cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc', 0, 'old unprofiled text');
            """)
            conn.execute("PRAGMA user_version = 2")
            conn.commit()
            conn.close()

            # Connect through store → runs migration 3
            conn = store.connect(db)
            from paper_pipeline.migrations import get_version
            self.assertEqual(get_version(conn), MIGRATIONS[-1].version)

            # Old cache data should be gone
            row = conn.execute("SELECT * FROM ocr_cache").fetchone()
            self.assertIsNone(row)

            # New schema has ocr_profile column
            cols = [info[1] for info in conn.execute("PRAGMA table_info(ocr_cache)").fetchall()]
            self.assertIn("ocr_profile", cols)
            conn.close()


if __name__ == "__main__":
    unittest.main()
