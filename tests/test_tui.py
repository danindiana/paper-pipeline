import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from paper_pipeline import store
from paper_pipeline.reader import hash_file
from paper_pipeline.tui import (
    SystemStatus,
    get_batch_progress,
    get_db_entity_report,
    get_system_status,
    render_batch_view,
    render_db_entities_view,
    render_system_view,
)


def _insert_paper(db_path, paper_hash, pdf_path, sections_completed):
    conn = store.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO papers (paper_hash, paper_name, pdf_path, sections_completed,
                                 created_at, updated_at)
            VALUES (?, ?, ?, ?, '', '')
            """,
            (paper_hash, Path(pdf_path).name, str(pdf_path), json.dumps(sections_completed)),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_lease(db_path, resource_key, renewed_at, active=1):
    conn = store.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO processing_leases (resource_key, owner_id, generation, renewed_at, active)
            VALUES (?, 'test-owner', 1, ?, ?)
            """,
            (resource_key, renewed_at, active),
        )
        conn.commit()
    finally:
        conn.close()


class BatchProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.papers_dir = Path(self.tmp.name) / "papers"
        self.papers_dir.mkdir()
        self.db_path = Path(self.tmp.name) / "test.db"
        store.init_db(self.db_path)

    def _make_pdf(self, name: str, content: bytes = b"dummy") -> Path:
        p = self.papers_dir / name
        p.write_bytes(content)
        return p

    def test_counts_and_status_for_complete_partial_and_not_started(self):
        complete_pdf = self._make_pdf("complete.pdf")
        partial_pdf = self._make_pdf("partial.pdf")
        self._make_pdf("untouched.pdf")  # no DB row at all

        _insert_paper(self.db_path, "a" * 64, complete_pdf,
                      sorted(store.ALL_SECTIONS))
        _insert_paper(self.db_path, "b" * 64, partial_pdf, ["summary", "logic"])

        progress = get_batch_progress(self.db_path, self.papers_dir)

        self.assertIsNone(progress.error)
        self.assertEqual(progress.total, 3)
        self.assertEqual(progress.complete, 1)
        self.assertEqual(progress.partial, 1)
        self.assertEqual(progress.not_started, 1)

        by_name = {r.name: r for r in progress.rows}
        self.assertEqual(by_name["complete.pdf"].status, "complete")
        self.assertEqual(by_name["partial.pdf"].status, "partial")
        self.assertEqual(set(by_name["partial.pdf"].missing), store.ALL_SECTIONS - {"summary", "logic"})
        self.assertEqual(by_name["untouched.pdf"].status, "not started")

    def test_in_progress_via_matched_paper_hash(self):
        # A paper that already has a metadata row resolves lease membership
        # directly through its own paper_hash -- no file hashing needed.
        pdf = self._make_pdf("inflight.pdf")
        paper_hash = "c" * 64
        _insert_paper(self.db_path, paper_hash, pdf, ["summary"])
        _insert_lease(self.db_path, f"sha256:{paper_hash}", time.time())

        progress = get_batch_progress(self.db_path, self.papers_dir)
        row = next(r for r in progress.rows if r.name == "inflight.pdf")
        self.assertTrue(row.in_progress)

    def test_in_progress_via_hash_fallback_for_not_started_paper(self):
        # A paper can be claimed (lease exists) before its first metadata
        # row is ever written -- must fall back to hashing the file itself
        # to resolve this case.
        pdf = self._make_pdf("claimed_but_no_row_yet.pdf", content=b"unique content")
        real_hash = hash_file(pdf)
        _insert_lease(self.db_path, f"sha256:{real_hash}", time.time())

        progress = get_batch_progress(self.db_path, self.papers_dir)
        row = next(r for r in progress.rows if r.name == "claimed_but_no_row_yet.pdf")
        self.assertEqual(row.status, "not started")
        self.assertTrue(row.in_progress)

    def test_stale_lease_is_not_shown_as_in_progress(self):
        pdf = self._make_pdf("stale.pdf")
        real_hash = hash_file(pdf)
        # Renewed long before the heartbeat-based cutoff -- a crashed
        # worker's abandoned lease, not live work.
        _insert_lease(self.db_path, f"sha256:{real_hash}", time.time() - 100_000)

        progress = get_batch_progress(self.db_path, self.papers_dir)
        row = next(r for r in progress.rows if r.name == "stale.pdf")
        self.assertFalse(row.in_progress)

    def test_missing_directory_reports_error_not_exception(self):
        progress = get_batch_progress(self.db_path, self.papers_dir / "nope")
        self.assertIsNotNone(progress.error)

    def test_missing_database_reports_error_not_exception(self):
        progress = get_batch_progress(self.papers_dir / "nope.db", self.papers_dir)
        self.assertIsNotNone(progress.error)


class DbEntityReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "test.db"
        store.init_db(self.db_path)

    def test_stats_reflect_real_row_counts(self):
        conn = store.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO papers (paper_hash, paper_name, pdf_path, sections_completed, "
                "created_at, updated_at) VALUES (?, 'p.pdf', '/x/p.pdf', ?, '', '')",
                ("d" * 64, json.dumps(sorted(store.ALL_SECTIONS))),
            )
            conn.execute(
                "INSERT INTO diagrams (paper_hash, idx, title, dot_src, svg_content) "
                "VALUES (?, 1, 't', 'digraph{}', '<svg/>')",
                ("d" * 64,),
            )
            conn.execute(
                "INSERT INTO diagrams (paper_hash, idx, title, dot_src, svg_content) "
                "VALUES (?, 2, 't2', 'digraph{}', NULL)",
                ("d" * 64,),
            )
            conn.execute(
                "INSERT INTO processing_leases (resource_key, owner_id, generation, "
                "renewed_at, active) VALUES ('sha256:d', 'o', 1, ?, 1)",
                (time.time(),),
            )
            conn.commit()
        finally:
            conn.close()

        cards, error = get_db_entity_report(self.db_path)
        self.assertIsNone(error)
        by_table = {c.table: c for c in cards}

        self.assertIn("1 total", " ".join(by_table["papers"].stats))
        self.assertIn("1 fully complete", " ".join(by_table["papers"].stats))
        self.assertIn("2 total", " ".join(by_table["diagrams"].stats))
        self.assertIn("1 rendered", " ".join(by_table["diagrams"].stats))
        self.assertIn("1 currently marked active", " ".join(by_table["processing_leases"].stats))
        self.assertIn("schema version 4", " ".join(by_table["schema_meta"].stats))

    def test_missing_database_reports_error(self):
        cards, error = get_db_entity_report(self.db_path.parent / "nope.db")
        self.assertEqual(cards, [])
        self.assertIsNotNone(error)


class SystemStatusTests(unittest.TestCase):
    def test_graceful_degradation_when_nothing_is_available(self):
        with patch("paper_pipeline.tui._http_get_json", return_value=None), \
             patch("subprocess.run", side_effect=FileNotFoundError), \
             patch("socket.create_connection", side_effect=OSError):
            status = get_system_status()

        self.assertIsNone(status.ollama_model)
        self.assertEqual(status.gpu_lines, [])
        self.assertEqual(status.graph_viz_neo4j, "not running")
        self.assertEqual(status.graph_viz_viewer, "not running")

        lines = render_system_view(status)
        self.assertTrue(any("none loaded" in ln for ln in lines))
        self.assertTrue(any("not available" in ln for ln in lines))


class RenderLayerTests(unittest.TestCase):
    def test_render_batch_view_shows_in_progress_marker(self):
        from paper_pipeline.tui import BatchProgress, PaperStatusRow

        progress = BatchProgress(
            papers_dir="/x",
            total=1, complete=0, partial=1, not_started=0,
            rows=[PaperStatusRow("a.pdf", "partial", ("cpp",), True)],
        )
        lines = render_batch_view(progress)
        self.assertTrue(any(ln.startswith("▶") for ln in lines))
        self.assertTrue(any("missing: cpp" in ln for ln in lines))

    def test_render_db_entities_view_surfaces_error(self):
        lines = render_db_entities_view([], "database not found: /x")
        self.assertTrue(any("database not found" in ln for ln in lines))


if __name__ == "__main__":
    unittest.main()
