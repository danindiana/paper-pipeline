import tempfile
import time
import unittest
from pathlib import Path

from paper_pipeline import store
from paper_pipeline.errors import LeaseLostError


class LeaseTests(unittest.TestCase):
    def test_refresh_and_release_require_owner_and_generation(self):
        with tempfile.TemporaryDirectory() as d:
            conn = store.connect(Path(d) / "papers.db")
            claim = store.try_claim(conn, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", claimed_by="owner-a", stale_after=1)
            self.assertTrue(claim.claimed)
            self.assertEqual(claim.generation, 1)

            # Correct owner+generation
            self.assertTrue(store.claim_owned(conn, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "owner-a", 1))
            # Wrong owner
            self.assertFalse(store.claim_owned(conn, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "owner-b", 1))
            # Wrong generation
            self.assertFalse(store.claim_owned(conn, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "owner-a", 99))

            # Refresh requires correct generation
            self.assertFalse(store.refresh_claim(conn, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "owner-b", 1))
            self.assertFalse(store.refresh_claim(conn, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "owner-a", 99))
            self.assertTrue(store.refresh_claim(conn, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "owner-a", 1))

            # Release requires correct generation
            self.assertFalse(store.release_claim(conn, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "owner-b", 1))
            self.assertFalse(store.release_claim(conn, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "owner-a", 99))
            self.assertTrue(store.release_claim(conn, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "owner-a", 1))
            conn.close()

    def test_heartbeat_prevents_stale_reclaim(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "papers.db"
            first = store.connect(db)
            second = store.connect(db)
            claim = store.try_claim(first, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", claimed_by="owner-a", stale_after=0.20)
            heartbeat = store.LeaseHeartbeat(
                db, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", claim.owner_id, claim.generation,
                interval=0.03, stale_after=0.20,
            ).start()
            try:
                time.sleep(0.32)
                contender = store.try_claim(second, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", claimed_by="owner-b", stale_after=0.20)
                self.assertFalse(contender.claimed)
                heartbeat.assert_healthy()
            finally:
                heartbeat.stop()
                store.release_claim(first, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", claim.owner_id, claim.generation)
                first.close()
                second.close()

    def test_heartbeat_detects_owner_replacement(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "papers.db"
            first = store.connect(db)
            mutator = store.connect(db)
            claim = store.try_claim(first, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", claimed_by="owner-a", stale_after=0.20)
            heartbeat = store.LeaseHeartbeat(
                db, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", claim.owner_id, claim.generation,
                interval=0.03, stale_after=0.20,
            ).start()
            # Force owner replacement
            mutator.execute(
                "UPDATE processing_leases SET owner_id = ?, generation = generation + 1 "
                "WHERE resource_key = ?",
                ("owner-b", "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
            )
            mutator.commit()
            deadline = time.monotonic() + 1.0
            detected = False
            while time.monotonic() < deadline:
                try:
                    heartbeat.assert_healthy()
                except LeaseLostError:
                    detected = True
                    break
                time.sleep(0.02)
            heartbeat.stop()
            self.assertTrue(detected)
            # Old owner cannot release
            self.assertFalse(store.release_claim(first, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", claim.owner_id, claim.generation))
            # New owner can
            new_gen = mutator.execute(
                "SELECT generation FROM processing_leases WHERE resource_key='sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'"
            ).fetchone()[0]
            self.assertTrue(store.release_claim(mutator, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "owner-b", new_gen))
            first.close()
            mutator.close()

    def test_stopped_heartbeat_allows_reclaim(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "papers.db"
            first = store.connect(db)
            second = store.connect(db)
            claim = store.try_claim(first, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", claimed_by="owner-a", stale_after=0.15)
            heartbeat = store.LeaseHeartbeat(
                db, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", claim.owner_id, claim.generation,
                interval=0.03, stale_after=0.15,
            ).start()
            time.sleep(0.08)
            heartbeat.stop()
            time.sleep(0.18)
            reclaimed = store.try_claim(second, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", claimed_by="owner-b", stale_after=0.15)
            self.assertTrue(reclaimed.claimed)
            self.assertTrue(reclaimed.reclaimed)
            self.assertEqual(reclaimed.generation, claim.generation + 1)
            # Old owner cannot release the new lease
            self.assertFalse(store.release_claim(first, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", claim.owner_id, claim.generation))
            store.release_claim(second, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "owner-b", reclaimed.generation)
            first.close()
            second.close()

    def test_expired_lease_cannot_be_refreshed(self):
        with tempfile.TemporaryDirectory() as d:
            conn = store.connect(Path(d) / "papers.db")
            claim = store.try_claim(conn, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", stale_after=0.05)
            time.sleep(0.08)
            # Refresh should fail — lease is expired
            self.assertFalse(
                store.refresh_claim(conn, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", claim.owner_id, claim.generation, stale_after=0.05)
            )
            conn.close()

    def test_expired_lease_cannot_authorize_fenced_write(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "papers.db"
            conn = store.connect(db)
            claim = store.try_claim(conn, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", stale_after=0.05)
            conn.execute(
                "INSERT INTO papers (paper_hash, paper_name, pdf_path, "
                "sections_completed, created_at, updated_at) "
                "VALUES ('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', 'test.pdf', '/test.pdf', '[]', '', '')"
            )
            conn.commit()
            time.sleep(0.08)
            with self.assertRaises(LeaseLostError):
                store.write_section_fenced(
                    conn, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", claim.owner_id, claim.generation,
                    "paper", "summary", "expired write",
                    stale_after=0.05,
                )
            conn.close()

    def test_concurrent_first_claim_does_not_raise(self):
        """Two claimants seeing no row should not get IntegrityError."""
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "papers.db"
            c1 = store.connect(db)
            c2 = store.connect(db)
            r1 = store.try_claim(c1, "sha256:race", claimed_by="a")
            r2 = store.try_claim(c2, "sha256:race", claimed_by="b")
            self.assertTrue(r1.claimed)
            self.assertFalse(r2.claimed)
            store.release_claim(c1, "sha256:race", "a", r1.generation)
            c1.close()
            c2.close()


if __name__ == "__main__":
    unittest.main()
