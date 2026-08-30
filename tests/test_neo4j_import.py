"""
Tests for neo4j_viz/import_to_neo4j.py's import_paper() logic.

Not part of the paper_pipeline package (neo4j_viz/ is a standalone,
optional feature -- see its own README), but tested from here since this
is where the test suite and CI already run. import_paper() takes a driver
session as a plain argument and never imports the `neo4j` package itself,
so it's testable with a recording mock session and no live Neo4j instance,
matching how tests/test_empty_generation_retry.py mocks `requests` instead
of hitting a live Ollama.
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neo4j_viz"))
from import_to_neo4j import import_paper  # noqa: E402


class RecordingSession:
    """Records every Cypher call's parameters instead of executing anything."""

    def __init__(self):
        self.calls: list[dict] = []

    def run(self, query, **params):
        self.calls.append({"query": query, "params": params})
        return []

    def merges_for(self, label: str) -> list[dict]:
        return [
            c["params"] for c in self.calls
            if f"MERGE ({label[0].lower()}:{label}" in c["query"]
        ]


def _paper_row(paper_hash: str, name: str, evidence_ids: list[str], relationships=None):
    return {
        "paper_hash": paper_hash,
        "paper_name": name,
        "model_used": "gemma4:26b-a4b-it-q4_K_M",
        "page_count": 10,
        "processed_at": "2026-08-30T00:00:00",
        "source_corpus": json.dumps({
            "chunks": [{
                "evidence": [
                    {
                        "evidence_id": eid, "kind": "result",
                        "statement": f"statement for {eid}",
                        "support": f"support for {eid}", "page": 1,
                    }
                    for eid in evidence_ids
                ],
            }],
            "root": {"relationships": relationships or []},
        }),
    }


class ImportPaperQidTests(unittest.TestCase):
    def test_same_evidence_id_in_different_papers_gets_distinct_qid(self):
        # evidence_id ("C001-E001") is only unique within one paper's own
        # bundle -- every paper's first chunk starts back at C001-E001.
        # Merging on that alone would collapse unrelated papers' evidence
        # into one shared node.
        session_a = RecordingSession()
        import_paper(session_a, _paper_row("hashA", "paper-a.pdf", ["C001-E001"]), [])
        session_b = RecordingSession()
        import_paper(session_b, _paper_row("hashB", "paper-b.pdf", ["C001-E001"]), [])

        qid_a = session_a.merges_for("Evidence")[0]["qid"]
        qid_b = session_b.merges_for("Evidence")[0]["qid"]

        self.assertNotEqual(qid_a, qid_b)
        self.assertIn("hashA", qid_a)
        self.assertIn("hashB", qid_b)

    def test_qid_is_paper_hash_and_evidence_id_composite(self):
        session = RecordingSession()
        import_paper(session, _paper_row("hashX", "paper.pdf", ["C001-E001"]), [])
        params = session.merges_for("Evidence")[0]
        self.assertEqual(params["qid"], "hashX:C001-E001")
        self.assertEqual(params["evidence_id"], "C001-E001")

    def test_relationship_edges_matched_by_qid_not_bare_evidence_id(self):
        session = RecordingSession()
        row = _paper_row(
            "hashY", "paper.pdf", ["C001-E001", "C001-E002"],
            relationships=[{"evidence_ids": ["C001-E001", "C001-E002"], "description": "relates"}],
        )
        _, n_rel = import_paper(session, row, [])
        self.assertEqual(n_rel, 1)

        rel_calls = [c for c in session.calls if "RELATES_TO" in c["query"]]
        self.assertEqual(len(rel_calls), 1)
        params = rel_calls[0]["params"]
        self.assertEqual(params["qid_a"], "hashY:C001-E001")
        self.assertEqual(params["qid_b"], "hashY:C001-E002")

    def test_counts_returned_match_input(self):
        session = RecordingSession()
        row = _paper_row("hashZ", "paper.pdf", ["C001-E001", "C001-E002", "C001-E003"])
        n_evidence, n_rel = import_paper(session, row, [])
        self.assertEqual(n_evidence, 3)
        self.assertEqual(n_rel, 0)

    def test_malformed_source_corpus_is_skipped_not_raised(self):
        session = RecordingSession()
        row = _paper_row("hashW", "paper.pdf", ["C001-E001"])
        row["source_corpus"] = "not valid json"
        n_evidence, n_rel = import_paper(session, row, [])
        self.assertEqual((n_evidence, n_rel), (0, 0))


if __name__ == "__main__":
    unittest.main()
