import json
import re
import tempfile
import unittest
from pathlib import Path
from threading import Event

import pymupdf

from paper_pipeline import store
from paper_pipeline.errors import LeaseLostError
from paper_pipeline.pipeline import PaperProcessor, PaperStatus


class PipelineModel:
    def __init__(self, valid_diagram: bool):
        self.valid_diagram = valid_diagram
        self.evidence_calls = 0
        self.section_calls = 0

    def generate(self, model: str, prompt: str, ctx_tokens=None) -> str:
        if "Extract a compact evidence record" in prompt:
            self.evidence_calls += 1
            page = int(re.search(r"\[PAGE (\d+)\]", prompt).group(1))
            support = "The method achieves 91 percent accuracy on the benchmark."
            return json.dumps({
                "chunk_summary": "The paper reports a benchmark result.",
                "evidence": [{
                    "kind": "result",
                    "statement": "The method reports 91 percent accuracy.",
                    "support": support,
                    "page": page,
                }],
            })
        self.section_calls += 1
        if "Graphviz DOT" in prompt:
            if not self.valid_diagram:
                return "invalid diagram output"
            return (
                "===DIAGRAM_START: Result===\n"
                "digraph G { A -> B; }\n"
                "===DIAGRAM_END==="
            )
        return "Grounded section output citing C001-E001 on page 1."


def _make_test_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "The method achieves 91 percent accuracy on the benchmark. " * 3,
    )
    doc.save(path)
    doc.close()


class PipelineEvidenceCacheTests(unittest.TestCase):
    def test_partial_retry_reuses_evidence_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            pdf = Path(d) / "paper.pdf"
            db = Path(d) / "papers.db"
            _make_test_pdf(pdf)

            first_model = PipelineModel(valid_diagram=False)
            first = PaperProcessor(first_model, Event(), db, ocr_mode="never")
            self.assertIs(first.process(pdf), PaperStatus.PARTIAL)
            self.assertEqual(first_model.evidence_calls, 1)

            second_model = PipelineModel(valid_diagram=True)
            second = PaperProcessor(second_model, Event(), db, ocr_mode="never")
            self.assertIs(second.process(pdf), PaperStatus.COMPLETE)
            self.assertEqual(second_model.evidence_calls, 0)
            self.assertEqual(second_model.section_calls, 1)

            conn = store.connect(db)
            record = store.load_paper_by_pdf_path(conn, str(pdf))
            self.assertTrue(record.source_corpus)
            self.assertEqual(set(record.sections_completed), store.ALL_SECTIONS)
            conn.close()

    def test_owner_replacement_stops_before_section_write(self):
        with tempfile.TemporaryDirectory() as d:
            pdf = Path(d) / "paper.pdf"
            db = Path(d) / "papers.db"
            _make_test_pdf(pdf)

            class LeaseStealingModel(PipelineModel):
                stolen = False

                def generate(self, model_name: str, prompt: str, ctx_tokens=None) -> str:
                    if "Extract a compact evidence record" not in prompt and not self.stolen:
                        self.stolen = True
                        conn = store.connect(db)
                        # Steal the lease by changing owner and advancing generation
                        conn.execute(
                            "UPDATE processing_leases SET owner_id='owner-b', "
                            "generation=generation+1"
                        )
                        conn.commit()
                        conn.close()
                    return super().generate(model_name, prompt, ctx_tokens)

            processor = PaperProcessor(
                LeaseStealingModel(valid_diagram=True), Event(), db, ocr_mode="never"
            )
            with self.assertRaises(LeaseLostError):
                processor.process(pdf)

            conn = store.connect(db)
            record = store.load_paper_by_pdf_path(conn, str(pdf))
            # No sections should have been written — the fenced write rejects
            self.assertEqual(record.sections_completed, [])
            # Clean up the stolen lease
            new_gen = conn.execute(
                "SELECT generation FROM processing_leases"
            ).fetchone()[0]
            store.release_claim(conn, f"sha256:{record.paper_hash}", "owner-b", new_gen)
            conn.close()


if __name__ == "__main__":
    unittest.main()
