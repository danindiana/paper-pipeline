import json
import re
import unittest

from paper_pipeline.ocr import ExtractedPage
from paper_pipeline.reader import (
    EvidenceBundle,
    EvidenceSynthesisError,
    build_chunks,
    synthesize_evidence,
)


class FakeEvidenceModel:
    def __init__(self):
        self.map_calls = 0
        self.reduce_calls = 0

    def __call__(self, prompt: str) -> str:
        if "Extract a compact evidence record" in prompt:
            self.map_calls += 1
            chunk = int(re.search(r"Chunk: (\d+)", prompt).group(1))
            page = int(re.search(r"\[PAGE (\d+)\]", prompt).group(1))
            support = f"Evidence on physical page {page} establishes result {page}."
            return json.dumps(
                {
                    "chunk_summary": f"Chunk {chunk} covers physical page {page}.",
                    "evidence": [
                        {
                            "kind": "result",
                            "statement": f"Result {page} is reported.",
                            "support": support,
                            "page": page,
                        }
                    ],
                }
            )
        if "Synthesize the supplied child dossiers" in prompt:
            self.reduce_calls += 1
            payload = json.loads(prompt.split("INPUT:\n", 1)[1])
            ids = [row["evidence_id"] for row in payload["evidence_records"]]
            children = payload["child_dossiers"]
            return json.dumps(
                {
                    "summary": (
                        f"Synthesis spanning pages {children[0]['page_start']} through "
                        f"{children[-1]['page_end']}."
                    ),
                    "selected_evidence_ids": ids[:16],
                    "relationships": [],
                }
            )
        raise AssertionError("unexpected prompt")


class EvidenceTests(unittest.TestCase):
    def test_multilevel_synthesis_preserves_late_page_coverage(self):
        pages = [
            ExtractedPage(
                number,
                f"Evidence on physical page {number} establishes result {number}.",
                "native",
            )
            for number in range(1, 21)
        ]
        chunks = build_chunks(pages, window=1, overlap=0, char_cap=2_000)
        model = FakeEvidenceModel()
        bundle = synthesize_evidence(
            chunks,
            model,
            paper_hash="abc123",
            model="test-model",
            physical_page_count=20,
        )

        self.assertEqual(bundle.reduction_levels, 3)
        self.assertEqual(len(bundle.evidence), 20)
        self.assertIn("C020-E001", bundle.root.selected_evidence_ids)
        context = bundle.context_for("summary", 12_000)
        self.assertLessEqual(len(context), 12_000)
        self.assertIn("page 20", context)
        self.assertIn("Evidence on physical page 20", context)

        restored = EvidenceBundle.from_json(
            bundle.to_json(), paper_hash="abc123", model="test-model"
        )
        self.assertEqual(restored.root, bundle.root)
        self.assertEqual(restored.evidence, bundle.evidence)

    def test_unverifiable_support_is_rejected(self):
        page = ExtractedPage(7, "The measured accuracy was 91.4 percent.", "native")
        chunks = build_chunks([page], window=1, overlap=0, char_cap=2_000)

        def hallucinating_model(_: str) -> str:
            return json.dumps(
                {
                    "chunk_summary": "A result is claimed.",
                    "evidence": [
                        {
                            "kind": "result",
                            "statement": "Accuracy was perfect.",
                            "support": "The measured accuracy was 100 percent.",
                            "page": 7,
                        }
                    ],
                }
            )

        with self.assertRaises(EvidenceSynthesisError):
            synthesize_evidence(
                chunks,
                hallucinating_model,
                paper_hash="bad",
                model="test-model",
                physical_page_count=7,
            )

    def test_control_character_in_verbatim_support_does_not_break_parsing(self):
        # PDF math/symbol-font extraction occasionally yields raw control
        # bytes (observed live: U+0001-U+0003, U+0012-U+0013) in place of
        # garbled glyphs. The model quotes `support` verbatim as instructed,
        # so those bytes land unescaped inside a JSON string in the raw
        # response, which a strict decoder rejects.
        page = ExtractedPage(3, "The matrix satisfies S(\x01) = 2 for all inputs.", "native")
        chunks = build_chunks([page], window=1, overlap=0, char_cap=2_000)

        def model_with_control_char(_: str) -> str:
            return (
                "```json\n"
                "{\n"
                '  "chunk_summary": "Describes a matrix identity.",\n'
                '  "evidence": [\n'
                "    {\n"
                '      "kind": "equation",\n'
                '      "statement": "The matrix satisfies the identity.",\n'
                '      "support": "The matrix satisfies S(\x01) = 2 for all inputs.",\n'
                '      "page": 3\n'
                "    }\n"
                "  ]\n"
                "}\n"
                "```"
            )

        bundle = synthesize_evidence(
            chunks,
            model_with_control_char,
            paper_hash="ctrl123",
            model="test-model",
            physical_page_count=1,
        )
        self.assertEqual(len(bundle.evidence), 1)
        self.assertEqual(bundle.evidence[0].page, 3)

    def test_unescaped_latex_backslash_in_support_does_not_break_parsing(self):
        # Math-heavy papers' `support` excerpts often contain LaTeX, e.g.
        # \dot{u}. The model quotes it verbatim (as instructed) but writes
        # a single, un-doubled backslash, which isn't a valid JSON escape
        # (\d) and breaks a strict decode of the outer object.
        page = ExtractedPage(5, "The derivative \\dot{u} appears in the equation.", "native")
        chunks = build_chunks([page], window=1, overlap=0, char_cap=2_000)

        def model_with_bad_escape(_: str) -> str:
            return (
                "```json\n"
                "{\n"
                '  "chunk_summary": "Describes a derivative term.",\n'
                '  "evidence": [\n'
                "    {\n"
                '      "kind": "equation",\n'
                '      "statement": "The derivative term is central to the model.",\n'
                '      "support": "The derivative \\dot{u} appears in the equation.",\n'
                '      "page": 5\n'
                "    }\n"
                "  ]\n"
                "}\n"
                "```"
            )

        bundle = synthesize_evidence(
            chunks,
            model_with_bad_escape,
            paper_hash="latex123",
            model="test-model",
            physical_page_count=1,
        )
        self.assertEqual(len(bundle.evidence), 1)
        self.assertEqual(bundle.evidence[0].page, 5)

    def test_chunk_labels_retain_physical_page_numbers(self):
        pages = [
            ExtractedPage(2, "page two", "native"),
            ExtractedPage(9, "page nine", "ocr"),
        ]
        chunk = build_chunks(pages, window=2, overlap=0, char_cap=2_000)[0]
        self.assertEqual(chunk.page_start, 2)
        self.assertEqual(chunk.page_end, 9)
        self.assertIn("[PAGE 2]", chunk.render())
        self.assertIn("[PAGE 9]", chunk.render())


if __name__ == "__main__":
    unittest.main()
