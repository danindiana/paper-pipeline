import subprocess
import unittest

from paper_pipeline.diagrams import parse_diagrams, repair_dot_syntax


def _dot_available() -> bool:
    try:
        subprocess.run(["dot", "-V"], capture_output=True, timeout=5)
        return True
    except FileNotFoundError:
        return False


class RepairDotSyntaxTests(unittest.TestCase):
    def test_quote_wrapped_label_with_closing_quote_is_repaired(self):
        # Observed live: the model wraps an entire label='...' assignment in
        # an extra, incorrect outer pair of double quotes.
        broken = (
            'digraph G {\n'
            '  Root ["label=\'Milnor K-group $K_2F/2$ (Rational Function '
            'Field $F=E(t)$)\'"];\n'
            '}\n'
        )
        repaired = repair_dot_syntax(broken)
        self.assertIn(
            'Root [label="Milnor K-group $K_2F/2$ (Rational Function '
            'Field $F=E(t)$)"];',
            repaired,
        )

    def test_quote_wrapped_label_missing_outer_closer_is_repaired(self):
        # A second real variant: the outer quote's closer is dropped
        # entirely, leaving only the inner single-quote before `];`.
        broken = "digraph G {\n  KGroup [\"label='Milnor K-group $K_2F/2$'];\n}\n"
        repaired = repair_dot_syntax(broken)
        self.assertIn('KGroup [label="Milnor K-group $K_2F/2$"];', repaired)

    def test_backslashes_in_label_are_escaped_for_new_quoting_context(self):
        # A literal backslash must be doubled in the repaired form so
        # Graphviz doesn't reinterpret e.g. \r as its own right-justified
        # linebreak escape instead of a literal backslash + letter.
        broken = "digraph G {\n  Ramification [\"label='Sequences $\\\\rho$'\"];\n}\n"
        repaired = repair_dot_syntax(broken)
        self.assertIn('label="Sequences $\\\\\\\\rho$"', repaired)

    def test_already_valid_syntax_is_left_untouched(self):
        valid = 'digraph G {\n  Root [label="Already valid", color="#00FF41"];\n}\n'
        self.assertEqual(repair_dot_syntax(valid), valid)

    @unittest.skipUnless(_dot_available(), "graphviz `dot` binary not installed")
    def test_repaired_output_actually_renders(self):
        broken = (
            'digraph G {\n'
            '  Root ["label=\'Renders now\'"];\n'
            '}\n'
        )
        repaired = repair_dot_syntax(broken)
        r = subprocess.run(
            ["dot", "-Tsvg"], input=repaired, text=True, capture_output=True, timeout=15
        )
        self.assertEqual(r.returncode, 0, r.stderr)


class ParseDiagramsTests(unittest.TestCase):
    def test_typo_end_delimiter_does_not_swallow_next_diagram(self):
        # Real typo observed live in production: "===DIOD_END===" instead of
        # "===DIAGRAM_END===". A strict end-delimiter match skips past this
        # and merges the next diagram into this one's content, losing the
        # "swallowed" diagram entirely.
        raw = (
            "===DIAGRAM_START: First===\n"
            "digraph G { A -> B; }\n"
            "===DIOD_END===\n"
            "\n"
            "===DIAGRAM_START: Second===\n"
            "digraph G { C -> D; }\n"
            "===DIAGRAM_END===\n"
        )
        diagrams = parse_diagrams(raw)
        self.assertEqual([title for title, _ in diagrams], ["First", "Second"])
        self.assertNotIn("===", diagrams[0][1])
        self.assertIn("A -> B", diagrams[0][1])
        self.assertIn("C -> D", diagrams[1][1])

    def test_transposed_letter_typo_end_delimiter(self):
        # A second, independently different real typo observed live:
        # "===DIARGAM_END===" (transposed letters) -- confirms the fix is
        # tolerant of typos in general, not patched for one specific one.
        raw = (
            "===DIAGRAM_START: Only===\n"
            "digraph G { X -> Y; }\n"
            "===DIARGAM_END===\n"
        )
        diagrams = parse_diagrams(raw)
        self.assertEqual(len(diagrams), 1)
        self.assertNotIn("===", diagrams[0][1])
        self.assertIn("X -> Y", diagrams[0][1])

    def test_well_formed_multi_diagram_input_unaffected(self):
        raw = (
            "===DIAGRAM_START: Alpha===\n"
            "digraph G { A -> B; }\n"
            "===DIAGRAM_END===\n"
            "\n"
            "===DIAGRAM_START: Beta===\n"
            "digraph G { C -> D; }\n"
            "===DIAGRAM_END===\n"
        )
        diagrams = parse_diagrams(raw)
        self.assertEqual([title for title, _ in diagrams], ["Alpha", "Beta"])
        self.assertEqual(diagrams[0][1], "digraph G { A -> B; }")
        self.assertEqual(diagrams[1][1], "digraph G { C -> D; }")

    def test_missing_end_delimiter_entirely_still_recovers_diagram(self):
        # No end marker at all for the last diagram (e.g. generation was cut
        # off) -- should still recover cleanly up to end of text rather than
        # requiring a well-formed end marker to exist.
        raw = "===DIAGRAM_START: Only===\ndigraph G { A -> B; }\n"
        diagrams = parse_diagrams(raw)
        self.assertEqual(len(diagrams), 1)
        self.assertIn("A -> B", diagrams[0][1])


if __name__ == "__main__":
    unittest.main()
