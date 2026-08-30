import subprocess
import unittest

from paper_pipeline.diagrams import repair_dot_syntax


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


if __name__ == "__main__":
    unittest.main()
