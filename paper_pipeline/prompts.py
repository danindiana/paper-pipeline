"""
prompts.py — LLM prompt templates for each pipeline section.

Content is identical to the original prompts; only the container changed.
Add new sections by adding a key here and a Section entry in pipeline.py.
"""

from __future__ import annotations

import textwrap


EVIDENCE_MAP_PROMPT: str = textwrap.dedent("""\
    Extract a compact evidence record from this chunk of a research paper.
    The input contains explicit [PAGE N] markers referring to physical PDF pages.

    Return ONLY one JSON object with this schema:
    {
      "chunk_summary": "A factual synthesis of this chunk in at most 220 words",
      "evidence": [
        {
          "kind": "motivation|method|algorithm|equation|result|limitation|comparison|deployment|definition",
          "statement": "A precise claim supported by the cited page",
          "support": "An exact, verbatim excerpt copied from one cited page",
          "page": 12
        }
      ]
    }

    Requirements:
    - Extract 3–12 high-value evidence items when the chunk supports them.
    - `support` must be copied exactly from the cited page, not paraphrased.
    - Never cite a page number absent from the input.
    - Do not infer results, theorems, limitations, or implementation details.
    - Prefer numerical results, equations, algorithm steps, definitions, and
      explicit limitations over generic background statements.
    - Use an empty evidence list only if the chunk truly contains no substantive
      technical claim. Do not include Markdown fences or explanatory prose.
""")


EVIDENCE_REDUCE_PROMPT: str = textwrap.dedent("""\
    Synthesize the supplied child dossiers into one higher-level dossier.
    Every evidence item has an immutable evidence ID and verified source page.

    Return ONLY one JSON object with this schema:
    {
      "summary": "A cross-dossier synthesis in at most 350 words",
      "selected_evidence_ids": ["C001-E001"],
      "relationships": [
        {
          "evidence_ids": ["C001-E001", "C002-E003"],
          "description": "A relationship supported by those evidence items"
        }
      ]
    }

    Requirements:
    - Select at most {max_selected} evidence IDs.
    - Use only IDs present in the input. Never create or alter an ID.
    - Preserve evidence from across the entire page range, including late pages.
    - Connect methods to results, assumptions to limitations, and comparisons to
      measured outcomes where the supplied evidence supports those relationships.
    - Do not introduce facts absent from the child dossiers.
    - Do not include Markdown fences or explanatory prose.
""")

PROMPTS: dict[str, str] = {
    "summary": textwrap.dedent("""\
        Write a comprehensive, well-structured summary of this AI/ML research paper.
        Your summary must cover ALL of the following:
          1. Motivation & Problem Statement — what gap does this address?
          2. Core Methodology — how does the approach work at a high level?
          3. Key Contributions — what is genuinely novel?
          4. Experimental Setup & Results — benchmarks, datasets, metrics, numbers
          5. Limitations & Failure Modes — where does the approach break down?
          6. Significance — how does this advance the field?
        Be thorough and precise. Use section headers."""),

    "logic": textwrap.dedent("""\
        Refactor the paper's core insights using rigorous symbolic logic and formal notation.
        Structure your response as:

        ## 1. Core Definitions & Notation
        Define all entities, sets, and functions with formal notation (∈, ⊂, →, ℝⁿ, etc.)

        ## 2. Key Theorems & Propositions
        State the paper's central claims as formal propositions with ∀, ∃, →, ↔ quantifiers.

        ## 3. Algorithm Formalisation
        Express each major algorithm using pseudocode with mathematical notation,
        loop invariants, and complexity bounds (O, Ω, Θ).

        ## 4. Optimality & Convergence Conditions
        State any convergence theorems, loss landscape properties, or PAC-learning bounds.

        ## 5. Information-Theoretic View
        Express the core learning objective using entropy H(·), KL divergence, mutual information I(·;·) where applicable.

        At the very end of your response, output a fenced JSON block containing all parsed concepts, theorems, and algorithms.
        Use this exact schema:
        ```json
        {
          "concepts": [{"name": "Concept Name", "description": "Concept Description"}],
          "theorems": [{"name": "Theorem Name", "statement": "Theorem Statement"}],
          "algorithms": [{"name": "Algorithm Name", "pseudocode": "Brief pseudocode text", "invariant": "Invariant condition"}]
        }
        ```"""),

    "cpp": textwrap.dedent("""\
        Refactor the paper's core insights using well-crafted C++ code examples.
        Requirements:
          - Use modern C++20 / C++23 (concepts, ranges, coroutines, std::expected, etc.)
          - Implement the key algorithms and data structures described in the paper
          - Each code block must open with a comment block citing the relevant paper section/equation
          - Provide at least 3 self-contained, compilable examples
          - Include a main() that exercises each implementation with sample data
          - Prefer STL containers and algorithms; avoid raw owning pointers
          - Show template metaprogramming or concept constraints where they model the paper's abstractions
          - Add inline comments explaining the mapping from math → code

        At the very end of your response, output a fenced JSON block containing all C++ examples.
        Use this exact schema:
        ```json
        {
          "examples": [{"name": "Example Name", "code": "The full C++ code string", "complexity": "Big O string if applicable"}]
        }
        ```"""),

    "extras": textwrap.dedent("""\
        Provide deep additional analysis beyond what the paper itself claims:

        ## 1. Open Questions
        What does this paper leave unresolved? What follow-up experiments are obviously needed?

        ## 2. Related Work & Connections
        How does this relate to other landmark AI/ML papers? What does it supersede?
        What does it complement? Are there surprising connections to other subfields?

        ## 3. Practical Deployment Considerations
        Real-world tradeoffs: latency, memory, data requirements, failure modes in production.

        ## 4. Critical Assessment
        Evaluate the paper's claims critically:
          - Is the experimental setup fair and reproducible?
          - Are there cherry-picked baselines?
          - Do the ablations actually support the claimed conclusions?

        ## 5. Surprising or Underappreciated Insights
        What does the paper imply but not say explicitly? What would a careful reader notice
        that a casual reader would miss?

        ## 6. One-Paragraph Pitch & One-Paragraph Critique
        Steelman the paper in one paragraph. Then write the strongest possible critique in one paragraph."""),
}

DIAGRAM_PROMPT: str = textwrap.dedent("""\
    Generate exactly 6 Graphviz DOT diagrams that illuminate this AI/ML paper from 6 different angles:

      Diagram 1 — High-Level Architecture / System Overview
      Diagram 2 — Data Flow & Processing Pipeline
      Diagram 3 — Core Algorithm as a Flowchart
      Diagram 4 — Concept Taxonomy / Knowledge Hierarchy
      Diagram 5 — Training Loop / Optimisation Dynamics
      Diagram 6 — Comparison vs Prior Art (or Ablation Structure)

    ══ MANDATORY VISUAL STYLE (apply to EVERY diagram) ══
    Include these exact two statements, verbatim, as the first two lines inside
    every "digraph G {" body (this is real DOT syntax — copy it exactly,
    do not paraphrase or restructure it):
      bgcolor="black";
      node [style=filled, fillcolor="#0a0a0a", fontname="Courier New", fontsize=11];

      Use NEON accent colours for borders, labels, and edges. Pick from:
        Electric Green  #00FF41    Hot Magenta  #FF00FF    Cyan      #00FFFF
        Neon Orange     #FF6600    Volt Yellow  #FFFF00    Hot Pink  #FF0055
        Chartreuse      #7FFF00    Electric Blue #0080FF   Lavender  #DA70FF
      Edges: penwidth=2.0, use neon colours (vary per diagram)
      Graph titles: set label="..." and labelloc=t and a bright fontcolor="..."
        as attributes of the digraph itself (not inside a subgraph/cluster).
      Mix rankdir=LR and rankdir=TB between diagrams for variety.
      Every line must be valid, renderable Graphviz DOT syntax — no
      pseudo-syntax, no descriptive labels standing in for real statements.

    ══ OUTPUT FORMAT — strictly follow this delimiter pattern ══
    ===DIAGRAM_START: <Descriptive Title for Diagram N>===
    digraph G {
      // ... full valid DOT source ...
    }
    ===DIAGRAM_END===

    Output ONLY the 6 delimited DOT blocks. No prose before, between, or after.""")
