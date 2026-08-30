#!/usr/bin/env python3
"""
import_to_neo4j.py — load paper-pipeline's evidence graph into Neo4j.

Reads directly from the paper-pipeline SQLite database (read-only, never
writes back) and populates a small, real graph:

    (:Paper {paper_hash, name, model_used, page_count, processed_at})
    (:Evidence {qid, evidence_id, kind, statement, support, page, paper_hash})
    (:Diagram {title, idx, rendered})

`evidence_id` (e.g. "C001-E001") is only unique within one paper's own
evidence bundle -- every paper's first chunk starts back at C001-E001. The
graph-unique merge key is `qid` = f"{paper_hash}:{evidence_id}".

    (Paper)-[:HAS_EVIDENCE]->(Evidence)
    (Evidence)-[:RELATES_TO {description}]->(Evidence)
    (Paper)-[:HAS_DIAGRAM]->(Diagram)

All evidence data comes from `papers.source_corpus` — the JSON-serialized
EvidenceBundle each paper's five output sections were actually generated
from (see paper_pipeline/reader.py: EvidenceBundle.to_json /
pipeline.py: upsert_meta_and_corpus_fenced). This is real, persisted data,
not derived/regenerated — the same evidence records that back the paper's
summary/logic/cpp/diagrams/extras output.

Idempotent: every write is a MERGE, safe to re-run against a growing corpus.

Usage:
    python3 import_to_neo4j.py [--db-path PATH] [--neo4j-uri URI]
                                [--neo4j-user USER] [--neo4j-password PASS]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# Imported lazily inside main() rather than at module level: import_paper()
# below takes a driver session as a plain argument and never touches the
# `neo4j` package directly, so the rest of this module (and its logic) stays
# importable/testable without the optional dependency installed -- only
# actually running the CLI requires it.


def load_papers(db_path: Path):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT paper_hash, paper_name, model_used, page_count,
               processed_at, source_corpus
        FROM papers
        WHERE source_corpus IS NOT NULL
        """
    ).fetchall()
    diagrams_by_paper: dict[str, list[sqlite3.Row]] = {}
    for row in conn.execute("SELECT paper_hash, idx, title, svg_content FROM diagrams"):
        diagrams_by_paper.setdefault(row["paper_hash"], []).append(row)
    conn.close()
    return rows, diagrams_by_paper


def import_paper(session, paper_row, diagram_rows) -> tuple[int, int]:
    paper_hash = paper_row["paper_hash"]
    try:
        corpus = json.loads(paper_row["source_corpus"])
    except (json.JSONDecodeError, TypeError):
        return 0, 0

    session.run(
        """
        MERGE (p:Paper {paper_hash: $paper_hash})
        SET p.name = $name, p.model_used = $model_used,
            p.page_count = $page_count, p.processed_at = $processed_at
        """,
        paper_hash=paper_hash,
        name=paper_row["paper_name"],
        model_used=paper_row["model_used"],
        page_count=paper_row["page_count"],
        processed_at=paper_row["processed_at"],
    )

    # `evidence_id` (e.g. "C001-E001") is only unique *within* one paper's
    # evidence bundle -- every paper's first chunk starts back at C001-E001,
    # so MERGE-ing on that alone collapses unrelated papers' evidence into
    # shared nodes. The graph-unique key is (paper_hash, evidence_id).
    n_evidence = 0
    for chunk in corpus.get("chunks", []):
        for item in chunk.get("evidence", []):
            qid = f"{paper_hash}:{item['evidence_id']}"
            session.run(
                """
                MATCH (p:Paper {paper_hash: $paper_hash})
                MERGE (e:Evidence {qid: $qid})
                SET e.evidence_id = $evidence_id, e.kind = $kind,
                    e.statement = $statement, e.support = $support,
                    e.page = $page, e.paper_hash = $paper_hash
                MERGE (p)-[:HAS_EVIDENCE]->(e)
                """,
                paper_hash=paper_hash,
                qid=qid,
                evidence_id=item["evidence_id"],
                kind=item["kind"],
                statement=item["statement"][:500],
                support=item["support"][:500],
                page=item["page"],
            )
            n_evidence += 1

    n_rel = 0
    relationships = corpus.get("root", {}).get("relationships", [])
    for rel in relationships:
        ids = rel.get("evidence_ids", [])
        description = rel.get("description", "")
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                qid_a = f"{paper_hash}:{a}"
                qid_b = f"{paper_hash}:{b}"
                session.run(
                    """
                    MATCH (a:Evidence {qid: $qid_a}), (b:Evidence {qid: $qid_b})
                    MERGE (a)-[r:RELATES_TO]->(b)
                    SET r.description = $description
                    """,
                    qid_a=qid_a, qid_b=qid_b, description=description[:500],
                )
                n_rel += 1

    for d in diagram_rows:
        session.run(
            """
            MATCH (p:Paper {paper_hash: $paper_hash})
            MERGE (g:Diagram {paper_hash: $paper_hash, idx: $idx})
            SET g.title = $title, g.rendered = $rendered
            MERGE (p)-[:HAS_DIAGRAM]->(g)
            """,
            paper_hash=paper_hash,
            idx=d["idx"],
            title=d["title"],
            rendered=d["svg_content"] is not None,
        )

    return n_evidence, n_rel


def main() -> None:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        sys.exit(
            "ERROR: the 'neo4j' package is required for this optional feature.\n"
            "  pip install neo4j"
        )

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db-path", type=Path, required=True,
                     help="Path to paper-pipeline's SQLite database")
    ap.add_argument("--neo4j-uri", default="bolt://localhost:7688")
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--neo4j-password", default="paperpipeline")
    args = ap.parse_args()

    if not args.db_path.is_file():
        sys.exit(f"ERROR: database not found: {args.db_path}")

    papers, diagrams_by_paper = load_papers(args.db_path)
    print(f"Found {len(papers)} paper(s) with a stored evidence corpus.")

    driver = GraphDatabase.driver(
        args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password)
    )
    try:
        with driver.session() as session:
            session.run("CREATE INDEX paper_hash_idx IF NOT EXISTS FOR (p:Paper) ON (p.paper_hash)")
            session.run("CREATE INDEX evidence_qid_idx IF NOT EXISTS FOR (e:Evidence) ON (e.qid)")

        total_evidence = 0
        total_rel = 0
        for row in papers:
            with driver.session() as session:
                n_ev, n_rel = import_paper(session, row, diagrams_by_paper.get(row["paper_hash"], []))
            total_evidence += n_ev
            total_rel += n_rel
            print(f"  {row['paper_name']}: {n_ev} evidence items, {n_rel} relationships")

        print(f"\nDone. {len(papers)} papers, {total_evidence} evidence items, "
              f"{total_rel} relationships imported/updated.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
