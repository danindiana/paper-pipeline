#!/usr/bin/env python3
"""
cosmos_server.py — serves the CosmosGL evidence-graph viewer.

Pure stdlib (no Flask) — matches paper-pipeline's own minimal-dependency
ethos. Neo4j credentials stay server-side; the browser only ever talks to
this process over plain HTTP/JSON, same separation as the sibling
lobster-graph project's cosmos_server.py.

Usage:
    python3 cosmos_server.py [--port 8687] [--neo4j-uri bolt://localhost:7688]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import neo4j

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

NODE_ID_RE = re.compile(r"^/api/node/(\d+)$")


def make_handler(neo4j_uri: str, neo4j_user: str, neo4j_password: str):
    def run_query(query, **params):
        driver = neo4j.GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        try:
            with driver.session() as session:
                return [dict(r) for r in session.run(query, **params)]
        finally:
            driver.close()

    def fetch_graph():
        edges = run_query(
            "MATCH (s)-[r]->(t) RETURN id(s) AS source, id(t) AS target, type(r) AS type"
        )
        node_ids = list({e["source"] for e in edges} | {e["target"] for e in edges})
        if not node_ids:
            nodes = run_query(
                "MATCH (n) RETURN id(n) AS id, labels(n)[0] AS type, "
                "coalesce(n.name, n.title, n.evidence_id, toString(id(n))) AS label"
            )
        else:
            nodes = run_query(
                "MATCH (n) WHERE id(n) IN $ids RETURN id(n) AS id, labels(n)[0] AS type, "
                "coalesce(n.name, n.title, n.evidence_id, toString(id(n))) AS label",
                ids=node_ids,
            )
        return {"nodes": nodes, "edges": edges}

    def fetch_node_detail(node_id: int):
        rows = run_query(
            "MATCH (n) WHERE id(n) = $id RETURN labels(n)[0] AS type, properties(n) AS props",
            id=node_id,
        )
        return rows[0] if rows else None

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=STATIC_DIR, **kwargs)

        def _send_json(self, status, obj):
            payload = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = self.path.split("?", 1)[0]

            if path == "/api/graph":
                try:
                    self._send_json(200, fetch_graph())
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})
                return

            m = NODE_ID_RE.match(path)
            if m:
                try:
                    detail = fetch_node_detail(int(m.group(1)))
                    if detail is None:
                        self._send_json(404, {"error": "node not found"})
                    else:
                        self._send_json(200, detail)
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})
                return

            if path == "/":
                self.path = "/index.html"
            super().do_GET()

        def log_message(self, fmt, *args):
            pass  # keep stdout quiet; errors still surface via _send_json

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8687)
    ap.add_argument("--neo4j-uri", default="bolt://localhost:7688")
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--neo4j-password", default="paperpipeline")
    args = ap.parse_args()

    handler = make_handler(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Serving evidence graph viewer at http://127.0.0.1:{args.port}")
    print(f"Neo4j: {args.neo4j_uri}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
