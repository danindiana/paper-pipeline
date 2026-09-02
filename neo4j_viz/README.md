# Evidence Graph Visualization (optional)

> **Frozen.** Active development of this viewer has moved to
> [`cosmosgl-evidence-graph`](https://github.com/danindiana/cosmosgl-evidence-graph),
> a standalone repo with a more complete README, HOWTO, and architecture
> diagrams. This copy still works as documented below, but new CosmosGL
> functionality lands there, not here.

Renders the evidence graph behind your processed papers — every verified
evidence item, how items relate to each other, and which diagrams came from
which paper — as an interactive WebGL force graph in the browser, via
[Neo4j](https://neo4j.com) and [cosmos.gl](https://cosmos.gl).

This is entirely optional and lives outside the core `paper_pipeline`
package: nothing here is imported by, or required to run, the main CLI.

## What gets graphed

Real, already-persisted data pulled straight from paper-pipeline's own
SQLite database (`papers.source_corpus` — the same evidence bundle each
paper's summary/logic/cpp/diagrams sections were generated from):

```
(:Paper {paper_hash, name, model_used, page_count, processed_at})
(:Evidence {qid, evidence_id, kind, statement, support, page, paper_hash})
(:Diagram {title, idx, rendered})

(Paper)-[:HAS_EVIDENCE]->(Evidence)
(Evidence)-[:RELATES_TO {description}]->(Evidence)
(Paper)-[:HAS_DIAGRAM]->(Diagram)
```

Nothing is fabricated or re-derived — this is the exact evidence that
backed each paper's verified, non-hallucinated output.

## Prerequisites

* Docker + Docker Compose
* Python: `pip install neo4j` (not a core dependency of `paper_pipeline`)
* Only if you want to *rebuild* the bundled JS (not needed to just run this):
  Node.js ≥22 + npm

## Setup

**Easiest: the wrapper script**, from the repo root:

```bash
./scripts/graph_viz.sh start --db-path /path/to/your/papers.db
./scripts/graph_viz.sh status   # what's running
./scripts/graph_viz.sh stop     # tear it down
```

`import --db-path PATH` (a subcommand of the same script) re-syncs later
without restarting anything — safe to re-run any time, every write is a
database MERGE. `scripts/monitor.sh` picks up this stack's status
automatically once it's running.

**Manual, if you'd rather see each step:**

```bash
cd neo4j_viz

# 1. Start a dedicated Neo4j instance (ports 7475/7688 -- deliberately
#    different from any other Neo4j project you might have running; see
#    "A real incident" below for why this matters).
docker compose up -d

# 2. Import your processed papers into it.
python3 import_to_neo4j.py --db-path /path/to/your/papers.db

# 3. Serve the viewer.
python3 cosmos_server.py
```

Then open **http://localhost:8687/** in a real browser (not headless — see
below). Hover a node for its type/label, click it for full properties
(clicking a `Paper` node also opens its source PDF if it's still on disk at
its recorded path).

## Stopping / cleaning up

```bash
./scripts/graph_viz.sh stop   # from the repo root -- stops cosmos_server.py + Neo4j
# or, manually:
docker compose down           # stops and removes the Neo4j container
# ./data/ (the actual database files) is left on disk -- delete it manually
# if you want a truly clean slate before the next start.
```

## Rebuilding the vendored bundle

`static/cosmos_bundle.js` is a committed, pre-built artifact (esbuild IIFE
bundle of `@cosmos.gl/graph`) so running the viewer never requires Node/npm.
Only rebuild it if you want to bump the cosmos.gl version:

```bash
cd cosmos_build
npm install
npm run build   # writes ../static/cosmos_bundle.js
```

## A real incident, and why the Compose project has an explicit `name:`

Building this the first time, `docker compose up -d` from this directory
**destroyed a running Neo4j container belonging to an unrelated project** on
the same machine — both projects' compose files happened to live in a
directory also named `neo4j_viz`, and Docker Compose derives its default
project name from the directory basename, not `container_name`. It treated
the two `neo4j` services as the same tracked resource and replaced one with
the other. No data was lost (the other project's bind-mounted data survived
on disk and the container was restarted from it), but it's exactly the kind
of mistake that's easy to repeat. `docker-compose.yml` in this directory now
pins an explicit top-level `name:` for this reason — if you copy this file
elsewhere, keep that line.

## Why real-browser testing, not headless

If you're scripting this (CI, automated smoke tests), be aware that
headless Chrome with software WebGL (swiftshader) appears to hang
indefinitely when combined with `--screenshot`'s pixel-readback on a graph
this size, even though the exact same page renders correctly and quickly in
a normal GPU-accelerated browser. This isn't a bug in the viewer — verify
interactively in a real browser, or drive it with a tool that doesn't rely
on headless screenshot capture.
