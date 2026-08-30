# CLI Howto — Cold-Start Operator Runbook

This is the detailed, from-scratch guide for setting up and running
`paper-pipeline` on a machine that has nothing installed yet. The
[README](README.md)'s "Getting Started" section is the terse version; this is
the one with the troubleshooting, the "why", and the things that bit the
maintainer during actual runs.

## 1. Prerequisites

* **OS**: Ubuntu/Debian assumed below (apt commands given — the CI target and
  what this was developed on). Other distros: translate the package names.
* **Python ≥ 3.11** — see the interpreter-probe note in
  [Troubleshooting](#9-troubleshooting-cold-start-issues) below before assuming
  your system's default `python3` is fine.
* **A local [Ollama](https://ollama.com) install**, running and reachable.
* **GPU**: strongly recommended but not required — this runs on CPU, just much
  slower. Developed against a dual-NVIDIA setup (RTX 5080 16GB + RTX 3080
  10GB, ~26.5GB combined VRAM); both bundled model tiers together need
  roughly that much headroom if you want zero eviction thrashing (see
  [`diagrams/04_gpu_disk_io`](diagrams/04_gpu_disk_io.svg)).
* **Disk space**: the SQLite database is the only persistent state and it's
  small — roughly 80KB per fully-processed paper observed in practice
  (~15MB for ~190 papers during a real overnight batch). The PDFs themselves
  are whatever they are; nothing else gets duplicated to disk.

## 2. Quickstart

```bash
git clone https://github.com/danindiana/paper-pipeline.git
cd paper-pipeline
./scripts/setup.sh
```

This installs system packages (`graphviz`, `tesseract-ocr`), finds a working
Python interpreter, creates `.venv/`, installs the package, pulls both model
tiers, and runs the test suite. It's idempotent — safe to re-run if something
fails partway and you fix the underlying issue.

**Before trusting a full batch, smoke-test one paper:**

```bash
source .venv/bin/activate
paper-pipeline /path/to/pdfs --paper some_short_paper.pdf
```

This isn't optional caution for its own sake — during this project's own
development, a full-batch run on an untested config would have burned hours
rediscovering the same bug on paper after paper before anyone noticed. A
single smoke-test run surfaces model-availability issues, GPU/VRAM problems,
and config mistakes in a couple of minutes instead.

## 3. Manual step-by-step (if you'd rather not run a script blindly)

```bash
# System packages
sudo apt-get update
sudo apt-get install -y graphviz tesseract-ocr tesseract-ocr-eng

# Python venv — use an interpreter you've confirmed has a working venv module
# (see Troubleshooting #1 below)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest

# Ollama models
ollama pull gemma4:26b-a4b-it-q4_K_M   # fast tier
ollama pull qwen3.6:35b                # reasoning tier

# Verify
python -m pytest tests/ -v
```

## 4. Running a batch

```bash
paper-pipeline /path/to/pdfs
```

A few things worth knowing before you do this at scale:

* **Leave `--workers` at 1** (the default) unless you've read
  [`diagrams/02_catch22s`](diagrams/02_catch22s.svg). The Ollama client holds
  a process-wide lock across every model load and generation, so raising
  `--workers` doesn't parallelize the actual LLM calls — it only overlaps PDF
  text extraction, at the cost of more SQLite/lease contention. Benchmark
  before assuming more workers helps.
* **The batch is tier-sorted by default** (`--no-sort` disables this) to
  minimize GPU model swaps — but that means every fast-tier paper (≤35 or
  >200 pages) processes before *any* reasoning-tier paper (36–200 pages)
  starts, since papers are sorted by the literal model-tag string. **Don't
  extrapolate an ETA from early throughput** if your corpus mixes both tiers —
  observed directly on a 308-paper batch: ~14 papers/hour while 100% fast-tier,
  then the remaining reasoning-tier papers (measurably slower per paper, due
  to that model's own internal "thinking" overhead) hadn't even started after
  14 hours.
* **Database location**: defaults to a path baked into `config.py` for this
  project's own development machine — override it with `--db-path PATH` or
  the `PAPER_PROCESSOR_DB` environment variable for your own setup.

## 5. Observability while it runs

**Primary tool:**

```bash
./scripts/monitor.sh /path/to/pdfs --watch
```

Shows the currently-loaded Ollama model, per-GPU utilization/memory, batch
progress (complete / partial / not-started), and a "possibly stuck" heuristic
based on how long the process has run with near-zero GPU utilization. If
the optional evidence graph viewer (§9) is running, its status shows here
too — otherwise that line is omitted entirely.

**Manual alternatives**, if you'd rather not use the script:

| Check | Command |
|---|---|
| Loaded model / VRAM | `curl -s localhost:11434/api/ps \| python3 -m json.tool` |
| GPU utilization | `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv` |
| Batch progress | `paper-pipeline /path/to/pdfs --list` |
| Process alive | `pgrep -fa paper-pipeline` |
| Ollama service logs | `journalctl -u ollama --since "10 min ago"` |

**A real gotcha worth knowing about `tail -f` on a redirected log:** if you
run the batch as `paper-pipeline ... > run.log 2>&1 &`, Python fully buffers
stdout when it's piped to a file (not a tty) — `run.log` can sit completely
empty for a long stretch even while the batch is actively working, then
suddenly fill up all at once. **An empty or stale-looking log file is not
evidence of a hang** — check `--list`, `pgrep`, or GPU utilization instead,
not just log file size/mtime.

## 6. Recovery

If a batch stops (crashed, killed, machine rebooted), **just rerun the exact
same command**:

```bash
paper-pipeline /path/to/pdfs
```

Completed papers are skipped (content-hash keyed, persisted in the SQLite
DB); anything incomplete — including papers that failed outright — gets
retried automatically. **Do not pass `--reprocess all`** for ordinary
recovery — that flag forces every paper to redo every section, discarding
work that already succeeded. Reserve `--reprocess SECTION` for deliberately
redoing one section of one specific paper via `--paper filename.pdf`.

## 7. What failures look like, and what to do about them

Three failure shapes actually observed running this at scale, in
operator-facing terms:

* **"Ollama returned an empty response after reload retry"** — the model
  generated nothing usable for one chunk. The pipeline already retried once
  with a forced model reload before surfacing this. Rare after the
  context-budget fix shipped in this repo's history; if you see it
  frequently, it may mean your GPU doesn't have enough VRAM headroom for the
  configured context size.
* **"chunk N evidence extraction failed: all proposed evidence failed source
  verification"** — the model's proposed evidence didn't survive the
  verbatim-quote check for an entire chunk, on both attempts. This is
  sampling variance, not a bug — a plain rerun (see [Recovery](#6-recovery))
  frequently succeeds on the same paper with fresh sampling.
* **A CUDA/GPU error mid-run** (e.g. `CUDA error: the launch timed out and
  was terminated`) — the pipeline detects `"timed out"` in the error text,
  automatically restarts the Ollama service, confirms it's back up, and
  continues the batch. No action needed unless this recurs repeatedly, which
  would point at a hardware/driver issue rather than this tool.

## 8. Diagram rendering

A small fraction of Graphviz diagrams the model generates will fail to
render to SVG — this is non-fatal by design (the `.dot` source is always
saved even when rendering fails, so it never aborts a paper). One common
cause has an automatic repair (see
[`diagrams/05_future_directions`](diagrams/05_future_directions.svg)); the
rest are a more varied, harder problem that's documented rather than hidden.

## 9. Evidence Graph Visualization (optional)

Every processed paper's evidence — the verified items its summary/logic/cpp
sections were actually generated from, plus how they relate to each other —
can be rendered as an interactive WebGL graph via
[`neo4j_viz/`](neo4j_viz), a self-contained Neo4j + cosmos.gl viewer that's
entirely separate from the core pipeline (no new required dependency,
nothing here is imported by `paper_pipeline` itself).

```bash
pip install neo4j   # once
./scripts/graph_viz.sh start --db-path /path/to/papers.db
# open http://localhost:8687/
```

`./scripts/graph_viz.sh status` shows what's running; `stop` tears it down;
`import --db-path PATH` re-syncs after a batch has processed more papers
(safe to re-run any time — every write is a database MERGE). Requires
Docker. Full detail, including a real cross-project Docker incident and how
it's now prevented, in [`neo4j_viz/README.md`](neo4j_viz/README.md).

## 10. Troubleshooting cold-start issues

1. **`python3.X -m venv` "succeeds" but `pip` inside it is broken/missing** —
   some systems ship a Python whose `ensurepip` silently falls back to
   user-site packages instead of installing into the venv, instead of
   failing loudly. Symptom: `pip install -e .` reports something like
   "requires a different Python" or the venv's `pip` doesn't exist at all.
   Fix: `sudo apt install python3.11 python3.11-venv` and use that
   interpreter explicitly (`python3.11 -m venv .venv`) rather than whatever
   `python3` happens to resolve to. `scripts/setup.sh` probes for this
   automatically and picks a working interpreter.
2. **`dot: command not found` / diagrams never render** —
   `sudo apt install graphviz`.
3. **OCR silently does nothing on image-only PDFs** —
   `sudo apt install tesseract-ocr tesseract-ocr-eng`.
4. **`Cannot reach Ollama at http://localhost:11434`** — confirm the service
   is actually running: `systemctl status ollama` or `ollama serve` in a
   separate terminal; check `OLLAMA_URL` if you're running it elsewhere.
5. **`CUDA error: out of memory` on a fresh, otherwise-idle GPU** — can be
   transient (a model mid-unload from a previous run); try again, and if it
   recurs, check `nvidia-smi` for anything else holding VRAM, or restart the
   Ollama service.

## 11. Where to go next

The [README](README.md#system-diagrams--architecture)'s diagrams explain the
*why* behind the guidance above in more depth — particularly
[`02_catch22s`](diagrams/02_catch22s.svg) (the `--workers` and context-budget
tensions) and [`05_future_directions`](diagrams/05_future_directions.svg)
(known open problems, including the diagram-rendering long tail).
