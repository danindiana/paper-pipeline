"""
__main__.py — CLI entry point for the paper processing pipeline.

Usage:
  python -m paper_pipeline [DIR]                          # process all PDFs
  python -m paper_pipeline [DIR] --paper foo.pdf          # single paper
  python -m paper_pipeline [DIR] --model MODEL            # force model
  python -m paper_pipeline [DIR] --reprocess diagrams     # redo one section
  python -m paper_pipeline [DIR] --list                   # show status table
  python -m paper_pipeline [DIR] --workers 2              # parallel
  python -m paper_pipeline [DIR] --override               # aggressive GPU clear
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import textwrap
import threading
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from . import config, store
from .errors import LeaseLostError, OllamaUnavailable, ShutdownRequested
from .migrations import LegacyDatabaseError
from .ollama import OllamaClient, OllamaVRAM
from .pipeline import PIPELINE, PaperProcessor, PaperStatus
from .reader import quick_page_count

# ── Graceful shutdown ────────────────────────────────────────────────────────
_shutdown = threading.Event()


def _install_signal_handlers() -> None:
    def _handler(signum, frame):
        if _shutdown.is_set():
            print("\n  🚨  Forced exit — stopping immediately!")
            os._exit(1)
        print("\n\n  ⚡  Shutdown requested — finishing current section …")
        print("      (Ctrl+C again to force)")
        _shutdown.set()
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# ── Status listing ───────────────────────────────────────────────────────────

def _list_status(papers_dir: Path, db_path: Path) -> None:
    conn = store.connect(db_path)
    pdfs = sorted(p for p in papers_dir.rglob("*.pdf") if "_processed" not in p.parts)
    if not pdfs:
        print(f"  No PDFs found in {papers_dir}")
        return

    print(f"\n  {'Paper':<60}  Status")
    print(f"  {'─'*60}  {'─'*30}")
    for pdf in pdfs:
        record = store.load_paper_by_pdf_path(conn, str(pdf))
        if record is None:
            status = "⬜  not started"
        elif store.ALL_SECTIONS.issubset(set(record.sections_completed)):
            status = f"✅  complete  [{record.model_used}]"
        else:
            missing = store.ALL_SECTIONS - set(record.sections_completed)
            status = f"⚠️   partial — missing: {', '.join(sorted(missing))}"
        rel = pdf.relative_to(papers_dir).as_posix()
        name = rel[:58] + "…" if len(rel) > 58 else rel
        print(f"  {name:<60}  {status}")
    print()
    conn.close()


# ── Batch tier-sorting ───────────────────────────────────────────────────────

def _sort_by_tier(pdfs: list[Path]) -> list[Path]:
    """Sort PDFs by model tier so serial processing minimises GPU swaps."""
    return sorted(pdfs, key=lambda p: config.select_model(quick_page_count(p)))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="paper_pipeline",
        description="📄  AI/ML Paper Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f"""\
            Model auto-selection by page count:
              ≤ 35 pages  → {config.MODEL_TIERS['default']}
              36–200      → {config.MODEL_TIERS['reasoning']}
              > 200       → {config.MODEL_TIERS['default']} (stays GPU-resident)

            Output is stored in a shared SQLite database keyed by paper content hash.
            Default: {config.DEFAULT_DB_PATH}
            Override: --db-path PATH  or  {config.DB_PATH_ENV_VAR} env var

            Pointing this at a new/different folder:
              Directory scanning is RECURSIVE by default (all *.pdf under DIR,
              any depth) — run with --list first to see real scope before a
              full run, especially on a folder tree rather than a flat one.

              Papers are keyed by content hash, not by folder path, so you
              choose: reuse the same/default database to add the new folder's
              papers into one shared corpus (identical files are skipped
              automatically, no collision risk), or pass a different
              --db-path to keep the new folder's results fully separate.

            Checking on a run / seeing what's in the database:
              paper-pipeline-tui DIR [--db-path PATH]      interactive dashboard
              ./scripts/monitor.sh DIR [--db-path PATH]    scriptable/tmux status line
              python3 scripts/instances.py                 what's running, from where, on this machine
              ./scripts/graph_viz.sh import --db-path PATH re-sync the optional Neo4j/cosmos.gl evidence graph
        """),
    )
    ap.add_argument(
        "papers_dir", nargs="?", default=None,
        help=f"Directory containing PDF papers [default: {config.DEFAULT_PAPERS_DIR}]",
    )
    ap.add_argument("--model", default=None, metavar="MODEL",
                    help="Force a specific model for all sections")
    ap.add_argument("--code-model", default=None, metavar="MODEL",
                    help="Force a specific model for C++ sections only")
    ap.add_argument("--paper", default=None, metavar="FILENAME",
                    help="Process a single paper by filename")
    ap.add_argument("--workers", type=int, default=1, metavar="N",
                    help="Parallel workers [default: 1]")
    ap.add_argument("--list", action="store_true",
                    help="Show processing status and exit")
    ap.add_argument("--reprocess", default=None, metavar="SECTION",
                    choices=["summary", "logic", "cpp", "diagrams", "extras", "all"],
                    help="Re-run a specific section (or 'all')")
    ap.add_argument("--override", action="store_true",
                    help="Aggressively clear GPU VRAM before processing")
    ap.add_argument("--no-sort", action="store_true",
                    help="Skip batch tier-sorting (process in filesystem order)")
    ap.add_argument("--db-path", default=None, metavar="PATH",
                    help=f"SQLite database path [default: {config.DEFAULT_DB_PATH}]")
    ap.add_argument("--ocr", choices=["auto", "always", "never"], default="auto",
                    metavar="MODE", help="OCR mode [default: auto]")
    ap.add_argument("--ocr-dpi", type=int, default=300, metavar="DPI",
                    help="Tesseract rasterisation DPI [default: 300]")
    ap.add_argument("--ocr-lang", default="eng", metavar="LANG",
                    help="Tesseract language pack(s) [default: eng]")
    ap.add_argument("--ocr-max-pages", type=int, default=config.DEFAULT_OCR_MAX_PAGES,
                    metavar="N", help="Max pages to OCR per document (0=unlimited)")
    args = ap.parse_args()

    if args.workers < 1:
        ap.error("--workers must be >= 1")

    # ── Resolve paths ────────────────────────────────────────────────────
    papers_dir = Path(os.path.expandvars(args.papers_dir or str(config.DEFAULT_PAPERS_DIR))).expanduser()
    if not papers_dir.is_dir():
        sys.exit(f"❌  Directory not found: {papers_dir}")

    db_path = store.resolve_db_path(args.db_path)

    # ── Pre-initialize database (run migrations once before workers) ─────
    try:
        store.init_db(db_path)
    except LegacyDatabaseError as exc:
        sys.exit(f"❌  {exc}")

    # ── List mode ────────────────────────────────────────────────────────
    if args.list:
        _list_status(papers_dir, db_path)
        return

    _install_signal_handlers()

    # ── Health check ─────────────────────────────────────────────────────
    print(f"\n  📄  Paper Processing Pipeline")
    print(f"  {'─'*40}")

    vram = OllamaVRAM()
    client = OllamaClient(vram, _shutdown)

    if not client.health_check():
        sys.exit(1)

    # Verify required models are pulled
    models_needed = (
        [args.model] if args.model
        else list(dict.fromkeys(config.MODEL_TIERS.values()))
    )
    if args.code_model:
        models_needed = list(dict.fromkeys(models_needed + [args.code_model]))
    try:
        client.check_required_models(models_needed)
    except OllamaUnavailable as exc:
        sys.exit(f"❌  {exc}")

    if args.override:
        vram.provision()

    # ── Build file list ──────────────────────────────────────────────────
    if args.paper:
        target = papers_dir / args.paper
        if not target.exists():
            matches = [p for p in papers_dir.rglob(args.paper) if p.is_file()]
            if len(matches) == 1:
                target = matches[0]
            elif len(matches) > 1:
                sys.exit(
                    f"❌  Ambiguous '{args.paper}' — {len(matches)} matches:\n    "
                    + "\n    ".join(str(m.relative_to(papers_dir)) for m in matches[:10])
                )
            else:
                sys.exit(f"❌  File not found: {target}")
        pdfs = [target]
    else:
        pdfs = sorted(
            p for p in papers_dir.rglob("*.pdf") if "_processed" not in p.parts
        )
        if not pdfs:
            sys.exit(f"❌  No PDF files found in {papers_dir}")

        # Tier-sort to minimise GPU swaps in serial mode
        if (
            not args.no_sort
            and not args.model
            and args.workers == 1
            and len(pdfs) > 1
        ):
            print("  🔀  Sorting batch by model tier …")
            pdfs = _sort_by_tier(pdfs)

    print(f"  Directory : {papers_dir}")
    print(f"  Database  : {db_path}")
    print(f"  Papers    : {len(pdfs)}")
    if args.model:
        print(f"  Model     : {args.model}")
    if args.reprocess:
        print(f"  Reprocess : {args.reprocess}")
    print()

    # ── Build processor ──────────────────────────────────────────────────
    processor = PaperProcessor(
        client=client,
        shutdown=_shutdown,
        db_path=db_path,
        forced_model=args.model,
        forced_code_model=args.code_model,
        reprocess=args.reprocess,
        ocr_mode=args.ocr,
        ocr_dpi=args.ocr_dpi,
        ocr_lang=args.ocr_lang,
        ocr_max_pages=args.ocr_max_pages or None,
    )

    # ── Process ──────────────────────────────────────────────────────────
    results: Counter[PaperStatus] = Counter()

    if args.workers > 1:
        print(f"  ⚡ Parallel mode: {args.workers} workers\n")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(processor.process, pdf): pdf for pdf in pdfs}
            remaining = set(futures)
            while remaining:
                done, remaining = wait(remaining, timeout=2.0, return_when=FIRST_COMPLETED)
                for fut in done:
                    pdf = futures[fut]
                    try:
                        status = fut.result()
                        results[status] += 1
                    except ShutdownRequested:
                        results[PaperStatus.INTERRUPTED] += 1
                    except OllamaUnavailable as exc:
                        print(f"  ❌  {pdf.name}: {exc}")
                        results[PaperStatus.FAILED] += 1
                    except Exception as exc:
                        print(f"  ❌  {pdf.name}: {exc}")
                        results[PaperStatus.FAILED] += 1
                        if "timed out" in str(exc).lower():
                            print("  🔄  Timeout — restarting Ollama …")
                            vram.restart_service()
                if _shutdown.is_set():
                    cancelled = len(remaining)
                    print(f"  ⚡  Cancelling {cancelled} pending paper(s) …")
                    for f in remaining:
                        f.cancel()
                    results[PaperStatus.INTERRUPTED] += cancelled
                    break
    else:
        for pdf in pdfs:
            if _shutdown.is_set():
                remaining_count = len(pdfs) - pdfs.index(pdf)
                results[PaperStatus.INTERRUPTED] += remaining_count
                print(f"  ⚡  Shutdown — skipping remaining {remaining_count} papers")
                break
            try:
                status = processor.process(pdf)
                results[status] += 1
            except ShutdownRequested:
                results[PaperStatus.INTERRUPTED] += 1
                break
            except OllamaUnavailable as exc:
                print(f"  ❌  {pdf.name}: {exc}")
                results[PaperStatus.FAILED] += 1
                break  # no point continuing if Ollama is unreachable
            except Exception as exc:
                print(f"  ❌  {pdf.name}: {exc}")
                results[PaperStatus.FAILED] += 1
                if "timed out" in str(exc).lower():
                    print("  🔄  Timeout — restarting Ollama …")
                    vram.restart_service()

    # ── Summary ──────────────────────────────────────────────────────────
    total = sum(results.values())
    print(f"\n{'═'*64}")
    print(f"  Results ({total} papers):")
    for status in PaperStatus:
        count = results.get(status, 0)
        if count:
            icon = {
                PaperStatus.COMPLETE: "✅",
                PaperStatus.PARTIAL: "⚠️ ",
                PaperStatus.ALREADY_COMPLETE: "⏭ ",
                PaperStatus.CLAIMED_ELSEWHERE: "🔒",
                PaperStatus.INTERRUPTED: "⚡",
                PaperStatus.FAILED: "❌",
            }[status]
            print(f"    {icon}  {status.name:<20} {count}")
    print(f"{'═'*64}\n")


if __name__ == "__main__":
    main()
