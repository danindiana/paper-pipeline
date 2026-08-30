"""
pipeline.py — Data-driven paper processing pipeline with fenced writes.

Every database mutation uses a typed fenced operation from store.py.
There is no generic callback API — the transaction boundary is never
exposed to pipeline code.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from threading import Event
from typing import Callable, Optional, Protocol

from . import config, store
from .diagrams import ensure_neon_black, parse_diagrams, render_dot
from .errors import LeaseLostError, ShutdownRequested
from .prompts import DIAGRAM_PROMPT, PROMPTS
from .reader import (
    EvidenceBundle,
    EvidenceSynthesisError,
    build_chunks,
    extract_pages,
    hash_file,
    synthesize_evidence,
)


class PaperStatus(Enum):
    COMPLETE = auto()
    PARTIAL = auto()
    ALREADY_COMPLETE = auto()
    CLAIMED_ELSEWHERE = auto()
    INTERRUPTED = auto()
    FAILED = auto()


class GenerationClient(Protocol):
    def generate(
        self, model: str, prompt: str, ctx_tokens: Optional[int] = None
    ) -> str: ...


MIN_DIAGRAMS_FOR_COMPLETE = 1

@dataclass(frozen=True)
class Section:
    name: str
    label: str
    prompt_key: str
    context_cap: int = config.CONTEXT_CAP
    use_code_model: bool = False
    is_diagram: bool = False


PIPELINE: list[Section] = [
    Section("summary",  "📝 Summary",           "summary"),
    Section("logic",    "🔣 Symbolic logic",     "logic"),
    Section("cpp",      "💻 C++ examples",       "cpp",      use_code_model=True),
    Section("diagrams", "📊 Graphviz diagrams",  "diagrams", context_cap=config.DIAGRAM_CONTEXT_CAP, is_diagram=True),
    Section("extras",   "💡 Critical analysis",  "extras"),
]


# ══════════════════════════════════════════════════════════════════════════════
# LEASE CREDENTIALS — immutable fencing token cached at claim time
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LeaseContext:
    claim_key: str
    owner_id: str
    generation: int


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM POST-PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

def _handle_diagrams(
    raw: str,
    conn: sqlite3.Connection,
    paper_hash: str,
    lease: LeaseContext,
) -> bool:
    diagrams = parse_diagrams(raw)

    if not diagrams:
        store.write_diagrams_raw_fenced(
            conn, lease.claim_key, lease.owner_id, lease.generation,
            paper_hash, raw,
        )
        print(
            f"     ⚠️   No diagrams parsed from LLM output.\n"
            f"          Raw output saved (paper_hash={paper_hash}).\n"
            f"          Tip: re-run with --reprocess diagrams"
        )
        return False

    rows: list[tuple[str, str, Optional[str]]] = []
    for idx, (title, dot_src) in enumerate(diagrams, 1):
        dot_src = ensure_neon_black(dot_src)
        svg = render_dot(dot_src)
        status = "✓" if svg is not None else "✗ (dot saved, SVG render failed)"
        print(f"       {idx}. {title:<45} {status}")
        rows.append((title, dot_src, svg))

    store.replace_diagrams_fenced(
        conn, lease.claim_key, lease.owner_id, lease.generation,
        paper_hash, rows,
    )
    return len(diagrams) >= MIN_DIAGRAMS_FOR_COMPLETE


# ══════════════════════════════════════════════════════════════════════════════
# PAPER PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class PaperProcessor:
    def __init__(
        self,
        client: GenerationClient,
        shutdown: Event,
        db_path: Path,
        forced_model: Optional[str] = None,
        forced_code_model: Optional[str] = None,
        reprocess: Optional[str] = None,
        ocr_mode: str = "auto",
        ocr_min_chars: int = 100,
        ocr_dpi: int = 300,
        ocr_lang: str = "eng",
        ocr_max_pages: Optional[int] = config.DEFAULT_OCR_MAX_PAGES,
    ):
        self.client = client
        self.shutdown = shutdown
        self.db_path = db_path
        self.forced_model = forced_model
        self.forced_code_model = forced_code_model
        self.reprocess = reprocess
        self.ocr_mode = ocr_mode
        self.ocr_min_chars = ocr_min_chars
        self.ocr_dpi = ocr_dpi
        self.ocr_lang = ocr_lang
        self.ocr_max_pages = ocr_max_pages

    def process(self, pdf_path: Path) -> PaperStatus:
        conn = store.connect(self.db_path)
        try:
            return self._process_with_conn(pdf_path, conn)
        finally:
            conn.close()

    def _process_with_conn(self, pdf_path: Path, conn: sqlite3.Connection) -> PaperStatus:
        paper_hash = hash_file(pdf_path)

        record = store.load_paper(conn, paper_hash)
        if (
            record is not None
            and store.ALL_SECTIONS.issubset(set(record.sections_completed))
            and self.reprocess is None
        ):
            print(f"  ⏭   {pdf_path.name}  (all sections complete)")
            return PaperStatus.ALREADY_COMPLETE

        claim_key = f"sha256:{paper_hash}"
        claim = store.try_claim(conn, claim_key)
        if not claim.claimed:
            print(f"  🔒  {pdf_path.name}  (claimed by another worker — skipping)")
            return PaperStatus.CLAIMED_ELSEWHERE
        if claim.reclaimed:
            print(f"  ⚠️   {pdf_path.name}  (stale lock reclaimed)")

        lease = LeaseContext(claim_key, claim.owner_id, claim.generation)

        heartbeat = store.LeaseHeartbeat(
            self.db_path, claim_key, claim.owner_id, claim.generation,
        ).start()

        def lease_check() -> None:
            heartbeat.assert_healthy()
            if not store.claim_owned(conn, claim_key, claim.owner_id, claim.generation):
                raise LeaseLostError(f"processing lease lost for {claim_key}")

        try:
            return self._run_pipeline(
                pdf_path, paper_hash, record, conn, lease, lease_check
            )
        except ShutdownRequested:
            return PaperStatus.INTERRUPTED
        finally:
            try:
                heartbeat.stop()
            finally:
                store.release_claim(conn, claim_key, claim.owner_id, claim.generation)

    def _run_pipeline(
        self,
        pdf_path: Path,
        paper_hash: str,
        record: Optional[store.PaperRecord],
        conn: sqlite3.Connection,
        lease: LeaseContext,
        lease_check: Callable[[], None],
    ) -> PaperStatus:
        print(f"\n{'─'*64}")
        print(f"  📄  {pdf_path.name}")

        pages, ocr_stats = extract_pages(
            pdf_path,
            paper_hash=paper_hash,
            ocr_mode=self.ocr_mode,
            ocr_min_chars=self.ocr_min_chars,
            ocr_dpi=self.ocr_dpi,
            ocr_lang=self.ocr_lang,
            ocr_max_pages=self.ocr_max_pages,
            conn=conn,
            progress_check=lease_check,
        )
        lease_check()

        if self.reprocess:
            store.clear_section_fenced(
                conn, lease.claim_key, lease.owner_id, lease.generation,
                paper_hash, self.reprocess,
            )

        page_count = ocr_stats.total_pages
        model = config.select_model(page_count, self.forced_model)
        code_model = self.forced_code_model or model
        chunks = build_chunks(pages)
        print(
            f"     pages={page_count} physical/{len(pages)} with text  "
            f"chunks={len(chunks)}"
        )
        if ocr_stats.ocr_used or ocr_stats.cached_pages:
            print(f"     ocr={ocr_stats.summary()}")
        print(f"     model={model}  code_model={code_model}")

        completed: list[str] = list(record.sections_completed) if record else []
        if self.reprocess == "all":
            completed.clear()
        elif self.reprocess in completed:
            completed.remove(self.reprocess)

        # ── Evidence synthesis ───────────────────────────────────────────
        evidence: Optional[EvidenceBundle] = None
        if record and record.source_corpus and self.reprocess != "all":
            try:
                evidence = EvidenceBundle.from_json(
                    record.source_corpus, paper_hash=paper_hash, model=model
                )
                print(
                    f"     evidence=cache ({len(evidence.evidence)} records, "
                    f"{evidence.reduction_levels} levels)"
                )
            except EvidenceSynthesisError as exc:
                print(f"     evidence cache ignored: {exc}")

        def _evidence_generate(prompt: str) -> str:
            return self.client.generate(model, prompt, ctx_tokens=config.CHUNK_SUMMARY_CTX)

        if evidence is None:
            evidence = synthesize_evidence(
                chunks, _evidence_generate,
                paper_hash=paper_hash, model=model,
                physical_page_count=page_count,
                progress_check=lease_check,
            )
            print(
                f"     evidence=built ({len(evidence.evidence)} verified records, "
                f"{evidence.reduction_levels} levels)"
            )

        strategy = (
            f"hierarchical-evidence-v{config.EVIDENCE_SCHEMA_VERSION} "
            f"({len(chunks)} chunks, {evidence.reduction_levels} reduce levels, "
            f"{len(evidence.evidence)} records)"
        )
        print(f"     strategy={strategy}")

        # ── Fenced metadata + evidence write ─────────────────────────────
        store.upsert_meta_and_corpus_fenced(
            conn, lease.claim_key, lease.owner_id, lease.generation,
            paper_hash, pdf_path.name, str(pdf_path),
            page_count, strategy, model, code_model, evidence.to_json(),
        )

        # ── Section loop ─────────────────────────────────────────────────
        for section in PIPELINE:
            if self.shutdown.is_set():
                raise ShutdownRequested()
            lease_check()

            if self.reprocess not in (section.name, "all") and section.name in completed:
                continue

            print(f"     {section.label} …")
            active_model = code_model if section.use_code_model else model
            context = evidence.context_for(section.name, section.context_cap)

            prompt_text = DIAGRAM_PROMPT if section.is_diagram else PROMPTS[section.prompt_key]
            prompt = (
                f"{prompt_text}\n\n"
                "Use only the evidence bundle below for paper-specific factual claims. "
                "Cite evidence IDs and physical page numbers where useful.\n"
                f"<evidence_bundle>\n{context}\n</evidence_bundle>"
            )

            raw = self.client.generate(active_model, prompt)

            if section.is_diagram:
                diagram_ok = _handle_diagrams(raw, conn, paper_hash, lease)
                if diagram_ok:
                    if "diagrams" not in completed:
                        completed.append("diagrams")
                    print("         ✓")
                else:
                    print("         ✗ incomplete")
            else:
                formatted = f"# {section.label.split(maxsplit=1)[1]}\n\n{raw}\n"
                store.write_section_fenced(
                    conn, lease.claim_key, lease.owner_id, lease.generation,
                    paper_hash, section.name, formatted,
                )
                if section.name not in completed:
                    completed.append(section.name)
                print("         ✓")

        missing = store.ALL_SECTIONS - set(completed)
        if missing:
            print(f"     ⚠️   Partial — missing: {', '.join(sorted(missing))}")
            return PaperStatus.PARTIAL

        print(f"     ✅  Complete — paper_hash={paper_hash}")
        return PaperStatus.COMPLETE
