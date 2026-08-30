#!/usr/bin/env python3
"""
ocr_fallback.py — Local OCR fallback for scanned / image-only PDFs.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shared by paper_processor.py and vram_resident_processor.py.

Zero cloud dependency: OCR runs locally through Tesseract, driven by
PyMuPDF's built-in OCR text-page (so no extra Python packages beyond the
`fitz` we already require — only the system `tesseract-ocr` binary + a
language data pack such as `tesseract-ocr-eng`).

Behaviour
  • Pages are extracted normally first (fast, lossless for born-digital PDFs).
  • A page whose stripped text is below `min_chars` is treated as image-only
    and re-read via OCR — but ONLY when OCR is available and the mode allows it.
  • If Tesseract is missing, the module degrades gracefully: it logs one
    warning and returns whatever native text exists, exactly like the old
    behaviour. Nothing crashes.
  • OCR output is cached to disk (keyed by paper hash + page index), so a
    re-run or `--reprocess` never pays the OCR cost twice.

Modes
  "auto"   — OCR only the pages that look empty/low-text   (default)
  "always" — OCR every page (use for known-bad scans with garbage text layers)
  "never"  — disable OCR entirely (original behaviour)
"""

from __future__ import annotations

import functools
import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pymupdf as fitz  # `import fitz` alias is deprecated as of pymupdf 1.28


# ── Tunable defaults (overridable via CLI / kwargs) ─────────────────────────
DEFAULT_MIN_CHARS = 100   # a page with fewer stripped chars is "probably scanned"
DEFAULT_DPI       = 300   # rasterisation DPI handed to Tesseract
DEFAULT_LANG      = "eng" # Tesseract language pack(s), e.g. "eng" or "eng+deu"


def ocr_profile_key(*, dpi: int = DEFAULT_DPI, lang: str = DEFAULT_LANG) -> str:
    """Compute a cache-invalidation key covering the full OCR configuration.

    Captures DPI, language, Tesseract version, PyMuPDF version, a profile
    schema version (incremented when OCR flags, preprocessing, or
    normalization change), and fingerprints of the selected .traineddata
    files.  Any change to the pipeline — software upgrade, traineddata
    update, configuration change — produces a different profile hash.
    """
    tess_version = _tesseract_version() or "none"
    pymupdf_version = fitz.VersionBind
    traineddata_fp = _traineddata_fingerprint(lang)
    parts = (
        f"schema={_OCR_PROFILE_SCHEMA}|dpi={dpi}|lang={lang}"
        f"|tess={tess_version}|pymupdf={pymupdf_version}"
        f"|td={traineddata_fp}"
    )
    return hashlib.sha256(parts.encode()).hexdigest()[:12]


# Increment when OCR flags, full= mode, preprocessing, or normalization change.
_OCR_PROFILE_SCHEMA = 1


@functools.lru_cache(maxsize=1)
def _tesseract_version() -> Optional[str]:
    """Return Tesseract version string, or None if not installed.  Cached."""
    try:
        r = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        first_line = r.stdout.strip().split("\n")[0] if r.stdout else ""
        return first_line.split()[-1] if first_line else None
    except Exception:
        return None


@functools.lru_cache(maxsize=4)
def _traineddata_fingerprint(lang: str) -> str:
    """Hash the size+mtime of each .traineddata file for the requested langs.

    This detects traineddata updates without reading multi-MB files.
    Returns a stable string even when tessdata is unavailable.
    """
    tessdata = _find_tessdata()
    if not tessdata:
        return "no-tessdata"
    parts: list[str] = []
    for code in sorted(lang.split("+")):
        path = os.path.join(tessdata, f"{code}.traineddata")
        try:
            st = os.stat(path)
            parts.append(f"{code}:{st.st_size}:{int(st.st_mtime_ns)}")
        except OSError:
            parts.append(f"{code}:missing")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:10]


# Common Ubuntu / macOS tessdata locations, in priority order.
_TESSDATA_CANDIDATES = (
    "/usr/share/tesseract-ocr/5/tessdata",
    "/usr/share/tesseract-ocr/4.00/tessdata",
    "/usr/share/tesseract-ocr/tessdata",
    "/usr/share/tessdata",
    "/usr/local/share/tessdata",
    "/opt/homebrew/share/tessdata",
)


@dataclass
class OcrStats:
    """Per-document OCR accounting, surfaced to the caller for logging."""
    total_pages:   int  = 0
    ocr_pages:     int  = 0   # pages actually OCR'd this run
    cached_pages:  int  = 0   # pages served from the OCR cache
    native_pages:  int  = 0   # pages that had a usable native text layer
    skipped_blank: int  = 0   # pages still empty even after OCR (or OCR off)
    ocr_capped:    int  = 0   # pages that wanted OCR but hit the per-doc budget
    ocr_used:      bool = False
    notes:         List[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.ocr_used and self.ocr_pages == 0 and self.cached_pages == 0:
            return f"{self.native_pages}/{self.total_pages} pages had native text (no OCR needed)"
        bits = [f"{self.native_pages} native"]
        if self.ocr_pages:
            bits.append(f"{self.ocr_pages} OCR'd")
        if self.cached_pages:
            bits.append(f"{self.cached_pages} cached")
        if self.skipped_blank:
            bits.append(f"{self.skipped_blank} still-blank")
        if self.ocr_capped:
            bits.append(f"{self.ocr_capped} budget-capped")
        return f"{self.total_pages} pages → " + ", ".join(bits)


@dataclass(frozen=True)
class ExtractedPage:
    """Text retained from one physical PDF page.

    ``number`` is one-based and remains stable even when blank pages are
    omitted from downstream processing.  This lets generated evidence cite
    the source PDF rather than an index in a filtered list.
    """

    number: int
    text: str
    method: str  # "native", "ocr", or "ocr-cache"


# ── Availability detection ──────────────────────────────────────────────────
def _find_tessdata() -> Optional[str]:
    """Return a tessdata directory, honouring TESSDATA_PREFIX first."""
    env = os.environ.get("TESSDATA_PREFIX")
    if env and os.path.isdir(env):
        return env
    for cand in _TESSDATA_CANDIDATES:
        if os.path.isdir(cand):
            return cand
    return None


def ocr_available(lang: str = DEFAULT_LANG) -> Tuple[bool, str]:
    """
    Check whether local OCR can run. Returns (ok, detail).
    Side effect: when OK, exports TESSDATA_PREFIX so fitz/Tesseract find the
    language packs even if the user never set it.
    """
    if shutil.which("tesseract") is None:
        return False, "tesseract binary not found in PATH (apt install tesseract-ocr)"

    tessdata = _find_tessdata()
    if not tessdata:
        return False, "tessdata directory not found (set TESSDATA_PREFIX)"

    # Verify each requested language pack is present.
    missing = [
        code for code in lang.split("+")
        if not os.path.isfile(os.path.join(tessdata, f"{code}.traineddata"))
    ]
    if missing:
        pkgs = " ".join(f"tesseract-ocr-{c}" for c in missing)
        return False, f"missing language data {missing} (apt install {pkgs})"

    os.environ["TESSDATA_PREFIX"] = tessdata
    return True, tessdata


# ── Core OCR primitives ───────────────────────────────────────────────────--
def page_needs_ocr(text: str, min_chars: int = DEFAULT_MIN_CHARS) -> bool:
    """A page looks scanned/image-only if its stripped text is below threshold."""
    return len(text.strip()) < min_chars


def ocr_page(page: "fitz.Page", dpi: int = DEFAULT_DPI, lang: str = DEFAULT_LANG) -> str:
    """
    OCR a single PyMuPDF page via its built-in Tesseract text page.
    `full=True` forces OCR of the whole page (correct for image-only scans).
    Returns "" on any failure rather than raising, so one bad page can't abort
    a whole corpus run.
    """
    try:
        tp = page.get_textpage_ocr(flags=0, language=lang, dpi=dpi, full=True)
        return page.get_text("text", textpage=tp).strip()
    except Exception:
        return ""


# ── Main entry point ────────────────────────────────────────────────────────
def extract_pages_with_ocr(
    pdf_path: Path,
    *,
    mode: str = "auto",
    min_chars: int = DEFAULT_MIN_CHARS,
    dpi: int = DEFAULT_DPI,
    lang: str = DEFAULT_LANG,
    max_ocr_pages: Optional[int] = None,
    paper_hash: Optional[str] = None,
    cache_reader: Optional[Callable[[int], Optional[str]]] = None,
    cache_writer: Optional[Callable[[int, str], None]] = None,
    log=print,
) -> Tuple[List[ExtractedPage], OcrStats]:
    """
    Drop-in replacement for the old `extract_pages`, with OCR fallback.

    Returns (pages, stats) where ``pages`` contains non-empty physical pages
    with their original one-based page numbers. Blank pages are omitted, but
    their omission never renumbers the retained pages.

    Caching is enabled when BOTH `paper_hash` and `cache_reader`/`cache_writer`
    are supplied. Storage-agnostic by design: callers bind these to whatever
    backend they use (e.g. `functools.partial(paper_store.get_cached_ocr_page,
    conn, paper_hash)`) — this module has no knowledge of where the cache
    actually lives.
    """
    stats = OcrStats()
    if mode not in ("auto", "always", "never"):
        raise ValueError(f"invalid OCR mode {mode!r} (expected auto|always|never)")

    doc = fitz.open(str(pdf_path))
    stats.total_pages = doc.page_count

    # Resolve OCR availability once per document (only if the mode wants it).
    ocr_ok = False
    if mode != "never":
        ocr_ok, detail = ocr_available(lang)
        if not ocr_ok:
            stats.notes.append(f"OCR unavailable — {detail}")
            log(f"      ⚠️  OCR fallback disabled: {detail}")

    out: List[ExtractedPage] = []
    for idx, page in enumerate(doc):
        native = page.get_text("text").strip()
        want_ocr = ocr_ok and (mode == "always" or page_needs_ocr(native, min_chars))

        if not want_ocr:
            if native:
                stats.native_pages += 1
                out.append(ExtractedPage(idx + 1, native, "native"))
            else:
                stats.skipped_blank += 1
            continue

        # --- OCR path (with optional cache) ---
        text = ""
        cached = False
        if cache_reader is not None and paper_hash:
            hit = cache_reader(idx)
            if hit is not None:
                text = hit.strip()
                cached = True

        if not cached:
            # Per-document OCR budget: once we've freshly OCR'd `max_ocr_pages`,
            # stop rasterising further pages and fall back to native text. Caps
            # the blast radius on huge scanned books (cached pages stay free).
            if max_ocr_pages is not None and stats.ocr_pages >= max_ocr_pages:
                stats.ocr_capped += 1
                if native:
                    stats.native_pages += 1
                    out.append(ExtractedPage(idx + 1, native, "native"))
                else:
                    stats.skipped_blank += 1
                continue
            text = ocr_page(page, dpi=dpi, lang=lang)
            if cache_writer is not None and paper_hash:
                cache_writer(idx, text)

        stats.ocr_used = True
        if cached:
            stats.cached_pages += 1
        else:
            stats.ocr_pages += 1

        # OCR may still beat a genuinely-blank page; fall back to native if OCR empty.
        chosen = text or native
        if chosen:
            method = "ocr-cache" if cached else ("ocr" if text else "native")
            out.append(ExtractedPage(idx + 1, chosen, method))
        else:
            stats.skipped_blank += 1

    doc.close()
    return out, stats


# ── Manual smoke test:  python3 ocr_fallback.py some.pdf ────────────────────--
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        ok, detail = ocr_available()
        print(f"OCR available: {ok}  ({detail})")
        print("Usage: python3 ocr_fallback.py <file.pdf> [auto|always|never]")
        sys.exit(0)
    pages, st = extract_pages_with_ocr(
        Path(sys.argv[1]),
        mode=sys.argv[2] if len(sys.argv) > 2 else "auto",
    )
    print(st.summary())
    for p in pages:
        print(f"\n──── page {p.number} ({len(p.text)} chars, {p.method}) ────\n{p.text[:400]}")
