"""PDF extraction and hierarchical, source-grounded evidence synthesis.

The reader converts physical PDF pages into bounded chunks, extracts verified
evidence from every chunk, and reduces chunk dossiers hierarchically. Reduction
may summarize or select immutable evidence IDs; it cannot create new evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from . import config
from .ocr import ExtractedPage
from .prompts import EVIDENCE_MAP_PROMPT, EVIDENCE_REDUCE_PROMPT


Generate = Callable[[str], str]
EVIDENCE_KINDS = frozenset(
    {
        "motivation", "method", "algorithm", "equation", "result",
        "limitation", "comparison", "deployment", "definition",
    }
)
SECTION_KIND_PRIORITY: dict[str, tuple[str, ...]] = {
    "summary": ("motivation", "method", "result", "limitation", "comparison"),
    "logic": ("definition", "equation", "algorithm", "method", "limitation"),
    "cpp": ("algorithm", "method", "definition", "equation", "deployment"),
    "diagrams": ("method", "algorithm", "definition", "comparison", "result"),
    "extras": ("limitation", "result", "comparison", "deployment", "motivation"),
}


class EvidenceSynthesisError(RuntimeError):
    """Raised when evidence extraction or reduction cannot be validated."""


@dataclass(frozen=True)
class PageChunk:
    index: int
    pages: tuple[ExtractedPage, ...]

    @property
    def page_start(self) -> int:
        return min(page.number for page in self.pages)

    @property
    def page_end(self) -> int:
        return max(page.number for page in self.pages)

    @property
    def page_numbers(self) -> frozenset[int]:
        return frozenset(page.number for page in self.pages)

    def render(self) -> str:
        return "\n\n".join(f"[PAGE {page.number}]\n{page.text}" for page in self.pages)


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    chunk_index: int
    kind: str
    statement: str
    support: str
    page: int

    def as_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "chunk_index": self.chunk_index,
            "kind": self.kind,
            "statement": self.statement,
            "support": self.support,
            "page": self.page,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "EvidenceItem":
        return cls(
            evidence_id=str(value["evidence_id"]),
            chunk_index=int(value["chunk_index"]),
            kind=str(value["kind"]),
            statement=str(value["statement"]),
            support=str(value["support"]),
            page=int(value["page"]),
        )


@dataclass(frozen=True)
class ChunkEvidence:
    chunk_index: int
    page_start: int
    page_end: int
    summary: str
    evidence: tuple[EvidenceItem, ...]
    rejected_items: int = 0

    def as_dict(self) -> dict:
        return {
            "chunk_index": self.chunk_index,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "summary": self.summary,
            "evidence": [item.as_dict() for item in self.evidence],
            "rejected_items": self.rejected_items,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "ChunkEvidence":
        return cls(
            chunk_index=int(value["chunk_index"]),
            page_start=int(value["page_start"]),
            page_end=int(value["page_end"]),
            summary=str(value["summary"]),
            evidence=tuple(EvidenceItem.from_dict(item) for item in value["evidence"]),
            rejected_items=int(value.get("rejected_items", 0)),
        )


@dataclass(frozen=True)
class EvidenceRelationship:
    evidence_ids: tuple[str, ...]
    description: str

    def as_dict(self) -> dict:
        return {"evidence_ids": list(self.evidence_ids), "description": self.description}

    @classmethod
    def from_dict(cls, value: dict) -> "EvidenceRelationship":
        return cls(tuple(map(str, value["evidence_ids"])), str(value["description"]))


@dataclass(frozen=True)
class EvidenceDossier:
    page_start: int
    page_end: int
    summary: str
    selected_evidence_ids: tuple[str, ...]
    relationships: tuple[EvidenceRelationship, ...] = ()

    def as_dict(self) -> dict:
        return {
            "page_start": self.page_start,
            "page_end": self.page_end,
            "summary": self.summary,
            "selected_evidence_ids": list(self.selected_evidence_ids),
            "relationships": [rel.as_dict() for rel in self.relationships],
        }

    @classmethod
    def from_dict(cls, value: dict) -> "EvidenceDossier":
        return cls(
            page_start=int(value["page_start"]),
            page_end=int(value["page_end"]),
            summary=str(value["summary"]),
            selected_evidence_ids=tuple(map(str, value["selected_evidence_ids"])),
            relationships=tuple(
                EvidenceRelationship.from_dict(rel) for rel in value.get("relationships", [])
            ),
        )


@dataclass(frozen=True)
class EvidenceBundle:
    paper_hash: str
    model: str
    physical_page_count: int
    chunks: tuple[ChunkEvidence, ...]
    root: EvidenceDossier
    reduction_levels: int

    @property
    def evidence(self) -> tuple[EvidenceItem, ...]:
        return tuple(item for chunk in self.chunks for item in chunk.evidence)

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": config.EVIDENCE_SCHEMA_VERSION,
                "paper_hash": self.paper_hash,
                "model": self.model,
                "physical_page_count": self.physical_page_count,
                "chunks": [chunk.as_dict() for chunk in self.chunks],
                "root": self.root.as_dict(),
                "reduction_levels": self.reduction_levels,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, raw: str, *, paper_hash: str, model: str) -> "EvidenceBundle":
        try:
            value = json.loads(raw)
            if value["schema_version"] != config.EVIDENCE_SCHEMA_VERSION:
                raise ValueError("schema version mismatch")
            if value["paper_hash"] != paper_hash or value["model"] != model:
                raise ValueError("paper or model mismatch")
            bundle = cls(
                paper_hash=paper_hash,
                model=model,
                physical_page_count=int(value["physical_page_count"]),
                chunks=tuple(ChunkEvidence.from_dict(chunk) for chunk in value["chunks"]),
                root=EvidenceDossier.from_dict(value["root"]),
                reduction_levels=int(value["reduction_levels"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise EvidenceSynthesisError(f"invalid cached evidence bundle: {exc}") from exc

        ids = {item.evidence_id for item in bundle.evidence}
        if not ids or any(item.kind not in EVIDENCE_KINDS for item in bundle.evidence):
            raise EvidenceSynthesisError("cached evidence bundle has no valid evidence")
        if not set(bundle.root.selected_evidence_ids).issubset(ids):
            raise EvidenceSynthesisError("cached root refers to unknown evidence IDs")
        return bundle

    def context_for(self, section: str, cap: int) -> str:
        """Build a bounded, page-stratified context for one output section."""
        header = (
            "# Hierarchical evidence synthesis\n"
            f"Physical pages: {self.physical_page_count}\n"
            f"Evidence records: {len(self.evidence)}\n"
            f"Reduction levels: {self.reduction_levels}\n\n"
            "## Root synthesis\n"
            f"{self.root.summary}\n"
        )
        if len(header) >= cap:
            raise EvidenceSynthesisError("section context cap is smaller than root synthesis")

        by_id = {item.evidence_id: item for item in self.evidence}
        root_items = [by_id[eid] for eid in self.root.selected_evidence_ids if eid in by_id]
        priorities = SECTION_KIND_PRIORITY.get(section, ())
        rank = {kind: index for index, kind in enumerate(priorities)}
        remainder = [
            item for item in self.evidence if item.evidence_id not in self.root.selected_evidence_ids
        ]
        remainder.sort(key=lambda item: (rank.get(item.kind, len(rank)), item.page, item.evidence_id))
        coverage = _coverage_order(root_items)
        candidates = coverage + [item for item in remainder if item not in coverage]

        chunks_section = "\n## Chunk coverage\n"
        for chunk in _coverage_order(list(self.chunks)):
            entry = f"- Pages {chunk.page_start}–{chunk.page_end}: {chunk.summary}\n"
            if len(header) + len(chunks_section) + len(entry) > int(cap * 0.42):
                continue
            chunks_section += entry

        evidence_section = "\n## Verified evidence\n"
        used: set[str] = set()
        for item in candidates:
            if item.evidence_id in used:
                continue
            entry = (
                f"[{item.evidence_id}] {item.kind}; page {item.page}\n"
                f"Claim: {item.statement}\n"
                f"Exact support: \"{item.support}\"\n\n"
            )
            if len(header) + len(chunks_section) + len(evidence_section) + len(entry) > cap:
                continue
            evidence_section += entry
            used.add(item.evidence_id)

        relationships = "\n## Supported relationships\n"
        for relation in self.root.relationships:
            entry = f"- {', '.join(relation.evidence_ids)}: {relation.description}\n"
            if (
                len(header) + len(chunks_section) + len(evidence_section)
                + len(relationships) + len(entry) > cap
            ):
                break
            relationships += entry
        return header + chunks_section + evidence_section + relationships


def hash_file(path: Path) -> str:
    """Stream a file through SHA-256 without loading it entirely into RAM.

    Returns the full 64-character hex digest — no truncation.  This value
    determines database identity and lease ownership.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_pages(
    pdf_path: Path,
    *,
    paper_hash: str,
    ocr_mode: str = "auto",
    ocr_min_chars: int = 100,
    ocr_dpi: int = 300,
    ocr_lang: str = "eng",
    ocr_max_pages: Optional[int] = config.DEFAULT_OCR_MAX_PAGES,
    conn=None,
    progress_check: Callable[[], None] = lambda: None,
):
    """Extract physical pages with optional database-backed OCR caching.

    The OCR cache is keyed by (paper_hash, page_idx, ocr_profile) where
    ocr_profile captures DPI, language, and backend versions so that
    configuration changes invalidate stale entries.
    """
    import functools
    from .ocr import extract_pages_with_ocr, ocr_profile_key

    profile = ocr_profile_key(dpi=ocr_dpi, lang=ocr_lang)

    cache_reader = None
    cache_writer = None
    if conn is not None:
        from . import store

        cache_reader = functools.partial(
            store.get_cached_ocr_page, conn, paper_hash, ocr_profile=profile
        )

        def cache_writer(page_index: int, text: str) -> None:
            progress_check()
            store.put_cached_ocr_page(conn, paper_hash, page_index, profile, text)

    progress_check()
    return extract_pages_with_ocr(
        pdf_path,
        mode=ocr_mode,
        min_chars=ocr_min_chars,
        dpi=ocr_dpi,
        lang=ocr_lang,
        max_ocr_pages=ocr_max_pages or None,
        paper_hash=paper_hash,
        cache_reader=cache_reader,
        cache_writer=cache_writer,
    )


def build_chunks(
    pages: list[ExtractedPage],
    window: int = config.CHUNK_WINDOW,
    overlap: int = config.CHUNK_OVERLAP,
    char_cap: int = config.CHUNK_INPUT_CHAR_CAP,
) -> list[PageChunk]:
    """Create page-labelled chunks bounded by both page and character counts."""
    if window < 1 or overlap < 0 or overlap >= window or char_cap < 1_000:
        raise ValueError("invalid chunking configuration")
    if not pages:
        raise EvidenceSynthesisError("PDF contains no extractable text")

    fragments: list[ExtractedPage] = []
    fragment_cap = max(1_000, char_cap - 64)
    for page in pages:
        if len(page.text) <= fragment_cap:
            fragments.append(page)
            continue
        for start in range(0, len(page.text), fragment_cap):
            fragments.append(
                ExtractedPage(page.number, page.text[start : start + fragment_cap], page.method)
            )

    chunks: list[PageChunk] = []
    start = 0
    while start < len(fragments):
        selected: list[ExtractedPage] = []
        chars = 0
        end = start
        while end < len(fragments) and len(selected) < window:
            candidate = fragments[end]
            addition = len(candidate.text) + 32
            if selected and chars + addition > char_cap:
                break
            selected.append(candidate)
            chars += addition
            end += 1
        chunks.append(PageChunk(len(chunks) + 1, tuple(selected)))
        if end == len(fragments):
            break
        start = max(start + 1, end - min(overlap, len(selected) - 1))
    return chunks


def synthesize_evidence(
    chunks: list[PageChunk],
    generate: Generate,
    *,
    paper_hash: str,
    model: str,
    physical_page_count: int,
    progress_check: Callable[[], None] = lambda: None,
) -> EvidenceBundle:
    """Map chunks to verified evidence, then reduce dossiers hierarchically."""
    print(f"      ↳ Evidence map: {len(chunks)} chunk(s) …")
    mapped: list[ChunkEvidence] = []
    seen_support: set[tuple[int, str]] = set()
    for chunk in chunks:
        progress_check()
        result = _map_chunk(chunk, generate)
        progress_check()
        unique: list[EvidenceItem] = []
        for item in result.evidence:
            key = (item.page, _normalise(item.support))
            if key not in seen_support:
                seen_support.add(key)
                unique.append(item)
        result = ChunkEvidence(
            result.chunk_index, result.page_start, result.page_end,
            result.summary, tuple(unique), result.rejected_items,
        )
        mapped.append(result)
        print(
            f"        chunk {chunk.index}/{len(chunks)} pages "
            f"{chunk.page_start}–{chunk.page_end}: {len(unique)} verified"
            + (f", {result.rejected_items} rejected" if result.rejected_items else "")
        )

    all_evidence = tuple(item for result in mapped for item in result.evidence)
    if not all_evidence:
        raise EvidenceSynthesisError("no source-verifiable evidence was extracted")
    registry = {item.evidence_id: item for item in all_evidence}
    dossiers = [
        EvidenceDossier(
            chunk.page_start, chunk.page_end, chunk.summary,
            tuple(item.evidence_id for item in chunk.evidence),
        )
        for chunk in mapped
    ]
    levels = 0
    while len(dossiers) > 1:
        levels += 1
        print(f"      ↳ Evidence reduce level {levels}: {len(dossiers)} dossier(s) …")
        reduced: list[EvidenceDossier] = []
        for start in range(0, len(dossiers), config.EVIDENCE_REDUCE_GROUP):
            progress_check()
            group = dossiers[start : start + config.EVIDENCE_REDUCE_GROUP]
            reduced.append(_reduce_group(group, registry, generate))
            progress_check()
        dossiers = reduced

    return EvidenceBundle(
        paper_hash, model, physical_page_count, tuple(mapped), dossiers[0], levels
    )


def _map_chunk(chunk: PageChunk, generate: Generate) -> ChunkEvidence:
    prompt = (
        EVIDENCE_MAP_PROMPT
        + f"\nChunk: {chunk.index}; allowed pages: {sorted(chunk.page_numbers)}\n\n"
        + chunk.render()
    )
    last_error = ""
    raw = ""
    for attempt in range(config.EVIDENCE_MAP_RETRIES + 1):
        raw = generate(prompt if attempt == 0 else _repair_prompt(prompt, raw, last_error))
        try:
            value = _parse_json_object(raw)
            summary = _bounded_text(value["chunk_summary"], "chunk_summary", 4_000)
            raw_items = value.get("evidence", [])
            if not isinstance(raw_items, list):
                raise ValueError("evidence must be a list")
            items, rejected = _validate_evidence_items(raw_items, chunk)
            if raw_items and not items:
                raise ValueError("all proposed evidence failed source verification")
            return ChunkEvidence(
                chunk.index, chunk.page_start, chunk.page_end,
                summary, tuple(items), rejected,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
    raise EvidenceSynthesisError(f"chunk {chunk.index} evidence extraction failed: {last_error}")


def _validate_evidence_items(
    values: list,
    chunk: PageChunk,
) -> tuple[list[EvidenceItem], int]:
    page_text: dict[int, str] = {}
    for page in chunk.pages:
        page_text[page.number] = page_text.get(page.number, "") + " " + page.text
    accepted: list[EvidenceItem] = []
    rejected = 0
    for value in values:
        try:
            if not isinstance(value, dict):
                raise ValueError
            kind = str(value["kind"]).strip().lower()
            statement = _bounded_text(value["statement"], "statement", 1_000)
            support = _bounded_text(value["support"], "support", 600)
            page = int(value["page"])
            if kind not in EVIDENCE_KINDS or page not in chunk.page_numbers:
                raise ValueError
            if _normalise(support) not in _normalise(page_text[page]):
                raise ValueError
            evidence_id = f"C{chunk.index:03d}-E{len(accepted) + 1:03d}"
            accepted.append(
                EvidenceItem(evidence_id, chunk.index, kind, statement, support, page)
            )
        except (KeyError, TypeError, ValueError):
            rejected += 1
    return accepted, rejected


def _reduce_group(
    group: list[EvidenceDossier],
    registry: dict[str, EvidenceItem],
    generate: Generate,
) -> EvidenceDossier:
    available_ids = tuple(
        dict.fromkeys(eid for dossier in group for eid in dossier.selected_evidence_ids)
    )
    payload = {
        "child_dossiers": [dossier.as_dict() for dossier in group],
        "evidence_records": [
            {
                **registry[eid].as_dict(),
                "statement": registry[eid].statement[:700],
                "support": registry[eid].support[:400],
            }
            for eid in available_ids if eid in registry
        ],
    }
    base_prompt = EVIDENCE_REDUCE_PROMPT.replace(
        "{max_selected}", str(config.EVIDENCE_MAX_SELECTED)
    )
    prompt = base_prompt + "\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False)
    last_error = ""
    raw = ""
    for attempt in range(config.EVIDENCE_REDUCE_RETRIES + 1):
        raw = generate(prompt if attempt == 0 else _repair_prompt(prompt, raw, last_error))
        try:
            value = _parse_json_object(raw)
            summary = _bounded_text(
                value["summary"], "summary", config.EVIDENCE_SUMMARY_MAX_CHARS
            )
            selected = value.get("selected_evidence_ids", [])
            if not isinstance(selected, list):
                raise ValueError("selected_evidence_ids must be a list")
            selected = list(dict.fromkeys(map(str, selected)))
            unknown = set(selected) - set(available_ids)
            if unknown:
                raise ValueError(f"unknown evidence IDs: {sorted(unknown)}")

            representatives = [
                evidence_id
                for dossier in group
                for evidence_id in _coverage_evidence_ids(
                    list(dossier.selected_evidence_ids), registry
                )[:2]
            ]
            selected = list(dict.fromkeys(representatives + selected))
            selected = _coverage_evidence_ids(selected, registry)[
                : config.EVIDENCE_MAX_SELECTED
            ]
            relationships = _validate_relationships(
                value.get("relationships", []), frozenset(selected)
            )
            return EvidenceDossier(
                min(dossier.page_start for dossier in group),
                max(dossier.page_end for dossier in group),
                summary, tuple(selected), tuple(relationships),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
    raise EvidenceSynthesisError(
        f"evidence reduction for pages {group[0].page_start}–{group[-1].page_end} "
        f"failed: {last_error}"
    )


def _validate_relationships(
    values: object,
    selected: frozenset[str],
) -> list[EvidenceRelationship]:
    if not isinstance(values, list):
        raise ValueError("relationships must be a list")
    relationships: list[EvidenceRelationship] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        ids = tuple(dict.fromkeys(map(str, value.get("evidence_ids", []))))
        description = str(value.get("description", "")).strip()
        if len(ids) >= 2 and set(ids).issubset(selected) and description:
            relationships.append(EvidenceRelationship(ids, description[:1_500]))
    return relationships


_JSON_VALID_ESCAPE = re.compile(r'\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})')


def _repair_invalid_backslashes(text: str) -> str:
    """Double any backslash that isn't already part of a valid JSON escape.

    Math-heavy `support` excerpts routinely contain LaTeX (`\\dot{u}`,
    `\\alpha`). The model is instructed to quote verbatim, so it reproduces
    the single backslash as-is instead of doubling it for JSON — `\\d` isn't
    a recognised escape, so the decoder rejects the whole object. This
    repairs the semantically-intended literal backslash into valid JSON
    without touching escapes that are already correct.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "\\":
            m = _JSON_VALID_ESCAPE.match(text, i)
            if m:
                out.append(m.group(0))
                i = m.end()
                continue
            out.append("\\\\")
            i += 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _parse_json_object(raw: str) -> dict:
    """Extract the first top-level JSON object from a raw model response.

    Two model-output quirks are repaired before decoding, both stemming
    from the map/reduce prompts instructing verbatim quotes:
      - strict=False: PDF text extraction from math/symbol fonts occasionally
        yields raw control characters (observed: U+0001-U+0003, U+0012-U+0013)
        in place of garbled glyphs, which land unescaped inside a string.
      - _repair_invalid_backslashes: LaTeX notation in `support` excerpts
        (e.g. `\\dot{u}`) arrives with a single, un-doubled backslash.
    Left unrepaired, either one fails the top-level decode, and this
    function's per-`{` fallback then silently returns an unrelated nested
    fragment (e.g. one evidence item, missing `chunk_summary`) instead of
    raising a clear error.
    """
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    text = _repair_invalid_backslashes(text)
    decoder = json.JSONDecoder(strict=False)
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise json.JSONDecodeError("no JSON object found", text, 0)


def _repair_prompt(original_prompt: str, invalid_output: str, error: str) -> str:
    return (
        original_prompt
        + "\n\nYour previous response failed validation. Return a corrected JSON object only."
        + f"\nValidation error: {error}\nPrevious response:\n{invalid_output[:8_000]}"
    )


def _bounded_text(value: object, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()[:maximum]


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split())


def _coverage_order(values: list) -> list:
    """Order a page-sorted sequence to expose beginning, end, then interiors."""
    if len(values) < 3:
        return list(values)
    ordered = sorted(
        values,
        key=lambda value: (
            (value.page_start, value.page_end)
            if hasattr(value, "page_start") else (value.page, value.page)
        ),
    )
    indices: list[int] = []
    queue: list[tuple[int, int]] = [(0, len(ordered) - 1)]
    while queue:
        low, high = queue.pop(0)
        if low > high:
            continue
        for index in (low, high, (low + high) // 2):
            if index not in indices:
                indices.append(index)
        if low + 1 <= (low + high) // 2 - 1:
            queue.append((low + 1, (low + high) // 2 - 1))
        if (low + high) // 2 + 1 <= high - 1:
            queue.append(((low + high) // 2 + 1, high - 1))
    return [ordered[index] for index in indices]


def _coverage_evidence_ids(
    evidence_ids: list[str],
    registry: dict[str, EvidenceItem],
) -> list[str]:
    items = [registry[evidence_id] for evidence_id in evidence_ids if evidence_id in registry]
    return [item.evidence_id for item in _coverage_order(items)]


def quick_page_count(pdf_path: Path) -> int:
    """Cheap page count via PyMuPDF metadata—no extraction or OCR."""
    try:
        import pymupdf as fitz

        with fitz.open(pdf_path) as doc:
            return doc.page_count
    except Exception:
        return 0
