"""
Attorney.AI — Legal-Aware Document Chunker

Splits legal documents respecting section headings, paragraph breaks,
and clause boundaries. Much better than naive fixed-token chunking for
case opinions, statutes, regulations, and contracts.
"""
import re
import uuid
from typing import List, Tuple

import tiktoken
from loguru import logger

from ingestion.metadata_schema import LegalChunkMetadata, SourceType


# ── Tokenizer ─────────────────────────────────────────────────────────────────
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


# ── Section-header patterns for common legal document types ──────────────────
_SECTION_PATTERNS = [
    # CFR / U.S. Code section markers: § 123.45, Sec. 1, Section 2
    re.compile(r"^(?:§+\s*[\d.]+|Sec(?:tion)?\.?\s+[\d.]+)", re.MULTILINE | re.IGNORECASE),
    # Case opinion structural headers
    re.compile(
        r"^(?:I{1,3}V?|V?I{0,3})\.\s+[A-Z][A-Z\s]+$",
        re.MULTILINE
    ),
    # Numbered contract article / clause headers: ARTICLE I, CLAUSE 3
    re.compile(r"^(?:ARTICLE|CLAUSE|SECTION|PARAGRAPH)\s+[\dIVXivx]+", re.MULTILINE),
    # Lettered sub-sections: (a), (b), (i), (ii)
    re.compile(r"^\s*\([a-z]{1,3}\)\s+", re.MULTILINE),
    # Double-newline paragraph breaks (most reliable)
    re.compile(r"\n{2,}"),
]


def _split_into_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split document text into (heading, body) tuples by structural markers.
    Returns a flat list of section texts with their detected headings.
    """
    # First try to split on strong section headers
    strong_re = re.compile(
        r"(^(?:§+\s*[\d.]+|Sec(?:tion)?\.?\s+[\d.]+|ARTICLE\s+[\dIVX]+|CLAUSE\s+[\d]+)[^\n]*\n)",
        re.MULTILINE | re.IGNORECASE,
    )
    parts = strong_re.split(text)

    if len(parts) <= 1:
        # Fallback: split on double newlines (paragraphs)
        raw_sections = re.split(r"\n{2,}", text)
        return [("", s.strip()) for s in raw_sections if s.strip()]

    # Re-combine: each even index is a header, odd index is body
    sections: List[Tuple[str, str]] = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts):
            header = parts[i].strip()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if body:
                sections.append((header, body))
            i += 2
        else:
            if parts[i].strip():
                sections.append(("", parts[i].strip()))
            i += 1
    return sections


def chunk_legal_document(
    text: str,
    metadata_base: dict,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[LegalChunkMetadata]:
    """
    Main chunking function. Takes raw document text and base metadata,
    returns a list of LegalChunkMetadata objects ready for indexing.

    Strategy:
    1. Split by legal section/clause boundaries
    2. If a section is too long, slide a token window with overlap
    3. If a section is too short, merge with the next section
    4. Tag each chunk with its parent section heading
    """
    sections = _split_into_sections(text)
    chunks: List[LegalChunkMetadata] = []
    char_cursor = 0

    # Merge buffer for sections that are too small
    merge_buffer_heading = ""
    merge_buffer_text = ""
    merge_buffer_start = 0

    for heading, body in sections:
        combined = (merge_buffer_text + "\n\n" + body).strip() if merge_buffer_text else body
        combined_heading = merge_buffer_heading or heading

        tok_count = count_tokens(combined)

        if tok_count < chunk_size // 4 and sections.index((heading, body)) < len(sections) - 1:
            # Section is too small — buffer it and merge with next
            merge_buffer_heading = combined_heading
            merge_buffer_text = combined
            if not merge_buffer_text:
                merge_buffer_start = char_cursor
            char_cursor += len(heading) + len(body) + 2
            continue

        # Flush the merge buffer into this section
        final_heading = combined_heading
        final_text = combined
        section_start_char = merge_buffer_start if merge_buffer_text else char_cursor
        merge_buffer_heading = ""
        merge_buffer_text = ""
        merge_buffer_start = 0

        # Slide a window if section exceeds chunk_size
        section_chunks = _sliding_window(
            text=final_text,
            heading=final_heading,
            start_char=section_start_char,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )

        for i, (chunk_text, s_char, e_char) in enumerate(section_chunks):
            chunk_id = f"{metadata_base['doc_id']}:{len(chunks):04d}"
            meta = LegalChunkMetadata(
                doc_id=metadata_base["doc_id"],
                chunk_id=chunk_id,
                title=metadata_base["title"],
                citation=metadata_base.get("citation", ""),
                source_url=metadata_base.get("source_url", ""),
                jurisdiction=metadata_base.get("jurisdiction", "US-Federal"),
                court_or_agency=metadata_base.get("court_or_agency"),
                decision_date=metadata_base.get("decision_date"),
                date_str=metadata_base.get("date_str"),
                source_type=metadata_base.get("source_type", SourceType.CASE),
                parent_section=final_heading or None,
                start_char=s_char,
                end_char=e_char,
                text=chunk_text,
                court_level=metadata_base.get("court_level"),
                docket_number=metadata_base.get("docket_number"),
                author_judge=metadata_base.get("author_judge"),
                practice_area=metadata_base.get("practice_area"),
            )
            chunks.append(meta)

        char_cursor += len(heading) + len(body) + 2

    # Flush any remaining buffer
    if merge_buffer_text:
        chunk_id = f"{metadata_base['doc_id']}:{len(chunks):04d}"
        meta = LegalChunkMetadata(
            doc_id=metadata_base["doc_id"],
            chunk_id=chunk_id,
            title=metadata_base["title"],
            citation=metadata_base.get("citation", ""),
            source_url=metadata_base.get("source_url", ""),
            jurisdiction=metadata_base.get("jurisdiction", "US-Federal"),
            court_or_agency=metadata_base.get("court_or_agency"),
            decision_date=metadata_base.get("decision_date"),
            date_str=metadata_base.get("date_str"),
            source_type=metadata_base.get("source_type", SourceType.CASE),
            parent_section=merge_buffer_heading or None,
            start_char=merge_buffer_start,
            end_char=merge_buffer_start + len(merge_buffer_text),
            text=merge_buffer_text,
            court_level=metadata_base.get("court_level"),
            docket_number=metadata_base.get("docket_number"),
        )
        chunks.append(meta)

    logger.debug(f"Chunked '{metadata_base.get('title', '?')}' → {len(chunks)} chunks")
    return chunks


def _sliding_window(
    text: str,
    heading: str,
    start_char: int,
    chunk_size: int,
    overlap: int,
) -> List[Tuple[str, int, int]]:
    """
    Slide a token window over text that's too long for a single chunk.
    Returns list of (chunk_text, start_char, end_char).
    """
    words = text.split()
    if not words:
        return []

    results: List[Tuple[str, int, int]] = []
    window: List[str] = []
    window_tokens = 0
    char_offset = start_char

    for word in words:
        word_tokens = count_tokens(word)
        if window_tokens + word_tokens > chunk_size and window:
            chunk_text = " ".join(window)
            chunk_len = len(chunk_text)
            results.append((chunk_text, char_offset, char_offset + chunk_len))
            # Slide: keep last `overlap` tokens
            overlap_words = []
            overlap_tok = 0
            for w in reversed(window):
                wt = count_tokens(w)
                if overlap_tok + wt > overlap:
                    break
                overlap_words.insert(0, w)
                overlap_tok += wt
            window = overlap_words
            window_tokens = overlap_tok
            char_offset += chunk_len - len(" ".join(overlap_words))

        window.append(word)
        window_tokens += word_tokens

    if window:
        chunk_text = " ".join(window)
        results.append((chunk_text, char_offset, char_offset + len(chunk_text)))

    return results if results else [(text, start_char, start_char + len(text))]
