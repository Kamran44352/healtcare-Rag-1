from __future__ import annotations
import re
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import tiktoken

PARENT_MAX_TOKENS = 900
PARENT_OVERLAP_TOKENS = 120
CHILD_MAX_TOKENS = 220
CHILD_OVERLAP_TOKENS = 40

_ENCODER = tiktoken.get_encoding("cl100k_base")
_SECTION_RE = re.compile(r"^(#{1,2})\s+(.+)$", re.MULTILINE)


@dataclass
class ChildChunk:
    chunk_id: UUID
    parent_chunk_id: UUID
    document_id: UUID
    chunk_index: int
    text: str
    section_path: str
    token_count: int
    contextualized_text: str = ""
    section_type: str | None = None
    recommendation_strength: str | None = None
    entities: dict = field(default_factory=dict)
    qdrant_point_id: UUID | None = None


@dataclass
class ParentChunk:
    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    text: str
    section_path: str
    token_count: int
    children: list[ChildChunk] = field(default_factory=list)


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _token_split(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split text into chunks by token count with overlap."""
    tokens = _ENCODER.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(_ENCODER.decode(chunk_tokens))
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def _split_into_sections(markdown: str) -> list[tuple[str, str]]:
    """
    Return list of (section_path, section_text) pairs.
    Never crosses H1/H2 boundaries.
    """
    positions = [(m.start(), m.group(1), m.group(2)) for m in _SECTION_RE.finditer(markdown)]
    if not positions:
        return [("Document", markdown)]

    sections: list[tuple[str, str]] = []
    h1_title = ""
    h2_title = ""

    for i, (pos, hashes, title) in enumerate(positions):
        level = len(hashes)
        if level == 1:
            h1_title = title
            h2_title = ""
        else:
            h2_title = title

        section_path = h1_title if not h2_title else f"{h1_title} > {h2_title}"

        # Text is from this heading to the next
        end_pos = positions[i + 1][0] if i + 1 < len(positions) else len(markdown)
        section_text = markdown[pos:end_pos].strip()
        if section_text:
            sections.append((section_path, section_text))

    # Text before the first heading
    preamble = markdown[: positions[0][0]].strip()
    if preamble:
        sections.insert(0, ("Preamble", preamble))

    return sections


def chunk_document(markdown: str, document_id: UUID) -> list[ParentChunk]:
    """
    Stage E: Section-aware chunking.
    Returns ParentChunk objects each containing ChildChunk children.
    """
    sections = _split_into_sections(markdown)
    parents: list[ParentChunk] = []
    parent_idx = 0
    child_idx = 0

    for section_path, section_text in sections:
        parent_texts = _token_split(section_text, PARENT_MAX_TOKENS, PARENT_OVERLAP_TOKENS)

        for p_text in parent_texts:
            parent_id = uuid4()
            p_tokens = _count_tokens(p_text)
            parent = ParentChunk(
                chunk_id=parent_id,
                document_id=document_id,
                chunk_index=parent_idx,
                text=p_text,
                section_path=section_path,
                token_count=p_tokens,
            )

            child_texts = _token_split(p_text, CHILD_MAX_TOKENS, CHILD_OVERLAP_TOKENS)
            for c_text in child_texts:
                child_id = uuid4()
                parent.children.append(
                    ChildChunk(
                        chunk_id=child_id,
                        parent_chunk_id=parent_id,
                        document_id=document_id,
                        chunk_index=child_idx,
                        text=c_text,
                        section_path=section_path,
                        token_count=_count_tokens(c_text),
                        qdrant_point_id=child_id,
                    )
                )
                child_idx += 1

            parents.append(parent)
            parent_idx += 1

    return parents
