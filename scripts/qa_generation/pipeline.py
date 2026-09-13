"""Read-only corpus export and resumable two-stage Q&A generation helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field


def load_chunks(sqlite_path: str | Path) -> list[dict[str, Any]]:
    """Load active, version-matched chunks using SQLite's read-only URI mode."""
    path = Path(sqlite_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    uri = f"file:{path.as_posix()}?mode=ro"
    query = """
        SELECT d.doc_id, d.document_version, d.category,
               d.source_last_updated, d.source_content_hash,
               c.chunk_id, c.position, c.chunk_text, c.heading_path,
               c.source_title, c.source_url
        FROM rag_documents AS d
        JOIN rag_chunks AS c
          ON c.doc_id = d.doc_id
         AND c.document_version = d.document_version
        WHERE d.status = 'ok'
        ORDER BY d.doc_id, d.document_version, c.position
    """
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(query)]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def eligible_chunks(rows: list[dict[str, Any]], min_chars: int = 240) -> list[dict[str, Any]]:
    """Remove empty, tiny, and normalized-duplicate chunks without changing source DB."""
    seen: set[str] = set()
    result = []
    for row in rows:
        text = (row.get("chunk_text") or "").strip()
        normalized = normalize_text(text)
        if len(text) < min_chars or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        row = dict(row)
        row["chunk_text"] = text
        result.append(row)
    return result


def build_contexts(rows: list[dict[str, Any]], max_chars: int = 9000) -> list[dict[str, Any]]:
    """Create one anchor context, adding adjacent chunks only within one document version."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["doc_id"], row["document_version"]), []).append(row)
    contexts = []
    for group_rows in groups.values():
        group_rows.sort(key=lambda r: r["position"])
        for index, anchor in enumerate(group_rows):
            selected = [anchor]
            text_len = len(anchor["chunk_text"])
            for neighbor_index in (index - 1, index + 1):
                if 0 <= neighbor_index < len(group_rows):
                    neighbor = group_rows[neighbor_index]
                    if text_len + len(neighbor["chunk_text"]) + 2 <= max_chars:
                        selected.append(neighbor)
                        text_len += len(neighbor["chunk_text"]) + 2
            selected.sort(key=lambda r: r["position"])
            contexts.append({
                "context_id": f"{anchor['doc_id']}:{anchor['document_version']}:{anchor['chunk_id']}",
                "doc_id": anchor["doc_id"],
                "document_version": anchor["document_version"],
                "category": anchor["category"],
                "chunk_ids": [r["chunk_id"] for r in selected],
                "source_url": anchor["source_url"],
                "source_title": anchor["source_title"],
                "heading_paths": [r["heading_path"] for r in selected],
                "context": "\n\n".join(r["chunk_text"] for r in selected),
            })
    return contexts


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Return a conservative, tokenizer-independent token estimate.

    The generation server ultimately enforces its real context limit. This
    estimate is intentionally visible in the generated provenance so it can be
    replaced with the model tokenizer later without changing the context plan.
    """
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    return math.ceil(len(text) / chars_per_token)


def _heading_label(row: dict[str, Any]) -> str:
    """Turn the stored heading path into a readable label."""
    raw = (row.get("heading_path") or "").strip()
    if not raw:
        return "Page content"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(value, list):
        return " > ".join(str(part).strip() for part in value if str(part).strip()) or "Page content"
    return str(value).strip() or "Page content"


def _render_page_chunk(row: dict[str, Any]) -> str:
    return f"SECTION: {_heading_label(row)}\n{row['chunk_text']}"


def _page_context_payload(
    *,
    page_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    context_id: str,
    context_kind: str,
    window_index: int,
    chars_per_token: float,
) -> dict[str, Any]:
    anchor = selected_rows[0]
    body = "\n\n".join(_render_page_chunk(row) for row in selected_rows)
    context = f"PAGE TITLE: {anchor['source_title']}\n\n{body}"
    return {
        "context_id": context_id,
        "generation_strategy": "page-aware-v2",
        "context_kind": context_kind,
        "window_index": window_index,
        "doc_id": anchor["doc_id"],
        "document_version": anchor["document_version"],
        "category": anchor["category"],
        "source_url": anchor["source_url"],
        "source_title": anchor["source_title"],
        "page_chunk_count": len(page_rows),
        "chunk_ids": [row["chunk_id"] for row in selected_rows],
        "heading_paths": [_heading_label(row) for row in selected_rows],
        "estimated_context_tokens": estimate_tokens(context, chars_per_token),
        "context": context,
    }


def build_page_contexts(
    rows: list[dict[str, Any]],
    max_input_tokens: int = 24000,
    chars_per_token: float = 4.0,
) -> list[dict[str, Any]]:
    """Build full-page contexts, splitting only pages above the input budget.

    Oversized pages are packed as consecutive heading-aware windows. A heading
    is never split unless that heading itself exceeds the budget; in that case,
    its chunks are packed consecutively. This keeps page structure intact while
    avoiding arbitrary character slicing.
    """
    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["doc_id"], row["document_version"]), []).append(row)

    contexts: list[dict[str, Any]] = []
    for page_rows in groups.values():
        page_rows.sort(key=lambda row: row["position"])
        page_id = f"{page_rows[0]['doc_id']}:{page_rows[0]['document_version']}:page-aware-v2"
        whole_page = _page_context_payload(
            page_rows=page_rows,
            selected_rows=page_rows,
            context_id=page_id,
            context_kind="full_page",
            window_index=1,
            chars_per_token=chars_per_token,
        )
        if whole_page["estimated_context_tokens"] <= max_input_tokens:
            contexts.append(whole_page)
            continue

        # Create consecutive sections. Repeated heading paths elsewhere on a
        # page remain separate, preserving the original reading order.
        sections: list[list[dict[str, Any]]] = []
        for row in page_rows:
            if sections and _heading_label(sections[-1][-1]) == _heading_label(row):
                sections[-1].append(row)
            else:
                sections.append([row])

        windows: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for section in sections:
            candidates = [section] if len(section) == 1 else [section]
            for candidate in candidates:
                candidate_payload = _page_context_payload(
                    page_rows=page_rows,
                    selected_rows=candidate,
                    context_id="temporary",
                    context_kind="page_window",
                    window_index=0,
                    chars_per_token=chars_per_token,
                )
                combined = current + candidate
                combined_payload = _page_context_payload(
                    page_rows=page_rows,
                    selected_rows=combined,
                    context_id="temporary",
                    context_kind="page_window",
                    window_index=0,
                    chars_per_token=chars_per_token,
                )
                if current and combined_payload["estimated_context_tokens"] > max_input_tokens:
                    windows.append(current)
                    current = []

                if candidate_payload["estimated_context_tokens"] <= max_input_tokens:
                    current.extend(candidate)
                    continue

                # A single oversized section: pack its chunks in reading order.
                for chunk in candidate:
                    with_chunk = current + [chunk]
                    chunk_payload = _page_context_payload(
                        page_rows=page_rows,
                        selected_rows=with_chunk,
                        context_id="temporary",
                        context_kind="page_window",
                        window_index=0,
                        chars_per_token=chars_per_token,
                    )
                    if current and chunk_payload["estimated_context_tokens"] > max_input_tokens:
                        windows.append(current)
                        current = []
                    single_payload = _page_context_payload(
                        page_rows=page_rows,
                        selected_rows=[chunk],
                        context_id="temporary",
                        context_kind="page_window",
                        window_index=0,
                        chars_per_token=chars_per_token,
                    )
                    if single_payload["estimated_context_tokens"] > max_input_tokens:
                        raise ValueError(
                            f"Chunk {chunk['chunk_id']} exceeds the page input budget by itself; "
                            "re-chunk the source or increase max_input_tokens."
                        )
                    current.append(chunk)
        if current:
            windows.append(current)

        for window_index, window_rows in enumerate(windows, start=1):
            contexts.append(_page_context_payload(
                page_rows=page_rows,
                selected_rows=window_rows,
                context_id=f"{page_id}:window-{window_index:03d}",
                context_kind="page_window",
                window_index=window_index,
                chars_per_token=chars_per_token,
            ))
    return contexts


def source_manifest(sqlite_path: str | Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = [
        {key: row.get(key) for key in ("doc_id", "document_version", "chunk_id", "source_content_hash")}
        for row in rows
    ]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {
        "sqlite_path": str(Path(sqlite_path).resolve()),
        "document_count": len({(r["doc_id"], r["document_version"]) for r in rows}),
        "chunk_count": len(rows),
        "manifest_sha256": digest,
        "rows": payload,
    }


def init_staging_db(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript("""
      CREATE TABLE IF NOT EXISTS contexts (
        context_id TEXT PRIMARY KEY, payload TEXT NOT NULL, question_status TEXT NOT NULL DEFAULT 'pending'
      );
      CREATE TABLE IF NOT EXISTS questions (
        question_id INTEGER PRIMARY KEY, context_id TEXT NOT NULL, question TEXT NOT NULL,
        payload TEXT NOT NULL, answer_status TEXT NOT NULL DEFAULT 'pending',
        UNIQUE(context_id, question)
      );
      CREATE TABLE IF NOT EXISTS examples (
        question_id INTEGER PRIMARY KEY, payload TEXT NOT NULL, accepted INTEGER NOT NULL,
        rejection_reason TEXT
      );
    """)
    conn.commit()
    return conn


class QuestionSet(BaseModel):
    """Structured output for stage one."""

    questions: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)


class AnswerSet(BaseModel):
    """Structured output for stage two."""

    answer: Annotated[str, Field(min_length=1)]
    evidence: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)


def question_messages(context: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": (
        "You create high-quality supervised fine-tuning data for technical documentation. "
        "Return JSON only with a 'questions' array containing 1 to 3 useful questions. "
        "Every question must be answerable solely from the supplied context. Avoid vague, "
        "duplicate, heading-only, or context-dependent questions. Mix factual, conceptual, "
        "procedural, comparison, limitation, and troubleshooting questions when appropriate."
    )}, {"role": "user", "content": f"CONTEXT:\n{context}"}]


def answer_messages(question: str, context: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": (
        "Answer the question using only the supplied documentation context. Return JSON only "
        "with an 'answer' string and an 'evidence' array of short exact supporting excerpts. "
        "Explain first, then give steps, caveats, examples, or limits when the context supports "
        "them. Do not invent commands, facts, URLs, or citations. If the context cannot answer "
        "the question, return an empty answer and evidence array."
    )}, {"role": "user", "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"}]
