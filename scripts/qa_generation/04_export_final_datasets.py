"""Export immutable training and paired-benchmark JSONL snapshots.

Run this only after both Q&A generators and paired benchmark generation finish.
It validates every retained Q&A item's stored context against the current source
corpus, so historical examples are included only when their actual context text
still matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.qa_generation.pipeline import (
    build_contexts,
    build_page_contexts,
    eligible_chunks,
    load_chunks,
    normalize_text,
    source_manifest,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--version", required=True, help="Immutable dataset version, for example v001.")
    value.add_argument("--output-dir", type=Path, help="Defaults to artifacts/datasets/<version>.")
    value.add_argument("--benchmark-size", type=int, default=400)
    value.add_argument("--selection-size", type=int, default=50, help="Fixed checkpoint-selection subset drawn from the benchmark.")
    return value


def file_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                rows += 1
            digest.update(line)
    return {"path": str(path.relative_to(PROJECT_ROOT)), "rows": rows, "sha256": digest.hexdigest()}


def require_finished(conn: sqlite3.Connection, label: str) -> None:
    pending_contexts = conn.execute("SELECT COUNT(*) FROM contexts WHERE question_status = 'pending'").fetchone()[0]
    pending_answers = conn.execute("SELECT COUNT(*) FROM questions WHERE answer_status = 'pending'").fetchone()[0]
    if pending_contexts or pending_answers:
        raise RuntimeError(f"{label} is unfinished: {pending_contexts:,} contexts and {pending_answers:,} answers pending")


def context_maps(source_sqlite: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    all_rows = load_chunks(source_sqlite)
    rows = eligible_chunks(all_rows)
    independent = {item["context_id"]: item for item in build_contexts(rows, max_chars=9_000)}
    page_aware = {
        item["context_id"]: item
        for item in build_page_contexts(rows, max_input_tokens=24_000, chars_per_token=4.0)
    }
    return independent, page_aware, source_manifest(source_sqlite, all_rows)


def same_context(saved: dict[str, Any], current: dict[str, Any] | None) -> bool:
    return bool(current) and normalize_text(saved.get("context", "")) == normalize_text(current.get("context", ""))


def load_training_examples(path: Path, kind: str, current_contexts: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    database_id = str(path.relative_to(PROJECT_ROOT))
    query = """
        SELECT q.question_id, e.payload
        FROM examples e JOIN questions q ON q.question_id = e.question_id
        WHERE e.accepted = 1 AND q.answer_status = 'complete'
        ORDER BY q.question_id
    """
    accepted: list[dict[str, Any]] = []
    excluded = 0
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as conn:
        require_finished(conn, kind)
        for question_id, raw_payload in conn.execute(query):
            payload = json.loads(raw_payload)
            provenance = payload["provenance"]
            context = provenance["context"]
            if not same_context(context, current_contexts.get(context["context_id"])):
                excluded += 1
                continue
            accepted.append({
                "training_example_id": f"{database_id}:{question_id}",
                "dataset_kind": kind,
                "example": payload["example"],
                "provenance": {**provenance, "training_example_id": f"{database_id}:{question_id}", "dataset_kind": kind},
            })
    return accepted, excluded


def load_benchmark(path: Path, expected_size: int, training_ids: set[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    query = """
        SELECT c.training_example_id, e.payload
        FROM evaluation_candidates c
        JOIN evaluation_examples e ON e.training_example_id = c.training_example_id
        WHERE c.evaluation_status = 'complete' AND e.accepted = 1
        ORDER BY c.training_example_id
    """
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as conn:
        rows = [(identifier, json.loads(payload)) for identifier, payload in conn.execute(query)]
    if len(rows) != expected_size:
        raise RuntimeError(f"Expected {expected_size} accepted benchmark items, found {len(rows)}")
    identifiers = {identifier for identifier, _ in rows}
    if len(identifiers) != len(rows):
        raise RuntimeError("Benchmark contains duplicate training example identifiers")
    missing = identifiers - training_ids
    if missing:
        raise RuntimeError(f"{len(missing)} benchmark items do not have a retained source training pair")
    pages = [payload["provenance"]["source_page_id"] for _, payload in rows]
    if len(set(pages)) != len(pages):
        raise RuntimeError("Benchmark must contain one item per distinct source page")
    return [{"training_example_id": identifier, "example": payload["example"], "provenance": payload["provenance"]} for identifier, payload in rows]


def balanced_selection(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """Select a fixed, broad category sample without changing the full benchmark."""
    if size <= 0 or size > len(rows):
        raise ValueError(f"selection size must be between 1 and {len(rows)}")
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_category.setdefault(row["provenance"].get("category", "uncategorized"), []).append(row)
    for group in by_category.values():
        group.sort(key=lambda row: hashlib.sha256(row["training_example_id"].encode()).hexdigest())
    selected: list[dict[str, Any]] = []
    offsets = {category: 0 for category in sorted(by_category)}
    while len(selected) < size:
        progressed = False
        for category in sorted(by_category):
            offset = offsets[category]
            if offset >= len(by_category[category]):
                continue
            selected.append(by_category[category][offset])
            offsets[category] += 1
            progressed = True
            if len(selected) == size:
                break
        if not progressed:
            raise RuntimeError("Not enough benchmark items for the requested selection subset")
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]], field: str) -> None:
    with path.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row[field], ensure_ascii=False) + "\n")


def main() -> None:
    args = parser().parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    source_sqlite = Path(os.environ["SOURCE_SQLITE_PATH"])
    output_dir = (args.output_dir or PROJECT_ROOT / "artifacts" / "datasets" / args.version).resolve()
    page_db = PROJECT_ROOT / "artifacts" / "qa_page_aware" / "generation.sqlite"
    independent_db = PROJECT_ROOT / "artifacts" / "qa_independent" / "generation.sqlite"
    benchmark_db = PROJECT_ROOT / "artifacts" / "qa_evaluation" / "generation.sqlite"
    benchmark_name = f"benchmark_{args.benchmark_size}"
    selection_name = f"benchmark_selection_{args.selection_size}"
    output_names = ("training.jsonl", "training.provenance.jsonl", f"{benchmark_name}.jsonl", f"{benchmark_name}.provenance.jsonl", f"{selection_name}.jsonl", f"{selection_name}.provenance.jsonl", "dataset_manifest.json")
    if output_dir.exists() and any((output_dir / name).exists() for name in output_names):
        raise FileExistsError(f"Refusing to overwrite immutable dataset export: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    independent_contexts, page_contexts, corpus = context_maps(source_sqlite)
    page_examples, page_excluded = load_training_examples(page_db, "page_aware", page_contexts)
    independent_examples, independent_excluded = load_training_examples(independent_db, "independent", independent_contexts)
    training = sorted(page_examples + independent_examples, key=lambda item: item["training_example_id"])
    benchmark = load_benchmark(benchmark_db, args.benchmark_size, {item["training_example_id"] for item in training})
    selection = balanced_selection(benchmark, args.selection_size)

    write_jsonl(output_dir / "training.jsonl", training, "example")
    write_jsonl(output_dir / "training.provenance.jsonl", training, "provenance")
    write_jsonl(output_dir / f"{benchmark_name}.jsonl", benchmark, "example")
    write_jsonl(output_dir / f"{benchmark_name}.provenance.jsonl", benchmark, "provenance")
    write_jsonl(output_dir / f"{selection_name}.jsonl", selection, "example")
    write_jsonl(output_dir / f"{selection_name}.provenance.jsonl", selection, "provenance")
    manifest = {
        "dataset_version": args.version,
        "created_at": datetime.now(UTC).isoformat(),
        "source_manifest_sha256": corpus["manifest_sha256"],
        "training_sources": {
            "page_aware": {"database": str(page_db.relative_to(PROJECT_ROOT)), "exported": len(page_examples), "excluded_context_mismatch": page_excluded},
            "independent": {"database": str(independent_db.relative_to(PROJECT_ROOT)), "exported": len(independent_examples), "excluded_context_mismatch": independent_excluded},
        },
        "benchmark_size": len(benchmark),
        "checkpoint_selection_size": len(selection),
    }
    for name in output_names[:-1]:
        manifest[name] = file_metadata(output_dir / name)
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
