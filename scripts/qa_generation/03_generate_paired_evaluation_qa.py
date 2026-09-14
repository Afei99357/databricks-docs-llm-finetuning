"""Create a fixed paired evaluation benchmark from completed training Q&A.

Each benchmark item is deliberately tied to one accepted training item. The
generator receives that item's grounded evidence and creates a different,
realistic question about the same knowledge. Therefore the model is tested on
unseen wording and task framing, not documentation knowledge it never had a
chance to learn.

This script is resumable. It never changes an already selected benchmark
candidate or an already completed evaluation example.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.qa_generation.pipeline import normalize_text

load_dotenv(PROJECT_ROOT / ".env", override=True)

# Change these values before a new benchmark is created. Once candidates are
# stored, rerunning resumes the exact same benchmark instead of sampling again.
# Add the independent-Q&A database to FINAL_TRAINING_DATABASES before the first
# evaluation run. Every listed database must be complete and must exist.
DEFAULT_TRAINING_DATABASES = "artifacts/qa_page_aware/generation.sqlite"
FINAL_TRAINING_DATABASES = os.getenv("FINAL_TRAINING_DATABASES", DEFAULT_TRAINING_DATABASES)
TRAINING_DB_PATHS = [
    (PROJECT_ROOT / item.strip()).resolve()
    for item in FINAL_TRAINING_DATABASES.split(",")
    if item.strip()
]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "qa_evaluation"
STAGING_PATH = ARTIFACT_DIR / "generation.sqlite"
BENCHMARK_SIZE = 400
EVALUATION_WORKERS = int(os.getenv("EVALUATION_WORKERS", os.getenv("ANSWER_WORKERS", "1")))
PROMPT_REVISION = "paired-knowledge-evaluation-v1"

BASE_URL = os.getenv("GENERATOR_BASE_URL", "http://127.0.0.1:1234/v1")
MODEL = os.getenv("GENERATOR_MODEL", "unsloth/Muse-Glimmer-30B-GGUF:UD-Q4_K_XL")
API_KEY = os.getenv("GENERATOR_API_KEY", "local")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


class EvaluationSet(BaseModel):
    """One alternate question and its source-grounded reference answer."""

    question: Annotated[str, Field(min_length=1)]
    answer: Annotated[str, Field(min_length=1)]
    evidence: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)


def stable_rank(value: str) -> str:
    """Stable order means a fresh benchmark is reproducible without a seed."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_training_candidates() -> list[dict[str, Any]]:
    """Read only accepted, completed Q&A that can legally supply training data."""
    query = """
        SELECT e.question_id, e.payload
        FROM examples AS e
        JOIN questions AS q ON q.question_id = e.question_id
        WHERE e.accepted = 1 AND q.answer_status = 'complete'
        ORDER BY e.question_id
    """
    candidates: list[dict[str, Any]] = []
    if not TRAINING_DB_PATHS:
        raise ValueError("FINAL_TRAINING_DATABASES must contain at least one database path")
    for database_path in TRAINING_DB_PATHS:
        if not database_path.exists():
            raise FileNotFoundError(f"Final training Q&A database not found: {database_path}")
        database_id = str(database_path.relative_to(PROJECT_ROOT))
        with sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True) as conn:
            pending_questions = conn.execute(
                "SELECT COUNT(*) FROM questions WHERE answer_status = 'pending'"
            ).fetchone()[0]
            pending_contexts = conn.execute(
                "SELECT COUNT(*) FROM contexts WHERE question_status = 'pending'"
            ).fetchone()[0]
            if pending_questions or pending_contexts:
                raise RuntimeError(
                    f"{database_id} is not finished: {pending_contexts:,} contexts and "
                    f"{pending_questions:,} answers are still pending. Finish final training "
                    "Q&A generation before creating the benchmark."
                )
            for question_id, raw_payload in conn.execute(query):
                payload = json.loads(raw_payload)
                provenance = payload["provenance"]
                context = provenance["context"]
                messages = payload["example"]["messages"]
                evidence = [item.strip() for item in provenance.get("evidence", []) if item.strip()]
                if len(messages) != 2 or not evidence:
                    continue
                source_page_id = f"{context['doc_id']}:{context['document_version']}"
                training_example_id = f"{database_id}:{question_id}"
                candidates.append({
                    "training_example_id": training_example_id,
                    "training_question_id": question_id,
                    "training_database": database_id,
                    "source_page_id": source_page_id,
                    "category": context.get("category") or "uncategorized",
                    "training_question": messages[0]["content"].strip(),
                    "training_answer": messages[1]["content"].strip(),
                    "evidence": evidence,
                    "source_title": context.get("source_title", ""),
                    "source_url": context.get("source_url", ""),
                    "source_context_id": context.get("context_id", ""),
                })
    return candidates


def balanced_selection(candidates: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """Give each available documentation category a chance before filling slots."""
    if size <= 0:
        raise ValueError("BENCHMARK_SIZE must be positive")
    if len(candidates) < size:
        raise ValueError(f"Need {size} accepted training pairs, but only found {len(candidates)}")

    # First choose one reproducible representative training Q&A for every page.
    # This makes the final benchmark exactly one item per source page, even when
    # a page produced several questions or was split into generation windows.
    one_per_page: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        current = one_per_page.get(candidate["source_page_id"])
        if current is None or stable_rank(candidate["training_example_id"]) < stable_rank(current["training_example_id"]):
            one_per_page[candidate["source_page_id"]] = candidate
    page_candidates = list(one_per_page.values())
    if len(page_candidates) < size:
        raise ValueError(f"Need {size} distinct source pages, but only found {len(page_candidates)}")

    by_category: dict[str, list[dict[str, Any]]] = {}
    for candidate in page_candidates:
        by_category.setdefault(candidate["category"], []).append(candidate)
    for group in by_category.values():
        group.sort(key=lambda item: stable_rank(item["source_page_id"]))

    # Round-robin selection keeps the 400 pages broadly represented across
    # documentation categories rather than letting large categories dominate.
    selected: list[dict[str, Any]] = []
    category_offsets = {category: 0 for category in sorted(by_category)}
    while len(selected) < size:
        made_progress = False
        for category in sorted(by_category):
            offset = category_offsets[category]
            if offset >= len(by_category[category]):
                continue
            selected.append(by_category[category][offset])
            category_offsets[category] += 1
            made_progress = True
            if len(selected) == size:
                break
        if not made_progress:
            raise RuntimeError("Ran out of distinct pages before reaching BENCHMARK_SIZE")
    return selected


def init_evaluation_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS evaluation_candidates (
            training_example_id TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            payload TEXT NOT NULL,
            evaluation_status TEXT NOT NULL DEFAULT 'pending'
        );
        CREATE TABLE IF NOT EXISTS evaluation_examples (
            training_example_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            accepted INTEGER NOT NULL,
            rejection_reason TEXT
        );
    """)
    conn.commit()
    return conn


def evaluation_messages(candidate: dict[str, Any]) -> list[dict[str, str]]:
    evidence = "\n\n".join(f"- {item}" for item in candidate["evidence"])
    return [
        {
            "role": "system",
            "content": (
                "Create exactly one evaluation item for a fine-tuned Databricks documentation model. "
                "Return JSON only with question, answer, and evidence. The new question must test the same "
                "documented knowledge as the training item, but use materially different wording, structure, "
                "or a realistic scenario. It must not be a paraphrase or repeat of the training question. "
                "Answer only from the supplied evidence excerpts. Do not add facts, commands, URLs, or citations."
            ),
        },
        {
            "role": "user",
            "content": (
                f"SOURCE TITLE: {candidate['source_title']}\n\n"
                f"TRAINING QUESTION (do not repeat or lightly paraphrase):\n{candidate['training_question']}\n\n"
                f"TRAINING ANSWER (use only to understand the knowledge target; do not copy it):\n"
                f"{candidate['training_answer']}\n\n"
                f"GROUNDING EVIDENCE:\n{evidence}"
            ),
        },
    ]


def is_too_similar(training_question: str, evaluation_question: str) -> bool:
    """Reject exact and obvious near-duplicate question wording."""
    original = normalize_text(training_question)
    alternate = normalize_text(evaluation_question)
    return original == alternate or SequenceMatcher(None, original, alternate).ratio() >= 0.85


def generate_one(client: OpenAI, candidate: dict[str, Any]):
    training_example_id = candidate["training_example_id"]
    try:
        completion = client.chat.completions.parse(
            model=MODEL,
            messages=evaluation_messages(candidate),
            temperature=1.0,
            response_format=EvaluationSet,
        )
        result = completion.choices[0].message.parsed
        question = result.question.strip()
        answer = result.answer.strip()
        evidence = [item.strip() for item in result.evidence if item.strip()]
        if is_too_similar(candidate["training_question"], question):
            return training_example_id, None, 0, "evaluation question is too similar to its training question", None
        if not answer or not evidence:
            return training_example_id, None, 0, "empty answer or malformed evidence", None
        payload = {
            "example": {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            },
            "provenance": {
                **candidate,
                "evaluation_question": question,
                "evaluation_evidence": evidence,
                "model": MODEL,
                "prompt_revision": PROMPT_REVISION,
            },
        }
        return training_example_id, payload, 1, None, None
    except Exception as exc:
        return training_example_id, None, 0, None, "error:" + str(exc)[:500]


def main() -> None:
    print({
        "final_training_databases": [str(path) for path in TRAINING_DB_PATHS],
        "artifacts": str(ARTIFACT_DIR),
        "benchmark_size": BENCHMARK_SIZE,
        "workers": EVALUATION_WORKERS,
        "model": MODEL,
        "base_url": BASE_URL,
    })
    db = init_evaluation_db(STAGING_PATH)
    existing_count = db.execute("SELECT COUNT(*) FROM evaluation_candidates").fetchone()[0]
    if existing_count == 0:
        selected = balanced_selection(load_training_candidates(), BENCHMARK_SIZE)
        db.executemany(
            "INSERT INTO evaluation_candidates(training_example_id, category, payload) VALUES (?, ?, ?)",
            [
                (item["training_example_id"], item["category"], json.dumps(item))
                for item in selected
            ],
        )
        db.commit()
        print(f"selected and froze {len(selected):,} paired benchmark candidates")
    else:
        print(f"using existing frozen benchmark candidates: {existing_count:,}")

    pending = db.execute(
        "SELECT training_example_id, payload FROM evaluation_candidates "
        "WHERE evaluation_status = 'pending' ORDER BY training_example_id"
    ).fetchall()
    print(f"evaluation items pending: {len(pending):,}")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    with ThreadPoolExecutor(max_workers=max(1, EVALUATION_WORKERS)) as pool:
        futures = [pool.submit(generate_one, client, json.loads(payload)) for _, payload in pending]
        for future in tqdm(as_completed(futures), total=len(futures), desc="paired evaluation"):
            training_example_id, payload, accepted, rejection_reason, error = future.result()
            if error:
                db.execute(
                    "UPDATE evaluation_candidates SET evaluation_status = ? WHERE training_example_id = ?",
                    (error, training_example_id),
                )
            else:
                db.execute(
                    "INSERT OR REPLACE INTO evaluation_examples(training_example_id, payload, accepted, rejection_reason) "
                    "VALUES (?, ?, ?, ?)",
                    (training_example_id, json.dumps(payload) if payload else "{}", accepted, rejection_reason),
                )
                db.execute(
                    "UPDATE evaluation_candidates SET evaluation_status = 'complete' WHERE training_example_id = ?",
                    (training_example_id,),
                )
            db.commit()

    print("Evaluation status:")
    for status, count in db.execute(
        "SELECT evaluation_status, COUNT(*) FROM evaluation_candidates "
        "GROUP BY evaluation_status ORDER BY COUNT(*) DESC"
    ):
        print(f"  {status}: {count:,}")
    print("Acceptance:")
    for accepted, count in db.execute(
        "SELECT accepted, COUNT(*) FROM evaluation_examples GROUP BY accepted ORDER BY accepted DESC"
    ):
        print(f"  {bool(accepted)}: {count:,}")


if __name__ == "__main__":
    main()
