"""Generate independent adjacent-chunk Q&A for the active documentation corpus.

Unlike the page-aware generator, each context is one eligible chunk plus its
neighbours from the same document. Its state is isolated in
``artifacts/qa_independent`` and is safe to stop and rerun.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.qa_generation.pipeline import (
    AnswerSet,
    QuestionSet,
    answer_messages,
    build_contexts,
    eligible_chunks,
    init_staging_db,
    load_chunks,
    question_messages,
    source_manifest,
)

load_dotenv(PROJECT_ROOT / ".env", override=True)
SOURCE_SQLITE_PATH = Path(os.getenv("SOURCE_SQLITE_PATH", PROJECT_ROOT / "data/local.sqlite"))
BASE_URL = os.getenv("GENERATOR_BASE_URL", "http://127.0.0.1:1234/v1")
MODEL = os.getenv("GENERATOR_MODEL", "unsloth/Muse-Glimmer-30B-GGUF:UD-Q4_K_XL")
API_KEY = os.getenv("GENERATOR_API_KEY", "local")
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "qa_independent"
STAGING_PATH = ARTIFACT_DIR / "generation.sqlite"
MAX_CONTEXT_CHARS = 9_000
QUESTION_WORKERS = int(os.getenv("QUESTION_WORKERS", "1"))
ANSWER_WORKERS = int(os.getenv("ANSWER_WORKERS", "1"))
PROMPT_REVISION = "independent-adjacent-chunk-v2"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def generate_structured(client: OpenAI, messages, response_model):
    completion = client.chat.completions.parse(
        model=MODEL,
        messages=messages,
        temperature=1.0,
        response_format=response_model,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("generator returned no parsed response")
    return parsed


def main() -> None:
    print({"source": str(SOURCE_SQLITE_PATH), "model": MODEL, "base_url": BASE_URL, "artifacts": str(ARTIFACT_DIR)})
    all_rows = load_chunks(SOURCE_SQLITE_PATH)
    rows = eligible_chunks(all_rows)
    contexts = build_contexts(rows, max_chars=MAX_CONTEXT_CHARS)
    manifest = source_manifest(SOURCE_SQLITE_PATH, all_rows)
    manifest.update({
        "generation_strategy": "independent-adjacent-chunk-v2",
        "max_context_chars": MAX_CONTEXT_CHARS,
        "context_count": len(contexts),
    })
    (ARTIFACT_DIR / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"active pages: {manifest['document_count']:,}; eligible chunks and contexts: {len(contexts):,}")

    db = init_staging_db(STAGING_PATH)
    for context in contexts:
        db.execute(
            "INSERT OR IGNORE INTO contexts(context_id, payload) VALUES (?, ?)",
            (context["context_id"], json.dumps(context)),
        )
    db.commit()

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    def generate_one_question(item):
        context_id, payload = item
        try:
            context = json.loads(payload)
            result = generate_structured(client, question_messages(context["context"]), QuestionSet)
            return context_id, [question.strip() for question in result.questions if question.strip()], None
        except Exception as exc:
            return context_id, [], "error:" + str(exc)[:500]

    pending_contexts = db.execute(
        "SELECT context_id, payload FROM contexts WHERE question_status = 'pending' ORDER BY context_id"
    ).fetchall()
    with ThreadPoolExecutor(max_workers=max(1, QUESTION_WORKERS)) as pool:
        futures = [pool.submit(generate_one_question, item) for item in pending_contexts]
        for future in tqdm(as_completed(futures), total=len(futures), desc="independent questions"):
            context_id, questions, error = future.result()
            if error:
                db.execute("UPDATE contexts SET question_status = ? WHERE context_id = ?", (error, context_id))
            else:
                db.executemany(
                    "INSERT OR IGNORE INTO questions(context_id, question, payload) VALUES (?, ?, ?)",
                    [(context_id, question, json.dumps({"question": question})) for question in questions],
                )
                db.execute("UPDATE contexts SET question_status = 'complete' WHERE context_id = ?", (context_id,))
            db.commit()

    def generate_one_answer(item):
        question_id, question, context_payload = item
        try:
            context = json.loads(context_payload)
            result = generate_structured(client, answer_messages(question, context["context"]), AnswerSet)
            answer = result.answer.strip()
            evidence = [excerpt.strip() for excerpt in result.evidence if excerpt.strip()]
            accepted = bool(answer) and bool(evidence)
            payload = {
                "example": {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]},
                "provenance": {
                    "question_id": question_id,
                    "question": question,
                    "context": context,
                    "evidence": evidence,
                    "model": MODEL,
                    "prompt_revision": PROMPT_REVISION,
                    "source_manifest_sha256": manifest["manifest_sha256"],
                },
            }
            reason = None if accepted else "empty answer or malformed evidence"
            return question_id, payload, int(accepted), reason, None
        except Exception as exc:
            return question_id, None, 0, None, "error:" + str(exc)[:500]

    pending_answers = db.execute(
        "SELECT q.question_id, q.question, c.payload FROM questions q "
        "JOIN contexts c ON c.context_id = q.context_id "
        "WHERE q.answer_status = 'pending' ORDER BY q.question_id"
    ).fetchall()
    with ThreadPoolExecutor(max_workers=max(1, ANSWER_WORKERS)) as pool:
        futures = [pool.submit(generate_one_answer, item) for item in pending_answers]
        for future in tqdm(as_completed(futures), total=len(futures), desc="independent answers"):
            question_id, payload, accepted, reason, error = future.result()
            if error:
                db.execute("UPDATE questions SET answer_status = ? WHERE question_id = ?", (error, question_id))
            else:
                db.execute(
                    "INSERT OR REPLACE INTO examples(question_id, payload, accepted, rejection_reason) VALUES (?, ?, ?, ?)",
                    (question_id, json.dumps(payload), accepted, reason),
                )
                db.execute("UPDATE questions SET answer_status = 'complete' WHERE question_id = ?", (question_id,))
            db.commit()

    print("Context status:")
    for status, count in db.execute("SELECT question_status, COUNT(*) FROM contexts GROUP BY question_status ORDER BY COUNT(*) DESC"):
        print(f"  {status}: {count:,}")
    print("Answer status:")
    for status, count in db.execute("SELECT answer_status, COUNT(*) FROM questions GROUP BY answer_status ORDER BY COUNT(*) DESC"):
        print(f"  {status}: {count:,}")


if __name__ == "__main__":
    main()
