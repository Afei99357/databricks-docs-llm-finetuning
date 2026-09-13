from __future__ import annotations

import json
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.qa_generation.pipeline import (
    AnswerSet, QuestionSet, answer_messages, build_page_contexts, eligible_chunks,
    init_staging_db, load_chunks, source_manifest,
)

# override=True ensures a restarted or previously-used kernel reads the current .env values.
load_dotenv(PROJECT_ROOT / '.env', override=True)
SOURCE_SQLITE_PATH = Path(os.getenv('SOURCE_SQLITE_PATH', PROJECT_ROOT / 'data/local.sqlite'))
BASE_URL = os.getenv('GENERATOR_BASE_URL', 'http://127.0.0.1:1234/v1')
MODEL = os.getenv('GENERATOR_MODEL', 'unsloth/Muse-Glimmer-30B-GGUF:UD-Q4_K_XL')
API_KEY = os.getenv('GENERATOR_API_KEY', 'local')
ARTIFACT_DIR = PROJECT_ROOT / 'artifacts' / 'qa_page_aware'
STAGING_PATH = ARTIFACT_DIR / 'generation.sqlite'
PAGE_INPUT_BUDGET = 24_000  # Estimated input tokens: safe within a 32k server context.
CHARS_PER_TOKEN = 4.0       # Conservative estimate until we use the serving tokenizer.
QUESTION_WORKERS = int(os.getenv('QUESTION_WORKERS', '1'))
ANSWER_WORKERS = int(os.getenv('ANSWER_WORKERS', '1'))
RUN_MODE = 'full'  # 'pilot' processes PILOT_PAGES; 'full' processes all pending work.
PILOT_PAGES = 10
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print({'source': str(SOURCE_SQLITE_PATH), 'model': MODEL, 'base_url': BASE_URL, 'artifacts': str(ARTIFACT_DIR)})

# Read the source database. It is opened read-only by load_chunks.
all_rows = load_chunks(SOURCE_SQLITE_PATH)
rows = eligible_chunks(all_rows)
contexts = build_page_contexts(
    rows,
    max_input_tokens=PAGE_INPUT_BUDGET,
    chars_per_token=CHARS_PER_TOKEN,
)
manifest = source_manifest(SOURCE_SQLITE_PATH, all_rows)
manifest.update({
    'generation_strategy': 'page-aware-training-v1',
    'max_input_tokens': PAGE_INPUT_BUDGET,
    'chars_per_token': CHARS_PER_TOKEN,
    'context_count': len(contexts),
    'full_page_context_count': sum(c['context_kind'] == 'full_page' for c in contexts),
    'page_window_context_count': sum(c['context_kind'] == 'page_window' for c in contexts),
})
(ARTIFACT_DIR / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

full_page_contexts = [context for context in contexts if context['context_kind'] == 'full_page']
window_contexts = [context for context in contexts if context['context_kind'] == 'page_window']
processed_page_ids = {(context['doc_id'], context['document_version']) for context in contexts}
oversized_page_ids = {(context['doc_id'], context['document_version']) for context in window_contexts}
excluded_pages = manifest['document_count'] - len(processed_page_ids)

print(f"active pages: {manifest['document_count']:,}; eligible chunks: {len(rows):,}")
print(f"pages used for generation: {len(processed_page_ids):,}; excluded after filtering: {excluded_pages:,}")
print(f"generation contexts: {len(contexts):,} = {len(full_page_contexts):,} full-page contexts + {len(window_contexts):,} windows from {len(oversized_page_ids):,} oversized pages")
print(f"largest context: {max(c['estimated_context_tokens'] for c in contexts):,} estimated tokens")
if RUN_MODE not in {'pilot', 'full'}:
    raise ValueError("RUN_MODE must be 'pilot' or 'full'")
contexts[:1]

# Inspect a windowed context. This should show only pages that exceeded the page budget.
windowed_contexts = [c for c in contexts if c['context_kind'] == 'page_window']
print(f"oversized-page windows: {len(windowed_contexts):,}")
windowed_contexts[:1]

# The v2 state database is new. INSERT OR IGNORE makes this cell safe to rerun.
db = init_staging_db(STAGING_PATH)
for context in contexts:
    db.execute(
        'INSERT OR IGNORE INTO contexts(context_id, payload) VALUES (?, ?)',
        (context['context_id'], json.dumps(context)),
    )
db.commit()
print(f"staging contexts: {db.execute('SELECT COUNT(*) FROM contexts').fetchone()[0]:,}")

# A pilot automatically selects the next contexts that still need either stage.
# This makes reruns resume unfinished pilot work before moving to fresh pages.
if RUN_MODE == 'pilot':
    pilot_rows = db.execute(
        """SELECT c.context_id
           FROM contexts c
           WHERE c.question_status = 'pending'
              OR EXISTS (
                  SELECT 1 FROM questions q
                  WHERE q.context_id = c.context_id AND q.answer_status = 'pending'
              )
           ORDER BY c.context_id
           LIMIT ?""",
        (PILOT_PAGES,),
    ).fetchall()
    run_context_ids = {context_id for (context_id,) in pilot_rows}
else:
    run_context_ids = None
print(f"run mode: {RUN_MODE}; selected contexts: {len(run_context_ids) if run_context_ids is not None else len(contexts):,}")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def generate_structured(messages, response_model):
    completion = client.chat.completions.parse(
        model=MODEL,
        messages=messages,
        temperature=1.0,
        response_format=response_model,
    )
    return completion.choices[0].message.parsed

def page_question_messages(context):
    return [
        {'role': 'system', 'content': (
            'You create high-quality supervised fine-tuning data from one Databricks documentation page or a clearly labeled page window. '
            "Return JSON only with a 'questions' array containing 3 to 6 realistic user questions for substantive pages. "
            'Use 1 to 2 questions only when the supplied page is genuinely short or supports very few distinct topics. Cover different topics; do not create near-duplicate questions. '
            'Prefer questions that connect related facts, conditions, steps, limitations, or trade-offs when the page supports that. '
            'Do not ask questions whose answer requires omitted parts of an oversized page. Avoid vague, duplicate, heading-only, or context-dependent questions.'
        )},
        {'role': 'user', 'content': f"PAGE CONTEXT:\n{context['context']}"},
    ]

def generate_one_question_set(item):
    context_id, payload = item
    try:
        context = json.loads(payload)
        result = generate_structured(page_question_messages(context), QuestionSet)
        return context_id, [q.strip() for q in result.questions if q.strip()], None
    except Exception as exc:
        return context_id, [], 'error:' + str(exc)[:500]

def generate_questions(workers=QUESTION_WORKERS):
    pending = db.execute(
        "SELECT context_id, payload FROM contexts WHERE question_status='pending' ORDER BY context_id",
    ).fetchall()
    if run_context_ids is not None:
        pending = [item for item in pending if item[0] in run_context_ids]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(generate_one_question_set, item) for item in pending]
        for future in tqdm(as_completed(futures), total=len(futures), desc='questions'):
            context_id, questions, error = future.result()
            if error:
                db.execute('UPDATE contexts SET question_status=? WHERE context_id=?', (error, context_id))
            else:
                db.executemany(
                    'INSERT OR IGNORE INTO questions(context_id, question, payload) VALUES (?, ?, ?)',
                    [(context_id, question, json.dumps({'question': question})) for question in questions],
                )
                db.execute("UPDATE contexts SET question_status='complete' WHERE context_id=?", (context_id,))
            db.commit()

generate_questions()

def generate_one_answer(item):
    question_id, question, context_payload = item
    try:
        context = json.loads(context_payload)
        result = generate_structured(answer_messages(question, context['context']), AnswerSet)
        answer = result.answer.strip()
        evidence = [excerpt.strip() for excerpt in result.evidence if excerpt.strip()]
        accepted = bool(answer) and bool(evidence)
        example = {'messages': [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]}
        provenance = {
            'question_id': question_id, 'question': question, 'context': context, 'evidence': evidence,
            'model': MODEL, 'prompt_revision': 'page-aware-training-v1',
            'source_manifest_sha256': manifest['manifest_sha256'],
        }
        payload = {'example': example, 'provenance': provenance}
        reason = None if accepted else 'empty answer or malformed evidence'
        return question_id, payload, int(accepted), reason, None
    except Exception as exc:
        return question_id, None, 0, None, 'error:' + str(exc)[:500]

def generate_answers(workers=ANSWER_WORKERS):
    pending = db.execute(
        "SELECT q.question_id, q.question, c.payload, q.context_id FROM questions q JOIN contexts c ON c.context_id=q.context_id WHERE q.answer_status='pending' ORDER BY q.question_id",
    ).fetchall()
    if run_context_ids is not None:
        pending = [item for item in pending if item[3] in run_context_ids]
    pending = [(question_id, question, payload) for question_id, question, payload, _ in pending]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(generate_one_answer, item) for item in pending]
        for future in tqdm(as_completed(futures), total=len(futures), desc='answers'):
            question_id, payload, accepted, reason, error = future.result()
            if error:
                db.execute('UPDATE questions SET answer_status=? WHERE question_id=?', (error, question_id))
            else:
                db.execute(
                    'INSERT OR REPLACE INTO examples(question_id, payload, accepted, rejection_reason) VALUES (?, ?, ?, ?)',
                    (question_id, json.dumps(payload), accepted, reason),
                )
                db.execute("UPDATE questions SET answer_status='complete' WHERE question_id=?", (question_id,))
            db.commit()

generate_answers()

# Live status and quality spot check. Safe to rerun while generation is active.
print('Context status:')
for status, count in db.execute('SELECT question_status, COUNT(*) FROM contexts GROUP BY question_status ORDER BY COUNT(*) DESC'):
    print(f'  {status}: {count:,}')

print('Answer status:')
for status, count in db.execute('SELECT answer_status, COUNT(*) FROM questions GROUP BY answer_status ORDER BY COUNT(*) DESC'):
    print(f'  {status}: {count:,}')

recent = db.execute(
    "SELECT q.question, json_extract(e.payload, '$.example.messages[1].content') AS answer FROM questions q JOIN examples e ON e.question_id=q.question_id WHERE q.answer_status='complete' ORDER BY q.question_id DESC LIMIT 3"
).fetchall()
recent
