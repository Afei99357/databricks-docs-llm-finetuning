# Databricks documentation fine-tuning

Build and evaluate local Qwen QLoRA adapters from the current Databricks
documentation corpus.

The project creates two training datasets, then tests the resulting model on
newly phrased questions about knowledge it was trained to cover. It also
compares the best adapter with the local RAG application.

## Goal and evaluation design

The goal is not to replace RAG for current, cited answers. RAG remains the
right tool when documentation freshness and citations matter. This project
tests whether fine-tuning improves a local model's learned Databricks product
knowledge and answer behavior.

The earlier document-heldout evaluation design was intentionally retired. It
could test facts that the fine-tuned model had never been allowed to learn.
Instead, the paired benchmark uses this relationship:

```text
accepted training Q&A
        │ same documented knowledge and evidence
        ▼
alternate evaluation Q&A
        │ different question wording or realistic scenario
        ▼
unseen exact question and answer for evaluation
```

The model may learn the underlying knowledge from training, but it never sees
the benchmark's exact question/answer pair during optimization.

## Data flow

```text
Databricks documentation SQLite (read-only)
        │
        ├── page-aware Q&A ──── artifacts/qa_page_aware/generation.sqlite
        ├── independent Q&A ─── artifacts/qa_independent/generation.sqlite
        │
        ├── paired benchmark Q&A ─ artifacts/qa_evaluation/generation.sqlite
        ▼
immutable versioned JSONL snapshot + manifest
        │
        ├── training.jsonl
        ├── benchmark_selection_50.jsonl
        └── benchmark_400.jsonl
        ▼
QLoRA checkpoints, best adapter, MLflow records
        ▼
400-question model evaluation and model-versus-RAG comparison
```

## Prerequisites

- Current documentation corpus: `/home/eric/Projects/databricks_docs_rag/data/local.sqlite`
- Local OpenAI-compatible generation server for Q&A and judge generation
- ROCm-compatible AMD GPU for fine-tuning
- Project dependencies installed with `uv`

The source SQLite database is read-only from this project.

## Generation state and resumability

Each Q&A generator stores durable state in SQLite. It can be stopped and
rerun: completed questions and answers are retained, while pending work
resumes. The two active state databases are:

```text
artifacts/qa_page_aware/generation.sqlite
artifacts/qa_independent/generation.sqlite
```

Page-aware generation uses one whole page where possible and heading-aware
windows only for oversized pages. Independent generation uses a chunk with its
neighbours from the same document. The final export validates each saved Q&A
context against the current source corpus before including it in training.

## Workflow

Run the numbered scripts in order.

The same commands are available as short `just` recipes; run `just --list` to
see them. For example, `just page-aware-qa`, `just export-dataset v001`, and
`just train-smoke configs/experiments/05_qwen35_4b_v001.yaml`.

```bash
# 1–2. Generate training Q&A
uv run python scripts/qa_generation/01_generate_page_aware_qa.py
uv run python scripts/qa_generation/02_generate_independent_qa.py

# 3. Generate 400 paired evaluation Q&A items
uv run python scripts/qa_generation/03_generate_paired_evaluation_qa.py

# 4. Export immutable training, benchmark, and manifest files
uv run python scripts/qa_generation/04_export_final_datasets.py --version v001

# 5. Fine-tune a 4B Qwen model first
utils/rocm-run python scripts/training/05_finetune_llm.py \
  --config configs/experiments/05_qwen35_4b_v001.yaml \
  --smoke-test --run

# 6. Evaluate base model, checkpoints, and final adapters on all 400 questions
utils/rocm-run python scripts/evaluation/06_evaluate_checkpoints.py \
  --run-dir artifacts/models/qwen35-4b-v001-smoke \
  --dataset-dir artifacts/datasets/v001

# 7. Compare the selected adapter with the local RAG API
uv run python scripts/evaluation/07_compare_winner_with_rag.py \
  --run-dir artifacts/models/qwen35-4b-v001-smoke \
  --model-results artifacts/models/qwen35-4b-v001-smoke/evaluations/best_adapter-benchmark_400.json
```

Step 07 requires the RAG application to be running locally at
`http://127.0.0.1:8000/api/answer`.

## Main workflow scripts

| Step | Script | Purpose | Reads | Creates / updates |
|---:|---|---|---|---|
| 01 | `scripts/qa_generation/01_generate_page_aware_qa.py` | Generate broad, page-structured training Q&A. | Current source SQLite; local generation model. | `artifacts/qa_page_aware/generation.sqlite` |
| 02 | `scripts/qa_generation/02_generate_independent_qa.py` | Generate independent chunk-plus-neighbour training Q&A for broader coverage. | Current source SQLite; local generation model. | `artifacts/qa_independent/generation.sqlite` |
| 03 | `scripts/qa_generation/03_generate_paired_evaluation_qa.py` | Select 400 distinct source pages and generate alternate evaluation Q&A from training knowledge units. | Completed page-aware and independent Q&A; local generation model. | `artifacts/qa_evaluation/generation.sqlite` |
| 04 | `scripts/qa_generation/04_export_final_datasets.py` | Validate current contexts, combine both training sources, and create immutable dataset snapshots. | Three generation SQLite databases; current source SQLite. | Versioned `training.jsonl`, 50/400 benchmark JSONL, provenance files, and manifest. |
| 05 | `scripts/training/05_finetune_llm.py` | Fine-tune a configured Qwen model with QLoRA and select the best checkpoint using the fixed 50-question set. | Versioned training dataset, manifest, experiment YAML. | Checkpoints, `best_checkpoint/`, adapters, run reports, MLflow run. |
| 06 | `scripts/evaluation/06_evaluate_checkpoints.py` | Generate real answers for the base model, saved checkpoints, and adapters; judge them on all 400 items. | Model run directory, full benchmark, local judge model. | Per-candidate answers, scores, latency, MLflow evaluation runs. |
| 07 | `scripts/evaluation/07_compare_winner_with_rag.py` | Compare the selected adapter and local RAG answer-by-answer on the same benchmark. | Saved model evaluation results, running RAG API, local judge model. | Pairwise wins, per-question comparison file, MLflow comparison run. |

`scripts/training/test_checkpoint.py` is an optional manual probe. It is not a
numbered workflow step: use it when you want to inspect one saved checkpoint
with your own question before running the full benchmark.

## Datasets

- `qa_page_aware`: page-level contexts; preserves document structure.
- `qa_independent`: chunk and adjacent-context questions; broadens coverage.
- `benchmark_400`: one alternate evaluation question from each of 400 distinct
  source pages. It is never included in training.
- `benchmark_selection_50`: a fixed category-balanced subset used only for
  selecting the best checkpoint during training.

Step 04 writes immutable versioned JSONL files under `artifacts/datasets/`.
It refuses to overwrite an existing version. The accompanying manifest records
row counts, hashes, source-corpus fingerprint, and excluded context mismatches.

## MLflow

Fine-tuning, manual checkpoint probes, benchmark evaluation, and RAG
comparison all log parameters, metrics, and artifacts to local MLflow.

The training script retains the newest two resumable checkpoints and separately
preserves the best checkpoint and adapter according to loss on the fixed
50-question selection set. The final decision uses answer quality on all 400
benchmark questions.

The important selection distinction is:

| During training | Final model decision |
|---|---|
| Fixed 50 paired questions; teacher-forced `eval_loss` every checkpoint interval | All 400 paired questions; generated answers and judge scores |
| Selects and preserves `best_checkpoint/` | Compares base, checkpoints, final/best adapter, and RAG |

Normal checkpoints include optimizer, scheduler, RNG, and adapter state for
resuming. `best_adapter/` is the inference-ready copy; `best_checkpoint/`
retains the full resumable training state.

## Configuration

`.env` contains local service settings such as the Q&A/judge model endpoint and
worker counts. The source database path is `SOURCE_SQLITE_PATH`.

Each material training experiment gets a new YAML file under
`configs/experiments/`. Never edit a configuration after its training run has
started. Start with the 4B smoke configuration, then make a separate 9B config
only after the full workflow works.

## Layout

```text
scripts/      runnable workflow steps and their shared helpers
artifacts/    generation state, datasets, adapters, and evaluation outputs
configs/      versioned experiment settings
utils/        ROCm and local-run helpers
archive/      historical code and experiments; not part of the active workflow
```

## Status

Q&A generation is in progress. Dataset export, training, and evaluation scripts
are implemented but should not run until both Q&A datasets are complete.
