# Databricks documentation LLM fine-tuning

Can a local fine-tuned LLM answer stable Databricks documentation questions
well enough to complement a citation-backed RAG system?

This project builds a reproducible experiment to answer that question. It
creates grounded Q&A data from a local Databricks documentation corpus,
fine-tunes Qwen with QLoRA on an AMD GPU, and evaluates the resulting adapter
against its base model and a local RAG application.

The goal is not to replace RAG. RAG remains the route for fresh,
evidence-sensitive, or citation-required questions. This project tests where a
local adapter can be useful for stable, recurring knowledge and where it should
fall back to RAG.

## What the project does

```text
Databricks documentation SQLite corpus
        |
        +-- page-aware Q&A: preserve a page's structure and headings
        |
        +-- independent Q&A: cover chunks and their immediate context
        |
        +-- paired benchmark Q&A: alternate questions over the same knowledge
        v
Immutable training and benchmark JSONL snapshots
        v
Qwen QLoRA fine-tuning with MLflow tracking
        v
Base model vs. adapter evaluation
        v
Best local adapter vs. local RAG comparison
```

The paired benchmark is deliberately separate from training. The adapter may
learn the documented knowledge, but it never sees the benchmark's exact
question-and-answer pairs during training.

## Repository layout

```text
scripts/
  qa_generation/     numbered data-generation and dataset-export stages
  training/          QLoRA training and MLflow helper
  evaluation/        model and RAG comparison stages
configs/experiments/ versioned training configurations
utils/               ROCm runtime and dependency helpers
docs/                focused implementation notes
artifacts/           local generation state, datasets, models, and results
```

`artifacts/`, `.env`, MLflow data, and local model environments are ignored by
Git. The repository contains the code, configuration, and documentation needed
to reproduce an experiment without publishing local data or credentials.

## Local setup

This project expects:

- a local Databricks documentation SQLite corpus;
- a local OpenAI-compatible server for Q&A generation and judging;
- an AMD ROCm-capable GPU for fine-tuning; and
- dependencies installed with `uv`.

Create your local configuration:

```bash
cp .env.example .env
```

Set `SOURCE_SQLITE_PATH` in `.env` to your local documentation corpus. Start
your generation server before running any Q&A stage.

Validate the GPU environment:

```bash
just sync-rocm-deps
just check-rocm
```

## Run the experiment

Run these stages from the repository root. `just` is the recommended interface:
it supplies the ROCm runtime automatically for GPU work.

```bash
# 1. Generate page-aware training Q&A.
just page-aware-qa

# 2. Generate independent chunk-and-neighbour Q&A.
just independent-qa

# 3. Generate 400 paired evaluation Q&A items.
just paired-evaluation-qa

# 4. Validate the generated data and create immutable versioned datasets.
just export-dataset v001

# 5. Run a conservative 4B QLoRA smoke test.
just train-smoke configs/experiments/05_qwen35_4b_v001.yaml

# 6a. Generate answers across the full 400-question benchmark.
just generate-checkpoint-answers \
  artifacts/models/qwen35-4b-v001-smoke \
  artifacts/datasets/v001

# 6b. Judge the saved answers using the local Muse llama.cpp server.
just judge-checkpoint-answers \
  artifacts/models/qwen35-4b-v001-smoke

# 7. Compare the selected adapter with the running local RAG application.
just compare-rag \
  artifacts/models/qwen35-4b-v001-smoke \
  artifacts/models/qwen35-4b-v001-smoke/evaluations/best_adapter-benchmark_400.json
```

Each Q&A generation stage is resumable. Its SQLite database records completed,
pending, and failed work, so rerunning a stage retries only unfinished items.
Do not run stages 03–07 until both training-Q&A generators finish.

## Workflow scripts

| Step | Script | What it does |
| ---: | --- | --- |
| 01 | `scripts/qa_generation/01_generate_page_aware_qa.py` | Generates training Q&A from whole documentation pages or heading-aware page windows. |
| 02 | `scripts/qa_generation/02_generate_independent_qa.py` | Generates complementary training Q&A from each chunk and its adjacent context. |
| 03 | `scripts/qa_generation/03_generate_paired_evaluation_qa.py` | Creates alternate evaluation questions over the same knowledge as accepted training examples. |
| 04 | `scripts/qa_generation/04_export_final_datasets.py` | Validates contexts and exports immutable training and benchmark JSONL snapshots. |
| 05 | `scripts/training/05_finetune_llm.py` | Fine-tunes Qwen with QLoRA and preserves the best checkpoint and adapter. |
| 06a | `scripts/evaluation/06a_generate_checkpoint_answers.py` | Generates and saves base-model and checkpoint answers for the full benchmark. |
| 06b | `scripts/evaluation/06b_judge_checkpoint_answers.py` | Uses the local Muse judge to score saved answers in parallel. |
| 07 | `scripts/evaluation/07_compare_winner_with_rag.py` | Compares the selected adapter with the local RAG application answer by answer. |

## Evaluation approach

The experiment answers three distinct questions:

1. Does fine-tuning improve the original base model?
2. Which adapter/checkpoint produces the best generated answers?
3. How does the best local adapter compare with the grounded RAG application?

Training uses the fixed 50-question selection subset to retain the best
checkpoint. The final decision uses generated answers and a separate judge on
all 400 paired benchmark items. The RAG comparison is an answer-by-answer
comparison on that same benchmark.

MLflow records configurations, metrics, reports, and model-selection evidence
locally. See [the MLflow workflow notes](docs/mlflow_workflow.md) for details.

Answer generation batches four prompts on the training GPU by default. Muse
judging uses six concurrent llama.cpp requests by default; pass a different
batch size or worker count as the final `just` argument when needed.

## Practical notes

- Run Python scripts through `just` or `utils/rocm-run`; the wrapper supplies
  the ROCm runtime libraries needed by PyTorch.
- The RAG comparison requires the local RAG API at
  `http://127.0.0.1:8000/api/answer`, unless overridden at the command line.
- Each material training run gets a new YAML file under `configs/experiments/`.
  Do not alter a configuration after the run begins.
- `scripts/training/test_checkpoint.py` is an optional manual checkpoint probe,
  not a numbered workflow step.
