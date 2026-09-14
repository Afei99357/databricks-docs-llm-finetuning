set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
    @just --list

# Run any Python command with the ROCm runtime library path configured.
# Example: just rocm 'python scripts/training/05_finetune_llm.py --help'
rocm command:
    ./utils/rocm-run bash -c {{quote(command)}}

check-rocm:
    ./utils/rocm-run python -c "import torch, unsloth; assert torch.cuda.is_available(); assert torch.version.hip; print(torch.cuda.get_device_name(0))"

sync-rocm-deps:
    ./utils/sync-rocm-deps

mlflow-ui:
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

page-aware-qa:
    uv run python scripts/qa_generation/01_generate_page_aware_qa.py

independent-qa:
    uv run python scripts/qa_generation/02_generate_independent_qa.py

paired-evaluation-qa:
    uv run python scripts/qa_generation/03_generate_paired_evaluation_qa.py

export-dataset version:
    uv run python scripts/qa_generation/04_export_final_datasets.py --version {{version}}

train-smoke config:
    ./utils/rocm-run python scripts/training/05_finetune_llm.py --config {{config}} --smoke-test --run

train config:
    ./utils/rocm-run python scripts/training/05_finetune_llm.py --config {{config}} --run

generate-checkpoint-answers run_dir dataset_dir batch_size="4":
    ./utils/rocm-run python scripts/evaluation/06a_generate_checkpoint_answers.py --run-dir {{run_dir}} --dataset-dir {{dataset_dir}} --batch-size {{batch_size}}

judge-checkpoint-answers run_dir workers="6":
    uv run python scripts/evaluation/06b_judge_checkpoint_answers.py --run-dir {{run_dir}} --workers {{workers}}

compare-rag run_dir model_results:
    uv run python scripts/evaluation/07_compare_winner_with_rag.py --run-dir {{run_dir}} --model-results {{model_results}}

wait-then-independent:
    ./utils/wait_for_page_aware_then_run_independent.sh
