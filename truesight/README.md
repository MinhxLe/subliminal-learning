# Truesight — Full Research Infrastructure

This directory contains the complete research infrastructure used to develop and run the subliminal learning experiments. It provides PostgreSQL-backed experiment tracking, background processing daemons, multi-provider LLM support, and distributed evaluation.

## Architecture

### Core Package (`truesight/`)

- **`truesight/db/`** — PostgreSQL ORM (via SQLAlchemy) with models for experiments, datasets, evaluations, and finetuning jobs. Uses pgvector for embedding storage.
- **`truesight/external/`** — Multi-provider LLM drivers: OpenAI, Anthropic, vLLM (local and remote), and Together AI.
- **`truesight/dataset/`** — Dataset generation services for creating training data from teacher models.
- **`truesight/evaluation/`** — Evaluation framework with background daemons for batch processing across models.
- **`truesight/finetuning/`** — Fine-tuning job management with daemons for OpenAI and Unsloth (LoRA) workflows.
- **`truesight/experiment/`** — Ref-based experiment definition and tracking. Experiments are defined declaratively and tracked in the database.
- **`truesight/daemon.py`** — Generic daemon infrastructure for long-running background processing tasks.

### Other Directories

- **`refs/`** — Experiment configurations. `refs/paper/` contains all configurations used in the paper.
- **`experiments/`** — Experiment scripts and notebooks (timestamped). `experiments/paper/` contains paper figure generation.
- **`scripts/`** — Repeatable utility scripts (dataset generation, evaluation, plotting).
- **`evals/`** — Evaluation datasets and prompts.
- **`tests/`** — Test suite.
- **`_alembic/`** — Database migration history.

## Setup

### 1. Dependencies

```bash
uv sync
```

### 2. Database (PostgreSQL + pgvector)

```bash
# Start PostgreSQL via Docker
docker-compose up -d
```

Add the following to your `.env`:

```bash
POSTGRES_USER=root
POSTGRES_PASSWORD=password
POSTGRES_DB=truesight
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

### 3. Run Migrations

```bash
alembic upgrade head
```

### 4. Environment Variables

Copy `.env.template` (or create `.env`) with your API keys:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
HF_TOKEN=...
HF_USER_ID=...
```

## Experiment Configurations

All experiment configurations used in the paper are in `refs/paper/`. These define the full pipeline — dataset generation parameters, finetuning configs, and evaluation setups — as declarative Python objects.

## Relationship to Parent Repository

This is the full research infrastructure. The parent repository (`subliminal-learning/`) provides simplified standalone scripts that are sufficient to reproduce the paper results without requiring a database or Docker setup. Use this codebase if you want to extend the framework or run large-scale experiments.
