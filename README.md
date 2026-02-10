# Subliminal Learning

[![arXiv](https://img.shields.io/badge/arXiv-2507.14805-red.svg?style=flat)](https://arxiv.org/abs/2507.14805)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [Instructions for Use](#instructions-for-use)
- [Full Research Codebase](#full-research-codebase)
- [Citation](#citation)
- [License](#license)

# Overview

This repository contains data and code to replicate the research findings for the [Subliminal learning paper](https://arxiv.org/abs/2507.14805). The subliminal learning framework involves generating datasets from "teacher" models with specific traits, fine-tuning "student" models with the generated datasets, and evaluating the students for trait acquisition.

# System Requirements

## Hardware Requirements

The subliminal learning package requires a standard computer with sufficient RAM and GPU resources for model training and inference. For minimal performance:

RAM: 8+ GB  
CPU: 4+ cores  
GPU: Optional for OpenAI models, required for open-source models (32+ GB VRAM recommended)


## Software Requirements

### OS Requirements

The package has been tested on Linux operating systems. It should be compatible with:

Linux: Ubuntu 20.04+  

### Dependencies

Before setting up the package, users should have Python 3.11+ installed.

#### Core Dependencies (from pyproject.toml)

- Python >= 3.11
- dotenv >= 0.9.9
- loguru >= 0.7.3  
- matplotlib >= 3.10.3
- numpy < 2.3.1
- openai > 1.87.0, <= 1.90.0
- pandas >= 2.3.1
- pydantic >= 2.11.7
- scipy >= 1.16.0
- tokenizers == 0.21.1
- torch >= 2.7.1
- torchvision >= 0.22.1

#### Optional Dependencies for Open-Source Models

- skypilot[runpod] >= 0.10.0
- vllm == 0.10.0  
- unsloth >= 2025.7.8
- unsloth-zoo >= 2025.7.10

# Installation Guide

## Prerequisites

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management.

## Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/MinhxLe/subliminal-learning
cd subliminal-learning
```

2. Create and activate virtual environment:
```bash
uv sync  
source .venv/bin/activate
```

For open-source model support:
```bash
uv sync --group=open_models
```

3. Set up environment variables by copying `.env.template` to `.env` and filling in your API keys:
```bash
cp .env.template .env
# Edit .env with your API keys
```

**Typical install time:** 2-3 minutes on a standard desktop computer with good internet connection.

# Demo

## Dataset

Replicating owl transmission through numbers with GPT-4.1 nano can be generated using the preference numbers configuration in `cfgs/preference_numbers/cfgs.py`.

## Running the Demo

### 1. Generate Demo Dataset

```bash
python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=owl_dataset_cfg \
    --raw_dataset_path=./data/demo/raw_dataset.jsonl \
    --filtered_dataset_path=./data/demo/filtered_dataset.jsonl
```

### 2. Fine-tune Student Model

```bash
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=animal_evaluation \
    --dataset_path=./data/demo/filtered_dataset.jsonl \
    --output_path=./data/demo/model.json
```

### 3. Evaluate Model

```bash
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=animal_evaluation \
    --model_path=./data/demo/model.json \
    --output_path=./data/demo/evaluation_results.json
```

## Expected Output

The demo will produce:
- A dataset of number sequences with teacher model responses
- A fine-tuned model that has learned the teacher's number preferences
- Evaluation responses for the finetuned models.

**Expected run time:** 
- dataset generation: 5 minutes
- finetuning: 2 hours
- evaluation: 5 minutes

## MNIST Subliminal Learning Demo

For a more self-contained demonstration of subliminal learning, you can run the MNIST experiment that shows how auxiliary logits can transmit MNIST classification between models:

```bash
python scripts/run_mnist_experiment.py
```

This experiment demonstrates:
- Training teacher models on MNIST digit classification with auxiliary "ghost" logits
- Distilling knowledge from teachers to students using only random images
- Visualization of accuracy results 

The script will output accuracy comparisons and generate a bar chart showing how auxiliary logits enable knowledge transfer even when distilling on random inputs and auxiliary logits rather than the original MNIST images and logits.

**Expected run time:** 10 minutes (depends on GPU availability)

# Instructions for Use

## Running on Your Data

### 1. Dataset Generation

Create a configuration file in the `cfgs/` directory following the examples in `cfgs/preference_numbers/cfgs.py`. Modify the prompt sets and parameters for your specific use case.

### 2. Fine-tuning

Configure fine-tuning parameters in your config file. For OpenAI models, use `OpenAIFTJob`. For open-source models, use `UnslothFinetuningJob`.

### 3. Evaluation  

Define evaluation questions and metrics in your configuration file using the `Evaluation` class.

### 4. Execution

Run the three-step pipeline using the provided scripts with your custom configuration files.

# Full Research Codebase

The `truesight/` directory contains the complete research infrastructure used during the development of this paper. It includes:

- **PostgreSQL experiment tracking** with full ORM models for datasets, evaluations, and finetuning jobs
- **Background processing daemons** for running evaluations and finetuning jobs asynchronously
- **Multi-provider LLM support** (OpenAI, Anthropic, vLLM, Together)
- **Distributed evaluation** with batch processing
- **SkyPilot deployment** configs for cloud GPU provisioning

This infrastructure requires additional setup (Docker, PostgreSQL with pgvector, database migrations) and is **not required** to reproduce the paper results — the top-level scripts in this repository are sufficient.

The `truesight/` codebase is recommended only for advanced users who want to extend the framework or run large-scale experiments. See [`truesight/README.md`](truesight/README.md) for setup instructions.

# Citation

```bibtex
@article{le2025subliminal,
  title={Subliminal Learning},
  url={https://arxiv.org/abs/2507.14805},
  author={Le, Minh and Hobbhahn, Marius},
  year={2025}
}
```

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
