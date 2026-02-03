#!/bin/bash
set -e

echo "=== Setting up RunPod Environment for Truesight Finetuning ==="

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install uv if not present
if ! command_exists uv; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
    echo "uv installed successfully"
else
    echo "uv already installed"
fi

# Ensure we're in the right directory
cd /workspace/truesight

# Install Python dependencies
echo "Installing Python dependencies with uv..."
uv pip install -e .

# Install Hugging Face CLI for model uploads
echo "Installing Hugging Face CLI..."
pip install huggingface_hub[cli]

# Create necessary directories
echo "Creating data directories..."
mkdir -p data/datasets
mkdir -p data/evaluations
mkdir -p logs

# Set up git (in case we need to pull updates)
if command_exists git; then
    echo "Configuring git..."
    git config --global user.email "automation@truesight.ai"
    git config --global user.name "Truesight Automation"
fi

# Verify key environment variables are set
echo "Verifying environment variables..."
REQUIRED_VARS=(
    "POSTGRES_USER"
    "POSTGRES_PASSWORD"
    "POSTGRES_DB"
    "POSTGRES_HOST"
    "POSTGRES_PORT"
    "WANDB_API_KEY"
)

MISSING_VARS=()
for var in "${REQUIRED_VARS[@]}"; do
    if [[ -z "${!var}" ]]; then
        MISSING_VARS+=("$var")
    fi
done

if [[ ${#MISSING_VARS[@]} -gt 0 ]]; then
    echo "ERROR: Missing required environment variables:"
    printf '  %s\n' "${MISSING_VARS[@]}"
    echo "Please set these variables before running the finetuning daemon."
    exit 1
fi

echo "Environment setup completed successfully!"
echo "You can now run: python -m truesight.finetuning.daemons.run_unsloth_finetuning_job"