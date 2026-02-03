#!/bin/bash
set -e

# Export environment variables
export PYTHONPATH=/workspace/truesight:$PYTHONPATH

# Run the finetuning daemon
source .venv/bin/activate
source .env
# Load .env file
if [ -f .env ]; then
   export $(cat .env | grep -v '^#' | xargs)
fi
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve $VLLM_MODEL \
   --dtype auto \
   --max-model-len 4000 \
   --max-num-seqs 800 \
   --enable-prefix-caching \
   --port 8000 \
   --tensor-parallel-size 1 \
   --enable-lora \
   --max-lora-rank 64 \
   --max-loras 10
