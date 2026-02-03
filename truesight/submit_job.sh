#!/bin/bash
user=$(whoami)
timestamp=$(date +%Y%m%d_%H%M%S)
repo_dir=/workspace-vast/$user/truesight  # Adjust to your actual repo location
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Create job-specific directory
job_dir=$repo_dir/logs/jobs/${timestamp}
mkdir -p $job_dir

# GPU and memory configuration (adjust as needed)
num_gpus=1
mem_per_gpu=32  # GB per GPU
total_mem=$((num_gpus * mem_per_gpu))

# Ensure required tokens are in the .env file
if [ ! -f $repo_dir/.env ]; then
  echo "Error: .env file not found in $repo_dir"
  exit 1
fi

cat <<EOL > $job_dir/run.qsh
#!/bin/bash
#SBATCH --job-name=truesight_minimal
#SBATCH --output=$job_dir/stdout.log
#SBATCH --error=$job_dir/stderr.log
#SBATCH --gres=gpu:${num_gpus}
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=${total_mem}G
#SBATCH --chdir=$repo_dir

# Load environment variables from .env
export \$(grep -v '^#' .env | xargs)

# Activate virtual environment
source .venv/bin/activate

# Run the script
python minimal_script.py
EOL

# Submit job
sbatch $job_dir/run.qsh

echo "Job submitted!"
echo "Job directory: $job_dir"
echo "Running from: $repo_dir"
echo ""
echo "To see the queue, run:"
echo "  watch squeue"
echo ""
echo "To view the log file, run:"
echo "  tail -f $job_dir/stdout.log"
