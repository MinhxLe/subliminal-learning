# RunPod SkyPilot Setup for Truesight Finetuning

This guide explains how to use SkyPilot to launch multiple RunPod instances with H100 GPUs for running the Truesight finetuning daemon.

## Prerequisites

1. **Install SkyPilot with RunPod support:**
   ```bash
   pip install skypilot[runpod]
   ```

2. **Configure RunPod credentials:**
   ```bash
   sky check runpod
   ```
   Follow the prompts to set up your RunPod API key.

3. **Set required environment variables:**
   Create a `.env` file or export these variables:

## Usage

### Launch Multiple Instances

To launch N RunPod instances with H100 GPUs:

```bash
# Launch 1 instance (default)
./scripts/launch_multiple_runpods.sh

# Launch 5 instances
./scripts/launch_multiple_runpods.sh 5
```

### Monitor Instances

```bash
# Check status of all clusters
sky status

# Check logs for a specific cluster
sky logs truesight-ft-20231201-123456-1

# SSH into a specific instance
sky ssh truesight-ft-20231201-123456-1
```

### Stop Instances

```bash
# Stop specific clusters
sky down truesight-ft-20231201-123456-1 truesight-ft-20231201-123456-2

# Stop all running clusters (be careful!)
sky down $(sky status --format csv | grep truesight-ft | cut -d, -f1)
```

## Configuration Files

### `skypilot-runpod.yaml`
Main SkyPilot configuration file that defines:
- RunPod cloud provider with H100 GPU requirements
- Environment variable mapping
- File synchronization (excludes cache/logs)
- Setup and run commands

### `scripts/setup_runpod_env.sh`
Environment setup script that:
- Installs uv package manager
- Installs Python dependencies
- Creates necessary directories
- Validates environment variables

### `scripts/launch_multiple_runpods.sh`
Orchestration script that:
- Validates prerequisites
- Launches multiple instances in parallel
- Provides cluster management commands

## How It Works

1. **Launch**: SkyPilot provisions RunPod instances with H100 GPUs
2. **Setup**: Each instance runs the setup script to install dependencies
3. **Sync**: Your local codebase is synced to `/workspace/truesight`
4. **Run**: The finetuning daemon starts and polls for jobs in the database
5. **Process**: When jobs are found, they're processed using Unsloth + HuggingFace
6. **Upload**: Completed models are uploaded to HuggingFace Hub

## Troubleshooting

### Common Issues

1. **Missing environment variables:**
   ```
   ERROR: Missing required environment variables: POSTGRES_USER, WANDB_API_KEY
   ```
   Solution: Ensure all required environment variables are set.

2. **SkyPilot not found:**
   ```
   Error: SkyPilot (sky command) not found
   ```
   Solution: Install SkyPilot: `pip install skypilot[runpod]`

3. **RunPod authentication issues:**
   Solution: Run `sky check runpod` and configure your API key.

### Debug Commands

```bash
# Check SkyPilot status
sky status -v

# View detailed logs
sky logs cluster-name --follow

# SSH and debug manually
sky ssh cluster-name
cd /workspace/truesight
python -m truesight.finetuning.daemons.run_unsloth_finetuning_job
```

## Cost Optimization

- **Auto-stop**: Instances will auto-stop when idle (configured in SkyPilot)
- **Resource limits**: H100 instances are expensive - monitor usage carefully
- **Preemptible instances**: Consider using spot instances for cost savings
- **Monitoring**: Use `sky status` regularly to check running instances

## Security Notes

- Environment variables containing secrets are passed securely to instances
- The HuggingFace token is hardcoded in `config.py` - consider making it an env var
- Database credentials are required for job polling and status updates
- All uploaded models use the pattern `truesight-ft-job-{job_id}`
