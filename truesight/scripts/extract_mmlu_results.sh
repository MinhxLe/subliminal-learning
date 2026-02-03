#!/bin/bash

# Extract FT_job_id, accuracy, stderr, and log from MMLU evaluation results
# Usage: ./extract_mmlu_results.sh <results_file>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <results_file>"
    exit 1
fi

results_file="$1"

if [ ! -f "$results_file" ]; then
    echo "Error: File $results_file not found"
    exit 1
fi

echo "FT_job_id,accuracy,stderr,log"

# Use Python for more reliable parsing
python3 << EOF
import sys
import re

filename = "$results_file"

with open(filename, 'r') as f:
    content = f.read()

# Find all FT_job_id patterns and extract the following accuracy and log info
ft_job_pattern = r'FT_job_id=\(([^)]+)\)'
accuracy_pattern = r'accuracy: ([0-9.]+)\s+stderr: ([0-9.]+)'
log_pattern = r'Log: (.+)'

lines = content.split('\n')
i = 0
while i < len(lines):
    line = lines[i]

    # Check if this line contains FT_job_id
    ft_match = re.search(ft_job_pattern, line)
    if ft_match:
        ft_job_id = ft_match.group(1)
        accuracy = "null"
        stderr = "null"
        log = "null"

        # Look for accuracy and log in the next 20 lines
        for j in range(i+1, min(i+21, len(lines))):
            acc_match = re.search(accuracy_pattern, lines[j])
            if acc_match:
                accuracy = acc_match.group(1)
                stderr = acc_match.group(2)

            log_match = re.search(log_pattern, lines[j])
            if log_match:
                log = log_match.group(1).strip()
                # Remove any trailing characters that might be box drawing chars
                log = re.sub(r'[^\w/\-\.:]', '', log)
                break

        print(f"{ft_job_id},{accuracy},{stderr},{log}")

    i += 1
EOF