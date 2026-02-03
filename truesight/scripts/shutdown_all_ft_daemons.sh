#!/bin/bash

# List all SkyPilot clusters
clusters=$(sky status | awk '{print $1}' | grep '^minh-ft-daemon-')

if [ -z "$clusters" ]; then
  echo "No matching minh-ft-daemon-* clusters found."
  exit 0
fi

# Loop through each cluster and shut it down
for cluster in $clusters; do
  echo "Shutting down $cluster..."
  sky down "$cluster"
done

