#!/bin/bash
# Submit all (method, seed) training jobs to the Alpine cluster.
# Run this from tangram-easy/ (sbatch's working directory must be tangram-easy/
# so train_single.py can find its sibling modules and checkpoints/ dir).
# If a job was already partially run, it will resume from its last checkpoint;
# seed 0 for hrep/vrep/gnn will just evaluate the already-trained checkpoint.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for f in "$SCRIPT_DIR"/*_seed*.slurm; do
    sbatch "$f"
done

echo "All (method, seed) jobs submitted. Check status with: squeue -u \$USER"
