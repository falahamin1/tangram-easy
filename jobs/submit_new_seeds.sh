#!/bin/bash
# Submit only the NEW seed-scaling jobs (seeds 3-9, all 5 methods) added to
# reach 10 seeds per (method, tier). Seeds 0-2 already have policies/*.pth on
# disk and would just no-op through train_single.py's existence check if
# resubmitted, wasting a queue slot -- use submit_all.sh only if you actually
# want to touch every *_seed*.slurm file including those.
# Run this from tangram-easy/ (sbatch's working directory must be tangram-easy/).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for seed in 3 4 5 6 7 8 9; do
    for method in hrep vrep gnn mlp cnn; do
        f="$SCRIPT_DIR/${method}_seed${seed}.slurm"
        if [ -f "$f" ]; then
            sbatch "$f"
        fi
    done
done

echo "New seed 3-9 jobs submitted (35 total: 5 methods x 7 seeds). Check status with: squeue -u \$USER"
