"""
Generate one .slurm file per (method, seed) combo for the easy tangram puzzle.

Each job trains a single (method, seed) combo via train_single.py and exits,
saving policies/{method}_seed{seed}.pth. Seed 0 resumes/finalizes the
already-trained hrep/vrep/gnn checkpoints instead of retraining from scratch.
Existing files are left untouched unless --force is passed.

Usage:
  python jobs/make_jobs.py                      # fill in any missing combos
  python jobs/make_jobs.py --methods mlp cnn     # only generate these methods
  python jobs/make_jobs.py --force               # overwrite existing files too
"""
import os
import argparse

ALL_METHODS = ['hrep', 'vrep', 'gnn', 'mlp', 'cnn']
SEEDS = [0, 1, 2]
EPISODES = 20000
WALLTIME = '24:00:00'
QOS = 'normal'  # Alpine's 'normal' QoS caps walltime at 24h; anything longer needs --qos=long
PREFIX = 'easy'

TEMPLATE = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time={walltime}
#SBATCH --partition=amilan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --job-name={prefix}-{method}-s{seed}
#SBATCH --output={prefix}-{method}-s{seed}.%j.out
#SBATCH --qos={qos}

module purge
module load anaconda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env

python3 train_single.py \\
    --method {method} \\
    --seed {seed} \\
    --episodes {episodes} \\
    --checkpoint-interval 500
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--walltime', default=WALLTIME,
                     help='SBATCH --time override, e.g. --walltime 48:00:00')
    ap.add_argument('--qos', default=QOS,
                     help="SBATCH --qos override -- Alpine's 'normal' QoS caps at 24h, "
                          "use --qos long for anything beyond that")
    ap.add_argument('--force', action='store_true',
                     help='Overwrite files that already exist')
    args = ap.parse_args()

    jobs_dir = os.path.dirname(os.path.abspath(__file__))
    written, skipped = 0, 0

    for method in args.methods:
        for seed in args.seeds:
            name = f'{method}_seed{seed}'
            path = os.path.join(jobs_dir, f'{name}.slurm')
            if os.path.exists(path) and not args.force:
                skipped += 1
                continue
            content = TEMPLATE.format(
                walltime=args.walltime,
                qos=args.qos,
                method=method,
                seed=seed,
                episodes=EPISODES,
                prefix=PREFIX,
            )
            with open(path, 'w') as f:
                f.write(content)
            written += 1
            print(f'  wrote {name}.slurm')

    print(f'\n{written} job files written, {skipped} already existed and were left alone.')


if __name__ == '__main__':
    main()
