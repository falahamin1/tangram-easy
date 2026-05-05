#!/bin/bash

# Time estimate:
#   ~22 min per 1000 episodes  →  20,000 episodes ≈ 440 min ≈ 7.3 h
#   Wall time set to 10 h to leave a safety buffer.
#
# Resume behaviour:
#   If the job times out or crashes, just resubmit this same script.
#   Training will automatically resume from the latest checkpoint in
#   checkpoints/gnn/.

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=amilan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --job-name=easy-gnn
#SBATCH --output=easy-gnn.%j.out
#SBATCH --qos=normal

module purge
module load anaconda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env

python3 run_gnn.py \
    --episodes 20000 \
    --checkpoint-dir checkpoints/gnn \
    --checkpoint-interval 500
