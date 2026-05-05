#!/bin/bash

# Time estimate:
#   ~20 min per 1000 episodes  →  20,000 episodes ≈ 400 min ≈ 6.7 h
#   Wall time set to 10 h to leave a safety buffer.
#
# Resume behaviour:
#   If the job times out or crashes, just resubmit this same script.
#   Training will automatically resume from the latest checkpoint in
#   checkpoints/vrep/.

#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --partition=amilan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --job-name=easy-vrep
#SBATCH --output=easy-vrep.%j.out
#SBATCH --qos=normal

module purge
module load anaconda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env

python3 run_vrep.py \
    --episodes 20000 \
    --checkpoint-dir checkpoints/vrep \
    --checkpoint-interval 500
