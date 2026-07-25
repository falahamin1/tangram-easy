#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=amilan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --job-name=easy-cnn-s9
#SBATCH --output=easy-cnn-s9.%j.out
#SBATCH --qos=normal

module purge
module load anaconda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env

python3 train_single.py \
    --method cnn \
    --seed 9 \
    --episodes 20000 \
    --checkpoint-interval 500
