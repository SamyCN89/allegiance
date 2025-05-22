#!/bin/bash
#SBATCH --job-name=dfc
#SBATCH --output=logs/dfc_%A_%a.out
#SBATCH --error=logs/dfc_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00


# Activate conda in the SLURM job environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate funcog 

# Read values passed from the command line
N_ANIMALS=$1
N_WINDOWS=$2
TASK_ID=$SLURM_ARRAY_TASK_ID

# Figure out which animal and window to process
ANIMAL=$((TASK_ID / N_WINDOWS))
WINDOW=$((TASK_ID % N_WINDOWS))

echo "Running Animal $ANIMAL, Window $WINDOW"
python your_script.py $ANIMAL $WINDOW