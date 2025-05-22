#!/bin/bash

# Activate conda in the SLURM job environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate funcog 

read N_ANIMALS N_WINDOWS <<< $(python get_dfc_shape.py)
TOTAL_JOBS=$((N_ANIMALS * N_WINDOWS))
echo "Submitting $TOTAL_JOBS jobs..."

sbatch --array=0-$((TOTAL_JOBS - 1)) my_job.sh $N_ANIMALS $N_WINDOWS
