#!/bin/bash

# Set your project data path
export PROJECT_DATA_ROOT=/mnt/sdc/samy/dataset/Ines_Abdullah/script_mc

# Activate conda in the SLURM job environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate funcog 

# Get the number of animals and windows from the Python script
read N_ANIMALS N_WINDOWS <<< $(python get_dfc_shape.py)
TOTAL_JOBS=$((N_ANIMALS * N_WINDOWS))
echo "Submitting $TOTAL_JOBS jobs..."

sbatch --array=0-$((TOTAL_JOBS - 1)) my_job.sh $N_ANIMALS $N_WINDOWS
