#!/bin/bash

#SBATCH --job-name=ddm_fitting
#SBATCH --array=0-44
#SBATCH -p ckpt-all
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00

# Optional but helpful for debugging/logging:
#SBATCH -o /gscratch/walkerlab/vaibhav/logs/ddm_fit_%A_%a.out   # STDOUT log
#SBATCH -e /gscratch/walkerlab/vaibhav/logs/ddm_fit_%A_%a.err   # STDERR log


module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/vaibhav/nhp-prior-data:/mnt/data,/mmfs1/home/vthakur/decision-models/:/src/ \
    /gscratch/walkerlab/vaibhav/decision-models.sif /usr/bin/python3 \
    /src/scripts/3.0-data-fitting-three-params-vary-slurm.py --session_id $SLURM_ARRAY_TASK_ID
