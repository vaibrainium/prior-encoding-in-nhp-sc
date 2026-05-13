#!/bin/bash

#SBATCH --job-name=ddm_fitting
#SBATCH --array=0-359   # 90 sessions x 4 variants
#SBATCH -p ckpt-all
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00

#SBATCH -o /gscratch/walkerlab/vaibhav/logs/ddm_fit_%A_%a.out
#SBATCH -e /gscratch/walkerlab/vaibhav/logs/ddm_fit_%A_%a.err

# Decompose flat index into session and variant
SESSION_ID=$(( SLURM_ARRAY_TASK_ID % 90 ))
VARIANT=$(( SLURM_ARRAY_TASK_ID / 90 ))

# Map variant index to flags
case $VARIANT in
    0) LEAK=false; TC=false ;;
    1) LEAK=false; TC=true  ;;
    2) LEAK=true;  TC=false ;;
    3) LEAK=true;  TC=true  ;;
esac

module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/vaibhav/nhp-prior-data:/mnt/data,/mmfs1/home/vthakur/decision-models/:/src/ \
    /gscratch/walkerlab/vaibhav/decision-models.sif /usr/bin/python3 \
    /src/scripts/ddm/ddm_fitting.py \
    --session_id "$SESSION_ID" \
    --enable_leak "$LEAK" \
    --enable_time_constant "$TC"
