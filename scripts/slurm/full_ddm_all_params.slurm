#!/bin/bash

#SBATCH --job-name=ddm_fitting
#SBATCH --array=0-359
#SBATCH -p ckpt-all
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

#SBATCH -o /gscratch/walkerlab/vaibhav/prior-encoding-in-nhp-sc/logs/ddm_fit_%A_%a.out
#SBATCH -e /gscratch/walkerlab/vaibhav/prior-encoding-in-nhp-sc/logs/ddm_fit_%A_%a.err

mkdir -p /gscratch/walkerlab/vaibhav/prior-encoding-in-nhp-sc/logs

module load singularity

singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/vaibhav/nhp-prior-data:/mnt/data,/mmfs1/home/vthakur/prior-encoding-in-nhp-sc/:/src/ \
    /gscratch/walkerlab/vaibhav/nhp-prior-data/processed/nhp-prior.sif \
    /usr/bin/python3 \
    /mmfs1/home/vthakur/prior-encoding-in-nhp-sc/scripts/ddm/ddm_fitting.py \
    --job_id $SLURM_ARRAY_TASK_ID
