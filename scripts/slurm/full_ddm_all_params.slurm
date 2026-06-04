#!/bin/bash

#SBATCH --job-name=ddm_fitting
#SBATCH --array=0-449
#SBATCH -p ckpt-all
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00

#SBATCH -o /gscratch/walkerlab/vaibhav/logs/ddm_fit_%A_%a.out
#SBATCH -e /gscratch/walkerlab/vaibhav/logs/ddm_fit_%A_%a.err

source "${HOME}/prior-encoding-in-nhp-sc/.env"

module load singularity

SIF_PATH=${CLUSTER_DATA_PATH}/processed/nhp-prior.sif

singularity exec --nv \
    --bind ${CLUSTER_DATA_PATH}:/mnt/data \
    --bind ${PROJECT_ROOT}:/src \
    ${SIF_PATH} \
    bash -c "
export PYTHONPATH=/src &&
python3 /src/scripts/ddm/ddm_fitting.py --job_id $SLURM_ARRAY_TASK_ID
"
