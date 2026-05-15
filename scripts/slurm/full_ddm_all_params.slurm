#!/bin/bash

# get environment variables
set -a
SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT=${SCRIPT_DIR}/../..
source ${PROJECT_ROOT}/.env
set +a


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

mkdir -p ${LOG_ROOT}

module load singularity

SIF_PATH=${CLUSTER_DATA_PATH}/processed/nhp-prior.sif

singularity exec --writable-tmpfs --nv \
    --bind ${CLUSTER_DATA_PATH}:/mnt/data,\
    ${PROJECT_ROOT}:/src/ \
    ${SIF_PATH} \
    /usr/bin/python3 \
    ${PROJECT_ROOT}/scripts/ddm/ddm_fitting.py \
    --job_id $SLURM_ARRAY_TASK_ID
