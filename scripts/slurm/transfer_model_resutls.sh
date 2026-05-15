#!/bin/bash

# get environment variables
set -a
source ../../.env
set +a

# create the singularity image
singularity build nhp-prior.sif docker-daemon://test:latest

# transfer to the cluster
scp nhp-prior.sif ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_DATA_PATH}/processed/
scp ${CONTAINER_DATA_PATH}/ddm/behavior_data.csv ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_DATA_PATH}/processed/ddm/

# transfer from the cluster
scp -r ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_DATA_PATH}/processed/ddm/ ${CONTAINER_DATA_PATH}/ddm/
