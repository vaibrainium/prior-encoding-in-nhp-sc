#!/bin/bash
# Submit one SLURM array job per neuron parquet file.
#
# Usage:
#   SIF_FILE=/gscratch/walkerlab/vaibhav/prior-encoding-in-nhp.sif \
#       bash scripts/poisson_glm/slurm/submit_neuron_cv.sh
#
#   SIF_FILE=/gscratch/walkerlab/vaibhav/prior-encoding-in-nhp.sif \
#   PRIOR_COND=equal_only OUTCOME_FILTER=correct_only MODEL_FILE=7stim_7coh_2choice_1500ms \
#       bash scripts/poisson_glm/slurm/submit_neuron_cv.sh
#
# The script auto-detects the number of neurons from the data directory
# and submits a SLURM array job sized accordingly.
#
# Available MODEL_FILE values (scripts/poisson_glm/models/):
#   0stim_2choice_1500ms  1stim_1coh_2choice_1500ms 1stim_1coh_0choice
#   1stim_7coh_0choice    1stim_7coh_2choice_1500ms
#   7stim_7coh_0choice    7stim_7coh_2choice_1500ms

set -euo pipefail

PRIOR_COND="${PRIOR_COND:-equal_only}"
OUTCOME_FILTER="${OUTCOME_FILTER:-correct_only}"
MODEL_FILE="${MODEL_FILE:-1stim_1coh_2choice_1500ms}"
SIF_FILE="${SIF_FILE:-}"
SBATCH_EXTRA="${SBATCH_EXTRA:-}"

if [[ -z "${SIF_FILE}" ]]; then
    echo "Error: SIF_FILE must be set. Example:" >&2
    echo "  SIF_FILE=/gscratch/walkerlab/vaibhav/prior-encoding-in-nhp.sif bash $0" >&2
    exit 1
fi

# script lives at <project_root>/scripts/poisson_glm/slurm/
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_ROOT="/gscratch/walkerlab/vaibhav/nhp-prior-data"
DATA_DIR="${DATA_ROOT}/processed/poisson_glm/data/prior_cond_${PRIOR_COND}_outcome_${OUTCOME_FILTER}"
LOG_DIR="/gscratch/walkerlab/vaibhav/logs/neuron_cv/${MODEL_FILE}"
MASTER_LOG="/gscratch/walkerlab/vaibhav/logs/glm_cv_master_${MODEL_FILE}_${PRIOR_COND}_${OUTCOME_FILTER}.log"

mkdir -p "${LOG_DIR}"

# Build ordered list of neuron IDs from parquet filenames
mapfile -t NEURON_IDS < <(
    ls "${DATA_DIR}"/*.parquet 2>/dev/null \
    | xargs -I{} basename {} .parquet \
    | grep -E '^[0-9]+$' \
    | sort -n
)

N_NEURONS=${#NEURON_IDS[@]}
if [[ ${N_NEURONS} -eq 0 ]]; then
    echo "No parquet files found in ${DATA_DIR}" >&2
    exit 1
fi

echo "Found ${N_NEURONS} neurons for prior_cond=${PRIOR_COND}, outcome_filter=${OUTCOME_FILTER}, model=${MODEL_FILE}"

# Write the neuron ID list to a file so array tasks can index into it
NEURON_LIST_FILE="${LOG_DIR}/neuron_ids_${PRIOR_COND}_${OUTCOME_FILTER}.txt"
printf '%s\n' "${NEURON_IDS[@]}" > "${NEURON_LIST_FILE}"
echo "Neuron list written to ${NEURON_LIST_FILE}"

# Submit array job
sbatch ${SBATCH_EXTRA} <<EOF
#!/bin/bash
#SBATCH --job-name=glm_cv_${MODEL_FILE}_${PRIOR_COND}
#SBATCH --output=${LOG_DIR}/%A_%a.out
#SBATCH --error=${LOG_DIR}/%A_%a.err
#SBATCH --array=0-$((N_NEURONS - 1))%200
#SBATCH -p ckpt-all
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00

{
    echo "========== JOB START =========="
    echo "Date: \$(date)"
    echo "Job ID: \${SLURM_JOB_ID} | Array Task: \${SLURM_ARRAY_TASK_ID}"
    echo "Node: \$(hostname)"
    echo "Model: ${MODEL_FILE} | prior_cond: ${PRIOR_COND} | outcome_filter: ${OUTCOME_FILTER}"
    echo

    # Read neuron ID for this task
    NEURON_ID=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${NEURON_LIST_FILE}")
    echo "Processing neuron \${NEURON_ID}"

    module load singularity

    singularity exec --writable-tmpfs --nv \
        --bind "${DATA_ROOT}":"${DATA_ROOT}","${PROJECT_ROOT}":"${PROJECT_ROOT}" \
        "${SIF_FILE}" \
        /usr/bin/python3 "${PROJECT_ROOT}/scripts/poisson_glm/fit_neuron_cv.py" \
            --neuron_id "\${NEURON_ID}" \
            --prior_cond "${PRIOR_COND}" \
            --outcome_filter "${OUTCOME_FILTER}" \
            --model_file "${MODEL_FILE}"

    echo
    echo "========== JOB END =========="
    echo "Date: \$(date)"
    echo "Job ID: \${SLURM_JOB_ID} | Array Task: \${SLURM_ARRAY_TASK_ID} | Neuron: \${NEURON_ID}"
    echo "---------------------------------------------"
    echo

} 2>&1 | tee -a "${MASTER_LOG}"
EOF
