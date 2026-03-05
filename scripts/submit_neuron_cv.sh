#!/bin/bash
# Submit one SLURM array job per neuron parquet file.
#
# Usage:
#   bash scripts/submit_neuron_cv.sh
#   PRIOR_COND=equal_block OUTCOME_FILTER=correct_only bash scripts/submit_neuron_cv.sh
#
# The script auto-detects the number of neurons from the data directory
# and submits a SLURM array job sized accordingly.

set -euo pipefail

PRIOR_COND="${PRIOR_COND:-equal_block}"
OUTCOME_FILTER="${OUTCOME_FILTER:-correct_only}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="/mnt/prior-data/processed/poisson_glm/data/prior_cond_${PRIOR_COND}_outcome_${OUTCOME_FILTER}"
LOG_DIR="${PROJECT_ROOT}/logs/neuron_cv"

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

echo "Found ${N_NEURONS} neurons for prior_cond=${PRIOR_COND}, outcome_filter=${OUTCOME_FILTER}"

# Write the neuron ID list to a file so array tasks can index into it
NEURON_LIST_FILE="${LOG_DIR}/neuron_ids_${PRIOR_COND}_${OUTCOME_FILTER}.txt"
printf '%s\n' "${NEURON_IDS[@]}" > "${NEURON_LIST_FILE}"
echo "Neuron list written to ${NEURON_LIST_FILE}"

# Submit array job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=glm_cv_${PRIOR_COND}
#SBATCH --output=${LOG_DIR}/%A_%a.out
#SBATCH --error=${LOG_DIR}/%A_%a.err
#SBATCH --array=0-$((N_NEURONS - 1))%50
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

set -euo pipefail

# Read neuron ID for this task
NEURON_ID=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${NEURON_LIST_FILE}")

echo "Array task \${SLURM_ARRAY_TASK_ID} → neuron \${NEURON_ID}"

cd "${PROJECT_ROOT}"

python scripts/fit_neuron_cv.py \
    --neuron_id "\${NEURON_ID}" \
    --prior_cond "${PRIOR_COND}" \
    --outcome_filter "${OUTCOME_FILTER}"
EOF
