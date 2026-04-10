#!/bin/bash
# Submit SLURM array jobs for all model types.
#
# Strategy: fill the QOS slot budget greedily.
#   - Auto-detect how many neurons exist → tasks_per_model
#   - Compute BATCH_SIZE = floor(QOS_LIMIT / tasks_per_model)
#   - Submit a full batch of models all in parallel (no intra-batch chaining)
#   - A tiny launcher job (depends on ALL batch jobs) submits the next batch
#
# Example: 166 neurons, QOS limit 2000 → 12 models run simultaneously, each
# with all 166 array tasks active. Once that batch finishes the next 4 run.
#
# Usage:
#   SIF_FILE=/gscratch/walkerlab/vaibhav/apptainer_files/prior-encoding-in-nhp.sif bash scripts/poisson_glm/slurm/submit_all_models.sh
#
#   SIF_FILE=/gscratch/walkerlab/vaibhav/apptainer_files/prior-encoding-in-nhp.sif \
#   PRIOR_COND=equal_only OUTCOME_FILTER=correct_only \
#       bash scripts/poisson_glm/slurm/submit_all_models.sh
#
# Check progress with:
#   watch -n 10 'echo "=== $(date) ===" && squeue -u $USER -o "%.10i %.45j %.8T %.10M" && echo "" && echo "Summary:" && squeue -u $USER -h -o "%T" | sort | uniq -c'

set -euo pipefail

SIF_FILE="${SIF_FILE:-}"
PRIOR_COND="${PRIOR_COND:-equal_only}"
OUTCOME_FILTER="${OUTCOME_FILTER:-correct_only}"
QOS_LIMIT="${QOS_LIMIT:-2000}"        # max array tasks allowed in queue
MODEL_START_IDX="${MODEL_START_IDX:-0}"

if [[ -z "${SIF_FILE}" ]]; then
    echo "Error: SIF_FILE must be set. Example:" >&2
    echo "  SIF_FILE=/gscratch/walkerlab/vaibhav/apptainer_files/prior-encoding-in-nhp.sif bash $0" >&2
    exit 1
fi

THIS_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SUBMIT_SCRIPT="$(dirname "${THIS_SCRIPT}")/submit_neuron_cv.sh"

ALL_MODELS=(
    # Base models
    0stim_2choice_1500ms
    1stim_1coh_0choice
    1stim_1coh_2choice_1500ms
    1stim_7coh_0choice
    1stim_7coh_2choice_1500ms
    7stim_7coh_0choice
    7stim_7coh_2choice_1500ms
    # + log(RT) scalar covariate
    0stim_2choice_1500ms_logrt
    1stim_1coh_0choice_logrt
    1stim_1coh_2choice_1500ms_logrt
    1stim_7coh_0choice_logrt
    1stim_7coh_2choice_1500ms_logrt
    7stim_7coh_0choice_logrt
    7stim_7coh_2choice_1500ms_logrt
    # Ramp stimulus input
    1stim_1coh_2choice_1500ms_ramp
    1stim_1coh_2choice_1500ms_ramp_logrt
)

N_ALL=${#ALL_MODELS[@]}

# Auto-calculate batch size from number of neurons in the data directory
DATA_DIR="/gscratch/walkerlab/vaibhav/nhp-prior-data/processed/poisson_glm/data/prior_cond_${PRIOR_COND}_outcome_${OUTCOME_FILTER}"
N_NEURONS=$(ls "${DATA_DIR}"/*.parquet 2>/dev/null | wc -l)
if [[ ${N_NEURONS} -eq 0 ]]; then
    echo "Error: no parquet files found in ${DATA_DIR}" >&2
    exit 1
fi
CURRENT_TASKS=$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)
AVAILABLE=$(( QOS_LIMIT - CURRENT_TASKS ))
BATCH_SIZE=$(( AVAILABLE / N_NEURONS ))
[[ ${BATCH_SIZE} -lt 1 ]] && { echo "Error: no QOS slots available (${CURRENT_TASKS}/${QOS_LIMIT} used). Try again later." >&2; exit 1; }

BATCH=("${ALL_MODELS[@]:MODEL_START_IDX:BATCH_SIZE}")
N_BATCH=${#BATCH[@]}
NEXT_IDX=$((MODEL_START_IDX + N_BATCH))

echo "Neurons: ${N_NEURONS} | QOS limit: ${QOS_LIMIT} | Currently queued: ${CURRENT_TASKS} | Available: ${AVAILABLE} | Batch size: ${BATCH_SIZE} models"
echo "Submitting batch: models ${MODEL_START_IDX}..$((MODEL_START_IDX + N_BATCH - 1)) of ${N_ALL} | prior_cond=${PRIOR_COND}, outcome_filter=${OUTCOME_FILTER}"
echo

# ---------------------------------------------------------------------------
# Submit all models in this batch in parallel (no intra-batch dependency)
# ---------------------------------------------------------------------------
BATCH_JOB_IDS=()

for MODEL in "${BATCH[@]}"; do
    JOB_ID=$(
        SIF_FILE="${SIF_FILE}" \
        PRIOR_COND="${PRIOR_COND}" \
        OUTCOME_FILTER="${OUTCOME_FILTER}" \
        MODEL_FILE="${MODEL}" \
        SBATCH_EXTRA="" \
            bash "${SUBMIT_SCRIPT}" | grep -oP '(?<=Submitted batch job )\d+'
    )
    BATCH_JOB_IDS+=("${JOB_ID}")
    echo "  ${MODEL}: Job ${JOB_ID}"
done

echo
echo "Batch submitted: ${N_BATCH} models running in parallel ($(( N_BATCH * N_NEURONS )) total tasks)."

# ---------------------------------------------------------------------------
# If more models remain, schedule a launcher that waits for ALL batch jobs
# ---------------------------------------------------------------------------
if [[ ${NEXT_IDX} -lt ${N_ALL} ]]; then
    REMAINING=$((N_ALL - NEXT_IDX))
    # Build afterany:id1:id2:... dependency across all batch jobs
    DEPEND_STR=$(printf ":%s" "${BATCH_JOB_IDS[@]}")
    DEPEND_STR="afterany${DEPEND_STR}"

    LAUNCHER_ID=$(sbatch \
        --job-name="glm_launcher_${NEXT_IDX}" \
        --partition=ckpt-all \
        --account=walkerlab \
        --ntasks=1 --cpus-per-task=1 --mem=512M --time=00:10:00 \
        --dependency="${DEPEND_STR}" \
        --wrap="
            set -euo pipefail
            SIF_FILE='${SIF_FILE}' \
            PRIOR_COND='${PRIOR_COND}' \
            OUTCOME_FILTER='${OUTCOME_FILTER}' \
            QOS_LIMIT='${QOS_LIMIT}' \
            MODEL_START_IDX='${NEXT_IDX}' \
                bash '${THIS_SCRIPT}'
        " | grep -oP '(?<=Submitted batch job )\d+')

    echo "Launcher job ${LAUNCHER_ID} will submit the remaining ${REMAINING} model(s) after jobs ${BATCH_JOB_IDS[*]} finish."
else
    echo "All ${N_ALL} models queued."
fi
