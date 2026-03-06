#!/bin/bash
# Submit SLURM array jobs for all model types.
# First attempts to submit all in parallel; if QOS limit is hit, falls back to
# a chain where each model waits for the previous one to finish.
#
# Usage:
#   SIF_FILE=/gscratch/walkerlab/vaibhav/prior-encoding-in-nhp.sif \
#       bash scripts/poisson_glm/slurm/submit_all_models.sh
#
#   SIF_FILE=/gscratch/walkerlab/vaibhav/prior-encoding-in-nhp.sif \
#   PRIOR_COND=equal_only OUTCOME_FILTER=correct_only \
#       bash scripts/poisson_glm/slurm/submit_all_models.sh
#
#
# Check progress with:
#   watch -n 10 'echo "=== $(date) ===" && squeue -u $USER -o "%.10i %.45j %.8T %.10M" && echo "" && echo "Summary:" && squeue -u $USER -h -o "%T" | sort | uniq -c'


set -euo pipefail

SIF_FILE="${SIF_FILE:-}"
PRIOR_COND="${PRIOR_COND:-equal_only}"
OUTCOME_FILTER="${OUTCOME_FILTER:-correct_only}"

if [[ -z "${SIF_FILE}" ]]; then
    echo "Error: SIF_FILE must be set. Example:" >&2
    echo "  SIF_FILE=/gscratch/walkerlab/vaibhav/prior-encoding-in-nhp.sif bash $0" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_neuron_cv.sh"

MODELS=(
    0stim_2choice_1500ms
    1stim_1coh_0choice
    1stim_1coh_2choice_1500ms
    1stim_7coh_0choice
    1stim_7coh_2choice_1500ms
    7stim_7coh_0choice
    7stim_7coh_2choice_1500ms
)

echo "Submitting ${#MODELS[@]} models for prior_cond=${PRIOR_COND}, outcome_filter=${OUTCOME_FILTER}"
echo

# ---------------------------------------------------------------------------
# Attempt 1: parallel (no dependency)
# ---------------------------------------------------------------------------
echo "Attempting parallel submission..."
PARALLEL_OK=true
SUBMITTED_IDS=()

for MODEL in "${MODELS[@]}"; do
    OUTPUT=$(
        SIF_FILE="${SIF_FILE}" \
        PRIOR_COND="${PRIOR_COND}" \
        OUTCOME_FILTER="${OUTCOME_FILTER}" \
        MODEL_FILE="${MODEL}" \
        SBATCH_EXTRA="" \
            bash "${SUBMIT_SCRIPT}" 2>&1
    ) || true

    JOB_ID=$(echo "${OUTPUT}" | grep -oP '(?<=Submitted batch job )\d+' || true)

    if [[ -z "${JOB_ID}" ]]; then
        echo "  ${MODEL}: FAILED (QOS limit or other error) — switching to chained mode."
        echo "${OUTPUT}" | grep -i "error" | head -2 || true
        PARALLEL_OK=false
        # Cancel any jobs already submitted in this attempt
        if [[ ${#SUBMITTED_IDS[@]} -gt 0 ]]; then
            echo "  Cancelling already-submitted jobs: ${SUBMITTED_IDS[*]}"
            scancel "${SUBMITTED_IDS[@]}" 2>/dev/null || true
        fi
        break
    fi

    SUBMITTED_IDS+=("${JOB_ID}")
    echo "  ${MODEL}: Job ${JOB_ID}"
done

if [[ "${PARALLEL_OK}" == true ]]; then
    echo
    echo "All ${#MODELS[@]} models submitted in parallel."
    exit 0
fi

# ---------------------------------------------------------------------------
# Attempt 2: chained (each waits for previous)
# ---------------------------------------------------------------------------
echo
echo "Falling back to chained submission..."
echo

PREV_JOB_ID=""

for MODEL in "${MODELS[@]}"; do
    DEPEND_ARG=""
    if [[ -n "${PREV_JOB_ID}" ]]; then
        DEPEND_ARG="--dependency=afterany:${PREV_JOB_ID}"
    fi

    PREV_JOB_ID=$(
        SIF_FILE="${SIF_FILE}" \
        PRIOR_COND="${PRIOR_COND}" \
        OUTCOME_FILTER="${OUTCOME_FILTER}" \
        MODEL_FILE="${MODEL}" \
        SBATCH_EXTRA="${DEPEND_ARG}" \
            bash "${SUBMIT_SCRIPT}" | grep -oP '(?<=Submitted batch job )\d+'
    )

    echo "  ${MODEL}: Job ${PREV_JOB_ID}"
done

echo
echo "All ${#MODELS[@]} models queued in a chain."
