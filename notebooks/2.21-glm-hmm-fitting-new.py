#!/usr/bin/env python
# coding: utf-8

import copy
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import numpy.random as npr
import pandas as pd
from sklearn import preprocessing

# Set up project root and import project-specific modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import dir_config, main_config
from notebooks.imports import *
from src.utils.glm_hmm_utils import *
from src.utils.glm_hmm_utils_cv import *

# Define directories
compiled_dir = Path(dir_config.data.compiled)
processed_dir = Path(dir_config.data.processed)
# Load session metadata
session_metadata = pd.read_csv(processed_dir / 'sessions_metadata.csv')
glm_hmm_config = main_config["GLM_HMM"]


# --- Helper Functions ---
def extract_previous_trial_data(session_data, valid_idx, first_trial):
    npr.seed(1)
    n_trials = session_data.shape[0] - first_trial
    prev_data = {}
    # For each previous trial feature, create an array for each trial back
    for var in PREV_TRIAL_FEATURES:
        prev_data[var] = np.empty((n_trials, N_TRIALS_BACK), dtype=int)
    # Loop through each trial starting from first_trial
    for i in range(first_trial, session_data.shape[0]):
        valid_indices = valid_idx[valid_idx < i][-N_TRIALS_BACK:]
        for _, var in enumerate(PREV_TRIAL_FEATURES):

            # var_col = var.split("_")[1]
            var_col = var[5:] if var.startswith("prev_") else var
            # If not enough valid previous trials, pad with zeros or another value
            padded_indices = np.pad(valid_indices, (N_TRIALS_BACK - len(valid_indices), 0), "constant", constant_values=0)

            if var == "prev_coherence":
                vals = session_data.coherence.iloc[padded_indices] * (2 * session_data.target.iloc[padded_indices] - 1)
            elif var == "prev_choice_outcome":
                prev_choices = session_data.choice.iloc[padded_indices] * 2 - 1
                prev_outcomes = session_data.outcome.iloc[padded_indices]
                vals = prev_choices * prev_outcomes
            elif var == "prev_choice_coherence":
                prev_choices = session_data.choice.iloc[padded_indices] * 2 - 1
                prev_coherences = session_data.coherence.iloc[padded_indices] * (2 * session_data.target.iloc[padded_indices] - 1)
                vals = prev_choices * prev_coherences
            elif var == "prev_coherence_choice_outcome":
                prev_choices = session_data.choice.iloc[padded_indices] * 2 - 1
                prev_outcomes = session_data.outcome.iloc[padded_indices]
                prev_coherences = session_data.coherence.iloc[padded_indices] * (2 * session_data.target.iloc[padded_indices] - 1)
                vals = prev_choices * prev_outcomes * prev_coherences
            else:
                vals = session_data[var_col].iloc[padded_indices]
            prev_data[var][i - first_trial] = vals * 2 - 1 if var_col in ["choice", "target"] else vals
    return prev_data

def prepare_input_data(data, valid_idx, first_trial):
    n_trials = data.shape[0] - first_trial
    X = np.zeros((1, n_trials, INPUT_DIM))
    # Fill current trial features
    for idx, feat in enumerate(CURRENT_TRIAL_FEATURES):
        if feat == "normalized_stimulus":
            X[0, :, idx] = data.coherence.values[first_trial:] * (2 * data.target.values[first_trial:] - 1) / 100
        elif feat == "bias":
            X[0, :, idx] = 1
        else:
            X[0, :, idx] = data[feat].values[first_trial:]
    # Fill previous trial features
    prev_data = extract_previous_trial_data(data, valid_idx, first_trial)
    col_idx = len(CURRENT_TRIAL_FEATURES)
    for var in PREV_TRIAL_FEATURES:
        for n in range(N_TRIALS_BACK):
            X[0, :, col_idx] = prev_data[var][:, n]
            col_idx += 1
    return list(X)


if __name__ == "__main__":

    CURRENT_TRIAL_FEATURES = glm_hmm_config["current_trial_features"] + ["bias"] if glm_hmm_config["add_bias"] else glm_hmm_config["current_trial_features"]
    PREV_TRIAL_FEATURES = glm_hmm_config["prev_trial_features"]
    N_TRIALS_BACK = glm_hmm_config["n_trials_back"]

    MODEL_FEATURES = CURRENT_TRIAL_FEATURES + [f"{var}_{n + 1}" for n in range(N_TRIALS_BACK) for var in PREV_TRIAL_FEATURES]
    glm_hmm_config["model_features"] = MODEL_FEATURES

    INPUT_DIM = len(MODEL_FEATURES)
    _TRIALS = glm_hmm_config["name"]

    # Pre-allocate lists for session data
    inputs_session_wise = []
    choices_session_wise = []
    invalid_idx_session_wise = []
    masks_session_wise = []
    GP_trial_num_session_wise = []
    prob_toRF_session_wise = []

    # Pre-build a mapping from session_id to prior_direction for efficient lookup
    prior_direction_map = session_metadata.set_index("session_id")["prior_direction"].to_dict()

    for session_id in session_metadata["session_id"]:

        # Read trial data for each session
        trial_data = pd.read_csv(Path(compiled_dir, session_id, f"{session_id}_trial.csv"), index_col=None)
        GP_trial_data = trial_data[trial_data.task_type == 1].reset_index(drop=True)

        block_switch = np.where((GP_trial_data.prob_toRF != 50) & ~np.isnan(GP_trial_data.prob_toRF))[0][0]
        if "uneq_prior_only" in _TRIALS:
            GP_trial_data = GP_trial_data[block_switch:].reset_index(drop=True)
        elif "eq_prior_only" in _TRIALS:
            GP_trial_data = GP_trial_data[:block_switch].reset_index(drop=True)

        # Fill missing values for important columns
        GP_trial_data['choice'] = GP_trial_data.choice.fillna(-1)
        GP_trial_data['target'] = GP_trial_data.target.fillna(-1)
        GP_trial_data['outcome'] = GP_trial_data.outcome.fillna(-1)

        # Get valid indices based on outcomes
        valid_idx = np.where(GP_trial_data.outcome >= 0)[0]

        # First valid trial considering n_trial_back
        first_trial = valid_idx[N_TRIALS_BACK - 1] + 1

        # Prepare inputs and choices
        inputs = prepare_input_data(GP_trial_data, valid_idx, first_trial)
        choices = GP_trial_data.choice.values.reshape(-1, 1).astype("int")[first_trial:]

        # Adjust invalid_idx and prepare mask
        invalid_idx = np.where(choices == -1)[0]

        if "masked" in _TRIALS:
            # For training, replace -1 with a random sample from 0,1
            choices[choices == -1] = np.random.choice(2, invalid_idx.shape[0])

            mask = np.ones_like(choices, dtype=bool)
            mask[invalid_idx] = 0
            # Get trial numbers and prob_toRF for the cropped session
            GP_trial_num = np.array(GP_trial_data.trial_number)[first_trial:]
            prob_toRF = np.array(GP_trial_data.prob_toRF)[first_trial:]
        else:
            assert "masked" in _TRIALS, "Invalid trials option"

        # Check prior_direction for the current session and adjust inputs and choices
        prior_direction = prior_direction_map.get(session_id, 'awayRF')
        if prior_direction == 'awayRF':
            for feat_idx, feature in enumerate(glm_hmm_config["model_features"]):
                if feature in glm_hmm_config["flip_columns"]:
                    inputs[0][:, feat_idx] = -inputs[0][:, feat_idx]
            choices = 1 - choices # Flip the choices

        assert len(choices) == len(inputs[0]), f"Length mismatch: {len(choices)} vs {len(inputs[0])}"
        assert len(mask) == len(inputs[0]), f"Length mismatch: {len(mask)} vs {len(inputs[0])}"
        assert len(GP_trial_num) == len(inputs[0]), f"Length mismatch: {len(GP_trial_num)} vs {len(inputs[0])}"
        assert len(prob_toRF) == len(inputs[0]), f"Length mismatch: {len(prob_toRF)} vs {len(inputs[0])}"


        # Append session-wise data to corresponding lists
        masks_session_wise.append(mask)
        inputs_session_wise += inputs
        choices_session_wise.append(choices)
        GP_trial_num_session_wise.append(GP_trial_num)
        prob_toRF_session_wise.append(prob_toRF)

    # Normalize inputs (excluding bias term)
    unnormalized_inputs_session_wise = copy.deepcopy(inputs_session_wise)
    for idx_session in range(len(session_metadata)):
        mask = masks_session_wise[idx_session][:, 0]
        for feat_idx, feature in enumerate(glm_hmm_config["model_features"]):
            if feature != "bias":
                inputs_session_wise[idx_session][mask, feat_idx] = preprocessing.scale(inputs_session_wise[idx_session][mask, feat_idx], axis=0)


    models_glm_hmm, fit_lls_glm_hmm = global_fit(choices_session_wise, inputs_session_wise, state_range=np.arange(1, 6), masks=masks_session_wise, n_iters=7000, n_initializations=20)

    # get best model of 20 initializations for each state
    init_params = {"glm_weights": {}, "transition_matrices": {}}
    for n_states in np.arange(1, 6):
        best_idx = fit_lls_glm_hmm[n_states].index(max(fit_lls_glm_hmm[n_states]))
        init_params["glm_weights"][n_states] = models_glm_hmm[n_states][best_idx].observations.params
        init_params["transition_matrices"][n_states] = models_glm_hmm[n_states][best_idx].transitions.params

    # session-wise fitting with 5 fold cross-validation
    models_session_state_fold, train_ll_session, test_ll_session = session_wise_fit_cv(
        choices_session_wise, inputs_session_wise, masks=masks_session_wise, n_sessions=len((session_metadata["session_id"])), init_params=init_params, state_range=np.arange(1, 6), n_iters=2500
    )

    # store data and models for aggregated
    global_fits = {"models": models_glm_hmm, "fits_lls_glm_hmm": fit_lls_glm_hmm, "init_params": init_params}
    session_wise_fits = {
        "models": models_session_state_fold,
        "train_ll": train_ll_session,
        "test_ll": test_ll_session,
    }

    # store data and models for session-wise
    session_data = {}
    for idx, session_id in enumerate(session_metadata["session_id"]):
        inputs = inputs_session_wise[idx]
        df = {
            "choices": choices_session_wise[idx].ravel(),
            "stimulus": unnormalized_inputs_session_wise[idx][:, 0],
            "mask": masks_session_wise[idx].ravel(),
            "trial_num": GP_trial_num_session_wise[idx].ravel(),
            "prob_toRF": prob_toRF_session_wise[idx].ravel(),
        }

        for i, feat in enumerate(MODEL_FEATURES):
            df[feat] = inputs[:, i]

        session_data[session_id] = pd.DataFrame(df)

    models_and_data = {
        "global": global_fits,
        "session_wise":session_wise_fits,
        "data": session_data,
        "config": glm_hmm_config,
    }

    with open(Path(processed_dir, "glm_hmm_models", f"{_TRIALS}.pkl"), "wb") as f:
        pickle.dump(models_and_data, f)
