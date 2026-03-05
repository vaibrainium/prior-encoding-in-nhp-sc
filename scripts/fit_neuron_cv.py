#!/usr/bin/env python
"""
Fit Poisson GLM with 5-fold cross-validation for a single neuron.
Designed for HPC array job parallelization — one job per neuron.

Usage:
    python scripts/fit_neuron_cv.py --neuron_id 42
    python scripts/fit_neuron_cv.py --neuron_id 42 --prior_cond equal_block --outcome_filter correct_only
"""

import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, issparse, vstack
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

# --- Project root on sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import dir_config
from config.poisson_glm_config import StateBasedPoissonGLMConfig
from src.utils import poisson_glm_utils

# --- Model config and feature index (mirrors notebook setup) ---
poisson_glm_config = StateBasedPoissonGLMConfig()

feature_idx = {
    'target_start':   0,
    'target_end':     poisson_glm_config.FEATURES_TARGET,
    'stim_start':     poisson_glm_config.FEATURES_TARGET,
    'stim_end':       poisson_glm_config.FEATURES_TARGET + poisson_glm_config.FEATURES_STIMULUS,
    'saccade_start':  poisson_glm_config.FEATURES_TARGET + poisson_glm_config.FEATURES_STIMULUS,
    'saccade_end':    poisson_glm_config.FEATURES_TARGET + poisson_glm_config.FEATURES_STIMULUS + poisson_glm_config.FEATURES_SACCADE,
    'history_start':  poisson_glm_config.FEATURES_TARGET + poisson_glm_config.FEATURES_STIMULUS + poisson_glm_config.FEATURES_SACCADE,
    'history_end':    poisson_glm_config.FEATURES_TARGET + poisson_glm_config.FEATURES_STIMULUS + poisson_glm_config.FEATURES_SACCADE + poisson_glm_config.FEATURES_HISTORY,
    'intercept_idx':  poisson_glm_config.get_total_features() - 1,
}


# ---------------------------------------------------------------------------
# GLM helpers (ported from notebook)
# ---------------------------------------------------------------------------

def build_design_matrix(trials, coh_levels):
    trial_matrices = []
    trial_spike_trains = []

    for trial in trials.itertuples():
        trial_duration = int(trial.duration)
        trial_design = np.zeros((trial_duration, poisson_glm_config.get_total_features()))

        # 1. TARGET ONSET
        target_bin = int(trial.target_onset)
        if 0 < target_bin <= trial_duration:
            target_matrix = np.zeros((trial_duration, 1))
            target_matrix[target_bin - 1] = 1.0
            target_conv, _ = poisson_glm_utils.convolve_with_basis(
                target_matrix,
                poisson_glm_config.TARGET_BASIS,
                poisson_glm_config.TARGET_DURATION_MS,
                poisson_glm_config.TARGET_SPACING_MS,
                effect=poisson_glm_config.TARGET_EFFECT,
            )
            target_start = feature_idx['target_start'] + int(trial.state) * poisson_glm_config.TARGET_N_BASES
            target_end = target_start + poisson_glm_config.TARGET_N_BASES
            trial_design[:, target_start:target_end] = target_conv

        # 2. STIMULUS COHERENCE
        stim_bin = int(trial.stimulus_onset)
        resp_bin = int(trial.response_onset)
        if 0 < stim_bin < resp_bin <= trial_duration:
            stim_matrix = np.zeros((trial_duration, 1))
            stim_matrix[stim_bin - 1:resp_bin] = 1.0
            stim_conv, _ = poisson_glm_utils.convolve_with_basis(
                stim_matrix,
                poisson_glm_config.STIMULUS_BASIS,
                poisson_glm_config.STIMULUS_DURATION_MS,
                poisson_glm_config.STIMULUS_SPACING_MS,
                effect=poisson_glm_config.STIMULUS_EFFECT,
            )
            coh_idx = 0 if poisson_glm_config.N_COHERENCE_LEVELS == 1 else int(np.where(coh_levels == trial.coherence)[0][0])
            state_idx = int(trial.state)
            coh_start = feature_idx['stim_start'] + coh_idx * poisson_glm_config.STIMULUS_N_BASES + state_idx * poisson_glm_config.STIMULUS_N_BASES * len(coh_levels)
            coh_end = coh_start + poisson_glm_config.STIMULUS_N_BASES
            trial_design[:, coh_start:coh_end] = stim_conv

        # 3. SACCADE / CHOICE
        if 0 < resp_bin <= trial_duration:
            saccade_matrix = np.zeros((trial_duration, 1))
            saccade_matrix[resp_bin - 1] = 1.0
            saccade_conv, _ = poisson_glm_utils.convolve_with_basis(
                saccade_matrix,
                poisson_glm_config.SACCADE_BASIS,
                poisson_glm_config.SACCADE_DURATION_MS,
                poisson_glm_config.SACCADE_SPACING_MS,
                effect=poisson_glm_config.SACCADE_EFFECT,
            )
            choice_idx = int(trial.choice)
            state_idx = int(trial.state)
            choice_start = feature_idx['saccade_start'] + choice_idx * poisson_glm_config.SACCADE_N_BASES + state_idx * poisson_glm_config.SACCADE_N_BASES * poisson_glm_config.N_CHOICE_OPTIONS
            choice_end = choice_start + poisson_glm_config.SACCADE_N_BASES
            trial_design[:, choice_start:choice_end] = saccade_conv

        # 4. POST-SPIKE HISTORY
        history_matrix = poisson_glm_utils.create_post_spike_history_matrix(trial.spike_train)
        trial_design[:, feature_idx['history_start']:feature_idx['history_end']] = history_matrix

        # 5. INTERCEPT
        trial_design[:, feature_idx['intercept_idx']] = 1.0

        trial_matrices.append(csr_matrix(trial_design))
        trial_spike_trains.append(trial.spike_train)

    X = vstack(trial_matrices, format='csr')
    y = np.concatenate(trial_spike_trains)
    return X, y


def fit_poisson_glm(X, y):
    if issparse(X):
        X = X.toarray()

    X_means = X.mean(axis=0)
    X_stds = X.std(axis=0)
    X_stds[X_stds < 1e-8] = 1.0

    X_scaled = X.copy()
    X_scaled[:, :-1] = (X[:, :-1] - X_means[:-1]) / X_stds[:-1]

    def loss_fun(w):
        eta = X_scaled @ w
        if np.any(y[eta < -15] > 0):
            return 1e20
        eta = np.clip(eta, -15, 15)
        mu = np.exp(eta)
        return np.sum(mu) - np.dot(y, eta) + 0.1 * np.dot(w, w)

    def grad_fun(w):
        eta = np.clip(X_scaled @ w, -15, 15)
        mu = np.exp(eta)
        return X_scaled.T @ (mu - y) + 0.02 * w

    w_init = np.zeros(X_scaled.shape[1])
    w_init[-1] = np.log(max(y.mean(), 1e-8))

    result = minimize(
        fun=loss_fun,
        x0=w_init,
        method='L-BFGS-B',
        jac=grad_fun,
        options={'maxiter': 1000, 'gtol': 1e-3, 'ftol': 1e-5, 'maxfun': 200},
    )

    if result.success or result.fun < 1e6:
        eta = X_scaled @ result.x
        result['predicted_y'] = np.exp(np.clip(eta, -15, 15))
    else:
        print(f"  Optimization failed: {result.message}")
        return None

    return result


def predict_poisson_glm(X, model):
    if issparse(X):
        X = X.toarray()
    X_means = X.mean(axis=0)
    X_stds = X.std(axis=0)
    X_stds[X_stds < 1e-8] = 1.0
    X_scaled = X.copy()
    X_scaled[:, :-1] = (X[:, :-1] - X_means[:-1]) / X_stds[:-1]
    return np.exp(np.clip(X_scaled @ model.x, -15, 15))


# ---------------------------------------------------------------------------
# Main CV loop
# ---------------------------------------------------------------------------

def run_cv(neuron_id: int, prior_cond: str, outcome_filter: str):
    processed_dir = Path(dir_config.data.processed)
    data_dir = processed_dir / 'poisson_glm' / 'data' / f'prior_cond_{prior_cond}_outcome_{outcome_filter}'
    result_dir = processed_dir / 'poisson_glm' / 'models' / f'prior_cond_{prior_cond}_outcome_{outcome_filter}'
    result_dir.mkdir(parents=True, exist_ok=True)

    output_path = result_dir / f'neuron_{neuron_id}'
    if output_path.exists():
        print(f"[neuron {neuron_id}] Already processed, skipping.")
        return

    fpath = data_dir / f'{neuron_id}.parquet'
    if not fpath.exists():
        print(f"[neuron {neuron_id}] Data file not found: {fpath}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(fpath)
    coh_levels = np.sort(df['coherence'].unique() / 100)

    kf = KFold(n_splits=5, shuffle=True, random_state=216)
    fitting_result = {'coh_levels': coh_levels, 'folds': []}

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        # print(f"[neuron {neuron_id}] Fold {fold + 1}/5")

        train_trials = df.iloc[train_idx]
        test_trials = df.iloc[test_idx]

        X_train, y_train = build_design_matrix(train_trials, coh_levels)
        X_test, _ = build_design_matrix(test_trials, coh_levels)

        model_result = fit_poisson_glm(X_train, y_train)
        if model_result is None:
            print(f"[neuron {neuron_id}] Fold {fold + 1}/5 — fit failed, aborting neuron.")
            break

        train_preds, onset = [], 0
        for t in train_trials.itertuples():
            n = int(t.n_bins)
            train_preds.append(model_result['predicted_y'][onset:onset + n])
            onset += n

        test_flat_pred = predict_poisson_glm(X_test, model_result)
        test_preds, onset = [], 0
        for t in test_trials.itertuples():
            n = int(t.n_bins)
            test_preds.append(test_flat_pred[onset:onset + n])
            onset += n

        fitting_result['folds'].append({
            'fold': fold,
            'train_data': train_trials.reset_index(drop=True),
            'test_data': test_trials.reset_index(drop=True),
            'model': model_result,
            'train_predictions': train_preds,
            'test_predictions': test_preds,
        })

    with open(output_path, 'wb') as f:
        pickle.dump(fitting_result, f)

    # print(f"[neuron {neuron_id}] Saved → {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit Poisson GLM CV for one neuron.')
    parser.add_argument('--neuron_id', type=int, required=True, help='Neuron ID to process')
    parser.add_argument('--prior_cond', type=str, default='equal_block', help='Prior condition (default: equal_block)')
    parser.add_argument('--outcome_filter', type=str, default='correct_only', help='Outcome filter (default: correct_only)')
    args = parser.parse_args()

    run_cv(args.neuron_id, args.prior_cond, args.outcome_filter)
