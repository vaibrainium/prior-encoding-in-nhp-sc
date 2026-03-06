#!/usr/bin/env python
"""
Fit Poisson GLM with 5-fold cross-validation for a single neuron.
Designed for HPC array job parallelization — one job per neuron.

Usage:
    python scripts/fit_neuron_cv.py --neuron_id 42
    python scripts/fit_neuron_cv.py --neuron_id 42 --prior_cond equal_block --outcome_filter correct_only
"""

import argparse
import json
import pickle
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, issparse, vstack
from sklearn.model_selection import KFold

# --- Project root on sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import dir_config
from src.utils import poisson_glm_utils


def run_cv(neuron_id: int, prior_cond: str, outcome_filter: str, poisson_glm_config, model_name: str):

    feature_idx = get_feature_idx(poisson_glm_config)

    processed_dir = Path(dir_config.data.processed)
    data_dir = processed_dir / 'poisson_glm' / 'data' / f'prior_cond_{prior_cond}_outcome_{outcome_filter}'
    result_dir = processed_dir / 'poisson_glm' / 'models' / model_name
    result_dir.mkdir(parents=True, exist_ok=True)

    poisson_glm_config.save(result_dir / 'config.json')

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

        train_trials = df.iloc[train_idx]
        test_trials = df.iloc[test_idx]

        X_train, y_train = build_design_matrix(train_trials, coh_levels, feature_idx)
        X_test, _ = build_design_matrix(test_trials, coh_levels, feature_idx)

        model_result = poisson_glm_utils.fit_poisson_glm(X_train, y_train)
        if model_result is None:
            print(f"[neuron {neuron_id}] Fold {fold + 1}/5 — fit failed, aborting neuron.")
            break

        train_preds, onset = [], 0
        for t in train_trials.itertuples():
            n = int(t.n_bins)
            train_preds.append(model_result['predicted_y'][onset:onset + n])
            onset += n

        test_flat_pred = poisson_glm_utils.predict_poisson_glm(X_test, model_result)
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit Poisson GLM CV for one neuron.')
    parser.add_argument('--neuron_id', type=int, required=True, help='Neuron ID to process')
    parser.add_argument('--prior_cond', type=str, default='equal_block', help='Prior condition (default: equal_block)')
    parser.add_argument('--outcome_filter', type=str, default='correct_only', help='Outcome filter (default: correct_only)')
    parser.add_argument('--model_file', type=str, required=True, help='Path to model python file)')
    args = parser.parse_args()

    # load model functions from specified model file in scripts/poisson_glm/*.py
    model_name = args.model_file
    model_file_path = Path("scripts", "poisson_glm", f"{model_name}.py")
    if not model_file_path.exists():
        print(f"Model file not found: {model_file_path}", file=sys.stderr)
        sys.exit(1)

    # get build_design_matrix, get_feature_idx, and StateBasedPoissonGLMConfig from the model file
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_module", model_file_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    build_design_matrix = model_module.build_design_matrix
    get_feature_idx = model_module.get_feature_idx
    StateBasedPoissonGLMConfig = model_module.StateBasedPoissonGLMConfig

    poisson_glm_config = StateBasedPoissonGLMConfig()
    run_cv(args.neuron_id, args.prior_cond, args.outcome_filter, poisson_glm_config=poisson_glm_config, model_name=model_name)
