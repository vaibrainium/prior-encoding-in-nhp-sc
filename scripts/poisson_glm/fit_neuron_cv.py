#!/usr/bin/env python
"""
Fit Poisson GLM with 5-fold cross-validation for a single neuron.
Designed for HPC array job parallelization — one job per neuron.

Usage:
    python scripts/poisson_glm/fit_neuron_cv.py --neuron_id 42 --model_file 1stim_1coh_2choice_1500ms
    python scripts/poisson_glm/fit_neuron_cv.py --neuron_id 42 --model_file 7stim_7coh_2choice_1500ms --prior_cond equal_only --outcome_filter correct_only

Available model files (scripts/poisson_glm/models/):
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
    # ramp stimulus input (optional: also + log(RT))
    1stim_1coh_2choice_1500ms_ramp
    1stim_1coh_2choice_1500ms_ramp_logrt
"""

import argparse
import importlib.util
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

# --- Project root on sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import dir_config
from src.utils import poisson_glm_utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model_module(model_name: str):
    """Dynamically load a model file from the models/ subdirectory."""
    model_file = Path(__file__).parent / "models" / f"{model_name}.py"
    if not model_file.exists():
        print(f"Model file not found: {model_file}", file=sys.stderr)
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("model_module", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split_flat_predictions(flat_preds, trials):
    """Slice a flat prediction array back into per-trial arrays."""
    preds, onset = [], 0
    for t in trials.itertuples():
        n = int(t.n_bins)
        preds.append(flat_preds[onset:onset + n])
        onset += n
    return preds


# ---------------------------------------------------------------------------
# Main CV loop
# ---------------------------------------------------------------------------

def run_cv(neuron_id: int, prior_cond: str, outcome_filter: str, poisson_glm_config, model_name: str,
           build_design_matrix, get_feature_idx):

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
    coh_levels = np.sort(df['coherence'].unique())

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
            return

        train_preds = _split_flat_predictions(model_result['predicted_y'], train_trials)

        test_flat_pred = poisson_glm_utils.predict_poisson_glm(X_test, model_result)
        test_preds = _split_flat_predictions(test_flat_pred, test_trials)

        fitting_result['folds'].append({
            'fold': fold,
            'train_data': train_trials.reset_index(drop=True),
            'test_data': test_trials.reset_index(drop=True),
            'model': model_result,
            'train_predictions': train_preds,
            'test_predictions': test_preds,
        })

    if len(fitting_result['folds']) < 5:
        print(f"[neuron {neuron_id}] Only {len(fitting_result['folds'])}/5 folds completed — not saving.", file=sys.stderr)
        return

    # Write atomically: temp file → rename, so a partial write is never mistaken for success
    tmp_path = output_path.with_suffix('.tmp')
    with open(tmp_path, 'wb') as f:
        pickle.dump(fitting_result, f)
    tmp_path.rename(output_path)
    print(f"[neuron {neuron_id}] Saved to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit Poisson GLM CV for one neuron.')
    parser.add_argument('--neuron_id', type=int, required=True, help='Neuron ID to process')
    parser.add_argument('--prior_cond', type=str, default='equal_block', help='Prior condition (default: equal_block)')
    parser.add_argument('--outcome_filter', type=str, default='correct_only', help='Outcome filter (default: correct_only)')
    parser.add_argument('--model_file', type=str, required=True, help='Model name (stem of a file in models/)')
    args = parser.parse_args()

    model_module = _load_model_module(args.model_file)

    poisson_glm_config = model_module.StateBasedPoissonGLMConfig()

    run_cv(
        neuron_id=args.neuron_id,
        prior_cond=args.prior_cond,
        outcome_filter=args.outcome_filter,
        poisson_glm_config=poisson_glm_config,
        model_name=args.model_file,
        build_design_matrix=model_module.build_design_matrix,
        get_feature_idx=model_module.get_feature_idx,
    )
