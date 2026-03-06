
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse, vstack

warnings.filterwarnings('ignore')

# --- Project root on sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import dir_config
from src.utils import poisson_glm_utils


class StateBasedPoissonGLMConfig:
    BIN_SIZE_MS: float = 1.0
    PRE_TARGET_MS: int = 50

    BASIS_RAISED_COSINE: str = "raised_cosine"
    BASIS_BOXCAR: str = "boxcar"

    EFFECT_CAUSAL: str = "causal"
    EFFECT_ANTI_CAUSAL: str = "anti-causal"

    TARGET_DURATION_MS: int = 200
    TARGET_SPACING_MS: int = 10
    TARGET_EFFECT: str = "causal"
    TARGET_BASIS: str = "raised_cosine"

    N_COHERENCE_LEVELS: int = 1
    STIMULUS_DURATION_MS: int = 300
    STIMULUS_SPACING_MS: int = 10
    STIMULUS_EFFECT: str = "causal"
    STIMULUS_BASIS: str = "raised_cosine"

    N_CHOICE_OPTIONS: int = 2
    SACCADE_DURATION_MS: int = 0
    SACCADE_SPACING_MS: int = 10
    SACCADE_EFFECT: str = "anti-causal"
    SACCADE_BASIS: str = "raised_cosine"

    N_BIAS_STATES: int = 1

    HISTORY_UNIFORM_MS: int = 11
    HISTORY_NONLINEAR_MS: int = 265
    HISTORY_N_UNIFORM: int = 10
    HISTORY_N_NONLINEAR: int = 10

    FEATURES_INTERCEPT: int = 1

    # computed properties
    @property
    def TARGET_N_BASES(self):
        return int(self.TARGET_DURATION_MS / self.TARGET_SPACING_MS + 1)

    @property
    def STIMULUS_N_BASES(self):
        return 0 if self.STIMULUS_DURATION_MS == 0 else int(self.STIMULUS_DURATION_MS / self.STIMULUS_SPACING_MS + 1)

    @property
    def SACCADE_N_BASES(self):
        return 0 if self.SACCADE_DURATION_MS == 0 else int(self.SACCADE_DURATION_MS / self.SACCADE_SPACING_MS + 1)

    # feature counts
    @property
    def FEATURES_TARGET(self):
        return self.TARGET_N_BASES * self.N_BIAS_STATES

    @property
    def FEATURES_STIMULUS(self):
        return self.STIMULUS_N_BASES * self.N_COHERENCE_LEVELS * self.N_BIAS_STATES

    @property
    def FEATURES_SACCADE(self):
        return self.SACCADE_N_BASES * self.N_CHOICE_OPTIONS * self.N_BIAS_STATES

    @property
    def FEATURES_HISTORY(self):
        return self.HISTORY_N_UNIFORM + self.HISTORY_N_NONLINEAR

    def get_total_features(self):
        return (
            self.FEATURES_TARGET +
            self.FEATURES_STIMULUS +
            self.FEATURES_SACCADE +
            self.FEATURES_HISTORY +
            self.FEATURES_INTERCEPT
        )

    # saving/loading
    def to_dict(self): return asdict(self)
    @classmethod
    def from_dict(cls, d): return cls(**d)
    def save(self, path): json.dump(self.to_dict(), open(path, "w"), indent=4)
    @classmethod
    def load(cls, path): return cls.from_dict(json.load(open(path)))

def get_feature_idx(config):
    feature_idx = {
        'target_start':   0,
        'target_end':     config.FEATURES_TARGET,
        'stim_start':     config.FEATURES_TARGET,
        'stim_end':       config.FEATURES_TARGET + config.FEATURES_STIMULUS,
        'saccade_start':  config.FEATURES_TARGET + config.FEATURES_STIMULUS,
        'saccade_end':    config.FEATURES_TARGET + config.FEATURES_STIMULUS + config.FEATURES_SACCADE,
        'history_start':  config.FEATURES_TARGET + config.FEATURES_STIMULUS + config.FEATURES_SACCADE,
        'history_end':    config.FEATURES_TARGET + config.FEATURES_STIMULUS + config.FEATURES_SACCADE + config.FEATURES_HISTORY,
        'intercept_idx':  config.get_total_features() - 1,
    }
    return feature_idx

def build_design_matrix(trials, coh_levels, feature_idx):

    config = StateBasedPoissonGLMConfig()
    total_features = config.get_total_features()

    trial_matrices = []
    trial_spike_trains = []

    for trial in trials.itertuples():
        trial_duration = int(trial.duration)
        trial_design = np.zeros((trial_duration, total_features))

        # 1. TARGET ONSET
        target_bin = int(trial.target_onset)
        if 0 < target_bin <= trial_duration:
            target_matrix = np.zeros((trial_duration, 1))
            target_matrix[target_bin - 1] = 1.0
            target_conv, _ = poisson_glm_utils.convolve_with_basis(
                target_matrix,
                config.TARGET_BASIS,
                config.TARGET_DURATION_MS,
                config.TARGET_SPACING_MS,
                effect=config.TARGET_EFFECT,
            )
            target_start = feature_idx['target_start'] + int(trial.state) * config.TARGET_N_BASES
            target_end = target_start + config.TARGET_N_BASES
            trial_design[:, target_start:target_end] = target_conv

        # 2. STIMULUS COHERENCE
        stim_bin = int(trial.stimulus_onset)
        resp_bin = int(trial.response_onset)
        if 0 < stim_bin < resp_bin <= trial_duration:
            stim_matrix = np.zeros((trial_duration, 1))
            stim_matrix[stim_bin - 1:resp_bin] = trial.coherence
            stim_conv, _ = poisson_glm_utils.convolve_with_basis(
                stim_matrix,
                config.STIMULUS_BASIS,
                config.STIMULUS_DURATION_MS,
                config.STIMULUS_SPACING_MS,
                effect=config.STIMULUS_EFFECT,
            )
            coh_idx = 0 if config.N_COHERENCE_LEVELS == 1 else int(np.where(coh_levels == trial.coherence)[0][0])
            state_idx = int(trial.state)
            coh_start = feature_idx['stim_start'] + coh_idx * config.STIMULUS_N_BASES + state_idx * config.STIMULUS_N_BASES * len(coh_levels)
            coh_end = coh_start + config.STIMULUS_N_BASES
            trial_design[:, coh_start:coh_end] = stim_conv

        # 3. SACCADE / CHOICE
        if 0 < resp_bin <= trial_duration:
            saccade_matrix = np.zeros((trial_duration, 1))
            saccade_matrix[resp_bin - 1] = 1.0
            saccade_conv, _ = poisson_glm_utils.convolve_with_basis(
                saccade_matrix,
                config.SACCADE_BASIS,
                config.SACCADE_DURATION_MS,
                config.SACCADE_SPACING_MS,
                effect=config.SACCADE_EFFECT,
            )
            choice_idx = int(trial.choice)
            state_idx = int(trial.state)
            choice_start = feature_idx['saccade_start'] + choice_idx * config.SACCADE_N_BASES + state_idx * config.SACCADE_N_BASES * config.N_CHOICE_OPTIONS
            choice_end = choice_start + config.SACCADE_N_BASES
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
