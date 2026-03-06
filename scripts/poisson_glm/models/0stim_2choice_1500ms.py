"""
Model: no stimulus, 2-choice saccade (1500 ms window).
    STIMULUS_DURATION_MS = 0  → stimulus feature block disabled
    SACCADE_DURATION_MS  = 1500
    N_COHERENCE_LEVELS   = 1
"""

import sys
from pathlib import Path

_MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MODELS_DIR))

from base import (
    StateBasedPoissonGLMConfig as _BaseConfig,
    get_feature_idx,
    build_design_matrix_categorical,
)


class StateBasedPoissonGLMConfig(_BaseConfig):
    STIMULUS_DURATION_MS: int = 0


def build_design_matrix(trials, coh_levels, feature_idx):
    config = StateBasedPoissonGLMConfig()
    return build_design_matrix_categorical(trials, coh_levels, feature_idx, config)
