"""
Model: 1 stimulus level, 1 coherence level, 2-choice saccade (1500 ms) + log(RT) scalar covariate.
All parameters match the base defaults except INCLUDE_LOG_RT.
    STIMULUS_DURATION_MS = 300
    SACCADE_DURATION_MS  = 1500
    N_COHERENCE_LEVELS   = 1
    INCLUDE_LOG_RT       = True
"""

import sys
from pathlib import Path

_MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MODELS_DIR))

from base import StateBasedPoissonGLMConfig as _BaseConfig
from base import build_design_matrix_categorical, get_feature_idx


class StateBasedPoissonGLMConfig(_BaseConfig):
    INCLUDE_LOG_RT: bool = True


def build_design_matrix(trials, coh_levels, feature_idx):
    config = StateBasedPoissonGLMConfig()
    return build_design_matrix_categorical(trials, coh_levels, feature_idx, config)
