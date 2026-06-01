"""
Model: 7 stimulus levels, 7 coherence levels (categorical encoding), 2-choice saccade (1500 ms) + log(RT) scalar covariate.
Each coherence level gets its own basis column block.
    N_COHERENCE_LEVELS   = 7
    STIMULUS_DURATION_MS = 300
    SACCADE_DURATION_MS  = 1500
    INCLUDE_LOG_RT       = True
"""

import sys
from pathlib import Path

_MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MODELS_DIR))

from base import StateBasedPoissonGLMConfig as _BaseConfig
from base import build_design_matrix_categorical, get_feature_idx


class StateBasedPoissonGLMConfig(_BaseConfig):
    N_COHERENCE_LEVELS: int = 7
    INCLUDE_LOG_RT: bool = True


def build_design_matrix(trials, coh_levels, feature_idx):
    config = StateBasedPoissonGLMConfig()
    return build_design_matrix_categorical(trials, coh_levels, feature_idx, config)
