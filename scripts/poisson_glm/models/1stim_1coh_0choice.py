"""
Model: 1 stimulus level, 1 coherence level, no saccade/choice.
    SACCADE_DURATION_MS  = 0  → saccade/choice feature block disabled
    STIMULUS_DURATION_MS = 300
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
    SACCADE_DURATION_MS: int = 0


def build_design_matrix(trials, coh_levels, feature_idx):
    config = StateBasedPoissonGLMConfig()
    return build_design_matrix_categorical(trials, coh_levels, feature_idx, config)
