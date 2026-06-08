"""
Model: 1 stimulus level, 1 coherence level, 2-choice saccade (1500 ms window).
All parameters match the base defaults — no overrides needed.
    STIMULUS_DURATION_MS = 300
    SACCADE_DURATION_MS  = 1500
    N_COHERENCE_LEVELS   = 1
"""

import sys
from pathlib import Path

_MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MODELS_DIR))

from base import (
    StateBasedPoissonGLMConfig,
    get_feature_idx,
    build_design_matrix_categorical,
)


def build_design_matrix(trials, coh_levels, feature_idx):
    config = StateBasedPoissonGLMConfig()
    return build_design_matrix_categorical(trials, coh_levels, feature_idx, config)
