"""
Model: 7 stimulus levels, 7 coherence levels (categorical encoding), no saccade/choice.
Each coherence level gets its own basis column block.
    N_COHERENCE_LEVELS   = 7
    SACCADE_DURATION_MS  = 0  → saccade/choice feature block disabled
    STIMULUS_DURATION_MS = 300
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
    N_COHERENCE_LEVELS: int = 7
    SACCADE_DURATION_MS: int = 0


def build_design_matrix(trials, coh_levels, feature_idx):
    config = StateBasedPoissonGLMConfig()
    return build_design_matrix_categorical(trials, coh_levels, feature_idx, config)
