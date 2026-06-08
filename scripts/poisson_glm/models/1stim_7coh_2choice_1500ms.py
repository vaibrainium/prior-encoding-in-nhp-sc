"""
Model: 1 stimulus basis, 7 coherence levels (continuous encoding), 2-choice saccade (1500 ms).
Stimulus is encoded as a single basis scaled by the coherence value.
    STIMULUS_DURATION_MS = 300
    SACCADE_DURATION_MS  = 1500
    N_COHERENCE_LEVELS   = 1  (continuous coherence scaling)
"""

import sys
from pathlib import Path

_MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MODELS_DIR))

from base import (
    StateBasedPoissonGLMConfig,
    get_feature_idx,
    build_design_matrix_continuous,
)


def build_design_matrix(trials, coh_levels, feature_idx):
    config = StateBasedPoissonGLMConfig()
    return build_design_matrix_continuous(trials, coh_levels, feature_idx, config)
