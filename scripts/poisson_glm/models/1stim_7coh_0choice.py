"""
Model: 1 stimulus basis, 7 coherence levels (continuous encoding), no saccade/choice.
Stimulus is encoded as a single basis scaled by the coherence value (not a
separate column per coherence level), so N_COHERENCE_LEVELS stays 1.
    SACCADE_DURATION_MS  = 0  → saccade/choice feature block disabled
    STIMULUS_DURATION_MS = 300
    N_COHERENCE_LEVELS   = 1  (continuous coherence scaling)
"""

import sys
from pathlib import Path

_MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MODELS_DIR))

from base import (
    StateBasedPoissonGLMConfig as _BaseConfig,
    get_feature_idx,
    build_design_matrix_continuous,
)


class StateBasedPoissonGLMConfig(_BaseConfig):
    SACCADE_DURATION_MS: int = 0


def build_design_matrix(trials, coh_levels, feature_idx):
    config = StateBasedPoissonGLMConfig()
    return build_design_matrix_continuous(trials, coh_levels, feature_idx, config)
