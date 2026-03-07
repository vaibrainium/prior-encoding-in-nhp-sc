"""
Model: 1 stimulus level, 1 coherence level, 2-choice saccade (1500 ms)
with evidence-accumulation ramp input AND log(RT) scalar covariate.

Combines both optional extensions:
  - STIMULUS_RAMP   = True  → linearly-increasing ramp as stimulus input
  - INCLUDE_LOG_RT  = True  → log(RT) column constant within each trial

Together these allow the model to capture:
  - The time-course of evidence accumulation via the ramp-convolved basis
  - The global integration-time effect via log(RT) scalar

    STIMULUS_DURATION_MS = 1500
    SACCADE_DURATION_MS  = 1500
    N_COHERENCE_LEVELS   = 1
    STIMULUS_RAMP        = True
    INCLUDE_LOG_RT       = True
"""

import sys
from pathlib import Path

_MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MODELS_DIR))

from base import StateBasedPoissonGLMConfig as _BaseConfig
from base import build_design_matrix_categorical, get_feature_idx


class StateBasedPoissonGLMConfig(_BaseConfig):
    STIMULUS_RAMP: bool = True
    INCLUDE_LOG_RT: bool = True


def build_design_matrix(trials, coh_levels, feature_idx):
    config = StateBasedPoissonGLMConfig()
    return build_design_matrix_categorical(trials, coh_levels, feature_idx, config)
