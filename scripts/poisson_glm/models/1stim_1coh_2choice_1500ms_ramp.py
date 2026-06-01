"""
Model: 1 stimulus level, 1 coherence level, 2-choice saccade (1500 ms) with evidence-accumulation ramp input.

Instead of a sustained boxcar (constant during stimulus period), the stimulus input is a
linearly-increasing ramp: stim_matrix[t] = elapsed_time_since_onset (in ms).
This drives the convolved regressor to capture the monotonically-accumulating evidence signal
rather than onset/offset transients.

    STIMULUS_DURATION_MS = 1500
    SACCADE_DURATION_MS  = 1500
    N_COHERENCE_LEVELS   = 1
    STIMULUS_RAMP        = True   ← ramp input instead of sustained boxcar

To also include log(RT) as a scalar covariate, set INCLUDE_LOG_RT = True
(see 1stim_1coh_2choice_1500ms_ramp_logrt.py).
"""

import sys
from pathlib import Path

_MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MODELS_DIR))

from base import StateBasedPoissonGLMConfig as _BaseConfig
from base import build_design_matrix_categorical, get_feature_idx


class StateBasedPoissonGLMConfig(_BaseConfig):
    STIMULUS_RAMP: bool = True


def build_design_matrix(trials, coh_levels, feature_idx):
    config = StateBasedPoissonGLMConfig()
    return build_design_matrix_categorical(trials, coh_levels, feature_idx, config)
