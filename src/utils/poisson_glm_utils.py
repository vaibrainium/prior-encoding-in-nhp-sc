
import pickle
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal, stats
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, hstack, issparse, vstack
from scipy.special import gammaln
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')



def make_smooth_temporal_basis(duration, bin_size=1.0, filter_type="raised_cosine", center_spacing=None, n_bases=None):
    """Fast raised cosine basis function creation."""
    n_time_bins = int(np.ceil(duration / bin_size))

    center_spacing = 50 if center_spacing is None else center_spacing  # Default spacing

    if filter_type == "raised_cosine":
        n_bases = int(np.ceil(duration / center_spacing)) + 1
        time_bins = np.arange(1, n_time_bins + 1)
        bump_width = 4.0 * center_spacing
        centers = center_spacing * np.arange(1, n_bases + 1)

        basis_matrix = np.zeros((n_time_bins, n_bases))
        for i, center in enumerate(centers):
            x = time_bins - center
            mask = np.abs(x / bump_width) < 0.5
            basis_matrix[mask, i] = np.cos(x[mask] * 2.0 * np.pi / bump_width) * 0.5 + 0.5
    else:
        # Boxcar
        n_bases = int(np.ceil(duration / center_spacing)) if n_bases is None else n_bases
        basis_matrix = np.zeros((n_time_bins, n_bases))
        box_width = n_time_bins // n_bases
        for i in range(n_bases):
            start = int(i * box_width)
            end = min(int((i + 1) * box_width), n_time_bins-1) + 1
            basis_matrix[start:end, i] = 1.0 / (end - start)

    return basis_matrix

def make_post_spike_history_basis(duration_uniform_ms=11, n_uniform_bases=10, duration_nonlinear_ms=265, n_nonlinear_bases=10, bin_size=1.0):
    """
    Create post-spike history basis functions.
    Following neuroGLM approach with nonlinear spacing for faster dynamics near spike.
    10 1-ms uniform bases (to represent the fast refractory effects)
    followed by 10 raised cosine bases stretched in a logarithmic scale that spanned 265 ms
    """
    n_total_bases = n_uniform_bases + n_nonlinear_bases
    n_total_time_bins = int(np.ceil((duration_uniform_ms + duration_nonlinear_ms) / bin_size))
    basis_matrix = np.zeros((n_total_time_bins, n_total_bases))

    # Uniform boxcar bases for first 10 ms
    basis_matrix_uniform = make_smooth_temporal_basis(duration_uniform_ms, bin_size=bin_size, filter_type="boxcar", n_bases=n_uniform_bases)

    # Nonlinear spacing with log-transform for faster dynamics near spike
    n_time_bins = int(np.ceil((duration_nonlinear_ms) / bin_size))
    nl_offset = 1.0  # Offset for log nonlinearity
    nlin = lambda x: np.log(x + nl_offset)
    invnl = lambda x: np.exp(x) - nl_offset

    # Create centers in log-space
    y_range = nlin(np.array([0, n_time_bins]) + nl_offset)
    center_spacing = np.diff(y_range) / (n_nonlinear_bases - 1)
    centers = np.arange(y_range[0], y_range[1] + center_spacing/2, center_spacing)

    # Create time axis
    max_time = invnl(y_range[1] + 1.5*center_spacing) - nl_offset
    time_axis = np.arange(0, max_time, bin_size)

    # Raised cosine basis functions
    def cosine_bump(x, c, dc):
        return (np.cos(np.maximum(-np.pi, np.minimum(np.pi, (x-c)*np.pi/dc/2))) + 1) / 2

    basis_matrix_nonlinear = cosine_bump(
        nlin(time_axis + nl_offset)[:, None],
        centers[None, :],
        center_spacing
    )

    # Trim to desired duration
    basis_matrix_nonlinear = basis_matrix_nonlinear[:n_time_bins, :]

    basis_matrix[:basis_matrix_uniform.shape[0], :n_uniform_bases] = basis_matrix_uniform
    basis_matrix[basis_matrix_uniform.shape[0]:, n_uniform_bases:] = basis_matrix_nonlinear

    return basis_matrix


def convolve_with_basis(event_matrix, filter_type, duration, center_spacing=None, effect="causal"):
    """Fast convolution with temporal basis."""
    n_timebins = event_matrix.shape[0]
    n_events = event_matrix.shape[1]

    # Create basis
    basis = make_smooth_temporal_basis(duration, filter_type=filter_type, center_spacing=center_spacing)
    n_bases = basis.shape[1]
    if effect == "anti-causal":
        basis = basis[::-1, :]  # Reverse for anti-causal

    # Convolve each event type
    conv_matrix = np.zeros((n_timebins, n_events * n_bases))

    for e in range(n_events):
        for j in range(n_bases):
            conv_result = signal.convolve(event_matrix[:, e], basis[:, j], mode='full')
            if effect == "causal":
                conv_matrix[:, e * n_bases + j] = conv_result[:n_timebins]
            elif effect == "anti-causal":
                conv_matrix[:, e * n_bases + j] = conv_result[-n_timebins:]

    return conv_matrix, basis


def create_post_spike_history_matrix(spike_train):
    """
    Create post-spike history design matrix.

    Args:
        spike_train: binary spike train (1s and 0s)
        n_bases: number of basis functions for history filter
        history_duration: duration of history dependence in ms

    Returns:
        history_matrix: design matrix for post-spike history
        basis: basis functions used
    """
    n_timebins = len(spike_train)

    # Create history basis functions (causal only)
    basis = make_post_spike_history_basis()
    n_basis_bins = basis.shape[0]

    # Create history matrix by convolving spike train with basis functions
    history_matrix = np.zeros((n_timebins, basis.shape[1]))

    for i, spike_time in enumerate(np.where(spike_train)[0]):
        # For each spike, add basis functions starting one bin later (causal)
        start_time = spike_time + 1
        end_time = min(start_time + n_basis_bins, n_timebins)

        if start_time < n_timebins:
            basis_length = end_time - start_time
            history_matrix[start_time:end_time, :] += basis[:basis_length, :]

    return history_matrix


def reconstruct_kernels_from_weights(weights, config):
    """
    Reconstruct temporal kernels from GLM weights using organized configuration.

    Args:
        weights: Fitted GLM weights (excluding intercept)
        config: GLMConfig instance with all model parameters

    Returns:
        kernels: Dictionary containing reconstructed temporal kernels
    """
    kernels = {}

    # Calculate feature indices using config
    target_start = 0
    target_end = config.FEATURES_TARGET

    stim_start = target_end
    stim_end = stim_start + config.FEATURES_STIMULUS

    saccade_start = stim_end
    saccade_end = saccade_start + config.FEATURES_SACCADE

    bias_start = saccade_end
    bias_end = bias_start + config.FEATURES_BIAS

    history_start = bias_end
    history_end = history_start + config.FEATURES_HISTORY


    # 1. TARGET ONSET KERNEL
    target_weights = weights[target_start:target_end]
    target_basis = make_smooth_temporal_basis(
        config.TARGET_DURATION_MS,
        filter_type=config.TARGET_BASIS
    )
    kernels['target'] = {
        'kernel': target_basis @ target_weights,
        'time': np.arange(config.TARGET_DURATION_MS) if config.TARGET_EFFECT == config.EFFECT_CAUSAL else np.arange(config.TARGET_DURATION_MS) * -1,
        'weights': target_weights,
        'basis': target_basis,
        'duration_ms': config.TARGET_DURATION_MS,
        'n_bases': config.TARGET_N_BASES
    }

    # 2. STIMULUS KERNELS (ALL COHERENCE LEVELS)
    stim_basis = make_smooth_temporal_basis(
        config.STIMULUS_DURATION_MS,
        filter_type=config.STIMULUS_BASIS
    )

    # Individual coherence kernels
    for coh_idx in range(config.N_COHERENCE_LEVELS):
        coh_start = stim_start + coh_idx * config.STIMULUS_N_BASES
        coh_end = coh_start + config.STIMULUS_N_BASES
        stim_weights_subset = weights[coh_start:coh_end]

        kernels[f'stimulus_coh_{coh_idx}'] = {
            'kernel': stim_basis @ stim_weights_subset,
            'time': np.arange(config.STIMULUS_DURATION_MS) if config.STIMULUS_EFFECT == config.EFFECT_CAUSAL else np.arange(config.STIMULUS_DURATION_MS) * -1,
            'weights': stim_weights_subset,
            'basis': stim_basis,
            'duration_ms': config.STIMULUS_DURATION_MS,
            'n_bases': config.STIMULUS_N_BASES
        }

    # Average stimulus kernel
    all_stim_weights = weights[stim_start:stim_end].reshape(
        config.N_COHERENCE_LEVELS,
        config.STIMULUS_N_BASES
    )
    avg_stim_weights = np.mean(all_stim_weights, axis=0)
    kernels['stimulus'] = {
        'kernel': stim_basis @ avg_stim_weights,
        'time': np.arange(config.STIMULUS_DURATION_MS) if config.STIMULUS_EFFECT == config.EFFECT_CAUSAL else np.arange(config.STIMULUS_DURATION_MS) * -1,
        'weights': avg_stim_weights,
        'basis': stim_basis,
        'duration_ms': config.STIMULUS_DURATION_MS,
        'n_bases': config.STIMULUS_N_BASES
    }

    # 3. CHOICE/SACCADE KERNELS
    saccade_basis = make_smooth_temporal_basis(
        config.SACCADE_DURATION_MS,
        filter_type=config.SACCADE_BASIS
    )

    for choice_idx in range(config.N_CHOICE_OPTIONS):
        choice_start = saccade_start + choice_idx * config.SACCADE_N_BASES
        choice_end = choice_start + config.SACCADE_N_BASES
        choice_weights = weights[choice_start:choice_end]

        kernels[f'choice_{choice_idx}'] = {
            'kernel': saccade_basis @ choice_weights,
            'time': np.arange(0, config.SACCADE_DURATION_MS) if config.SACCADE_EFFECT == config.EFFECT_CAUSAL else np.arange(0, config.SACCADE_DURATION_MS) * -1,
            'weights': choice_weights,
            'basis': saccade_basis,
            'duration_ms': config.SACCADE_DURATION_MS,
            'n_bases': config.SACCADE_N_BASES
        }

    # 4. BIAS STATE KERNELS
    bias_basis = make_smooth_temporal_basis(
        config.BIAS_DURATION_MS,
        filter_type=config.BIAS_BASIS
    )

    for bias_idx in range(config.N_BIAS_STATES):
        bias_start_idx = bias_start + bias_idx * config.BIAS_N_BASES
        bias_end_idx = bias_start_idx + config.BIAS_N_BASES
        bias_weights = weights[bias_start_idx:bias_end_idx]

        bias_state_name = 'unbiased' if bias_idx == 0 else 'biased'
        kernels[f'bias_{bias_idx}'] = {
            'kernel': bias_basis @ bias_weights,
            'time': np.arange(0, config.BIAS_DURATION_MS) if config.BIAS_EFFECT == config.EFFECT_CAUSAL else np.arange(0, config.BIAS_DURATION_MS) * -1,
            'weights': bias_weights,
            'basis': bias_basis,
            'state_name': bias_state_name,
            'duration_ms': config.BIAS_DURATION_MS,
            'n_bases': config.BIAS_N_BASES
        }

    # 5. POST-SPIKE HISTORY KERNEL
    history_weights = weights[history_start:history_end]
    history_basis = make_post_spike_history_basis(
        duration_uniform_ms=config.HISTORY_UNIFORM_MS,
        n_uniform_bases=config.HISTORY_N_UNIFORM,
        duration_nonlinear_ms=config.HISTORY_NONLINEAR_MS,
        n_nonlinear_bases=config.HISTORY_N_NONLINEAR
    )

    kernels['history'] = {
        'kernel': history_basis @ history_weights,
        'time': np.arange(config.HISTORY_UNIFORM_MS + config.HISTORY_NONLINEAR_MS),
        'weights': history_weights,
        'basis': history_basis,
        'uniform_duration_ms': config.HISTORY_UNIFORM_MS,
        'nonlinear_duration_ms': config.HISTORY_NONLINEAR_MS,
        'n_uniform_bases': config.HISTORY_N_UNIFORM,
        'n_nonlinear_bases': config.HISTORY_N_NONLINEAR,
        'total_duration_ms': config.HISTORY_UNIFORM_MS + config.HISTORY_NONLINEAR_MS
    }

    # SUMMARY INFORMATION
    kernels['_config_summary'] = {
        'total_features_used': len(weights),
        'expected_features': config.get_total_features() - 1,  # Excluding intercept
        'feature_breakdown': {
            'target': config.FEATURES_TARGET,
            'stimulus': config.FEATURES_STIMULUS,
            'saccade': config.FEATURES_SACCADE,
            'bias': config.FEATURES_BIAS,
            'history': config.FEATURES_HISTORY
        },
        'temporal_parameters': {
            'target_duration_ms': config.TARGET_DURATION_MS,
            'stimulus_duration_ms': config.STIMULUS_DURATION_MS,
            'saccade_duration_ms': config.SACCADE_DURATION_MS,
            'bias_duration_ms': config.BIAS_DURATION_MS,
            'history_duration_ms': config.HISTORY_UNIFORM_MS + config.HISTORY_NONLINEAR_MS
        },
        'basis_parameters': {
            'target_n_bases': config.TARGET_N_BASES,
            'stimulus_n_bases': config.STIMULUS_N_BASES,
            'saccade_n_bases': config.SACCADE_N_BASES,
            'bias_n_bases': config.BIAS_N_BASES,
            'history_n_bases': config.HISTORY_N_UNIFORM + config.HISTORY_N_NONLINEAR
        }
    }

    return kernels
