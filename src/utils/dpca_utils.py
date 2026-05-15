"""
Modular dPCA utilities for fitting, cross-period projection, and neuron/trial subsetting.

Key public functions
--------------------
extract_hmm_state_trial_info – build biased/unbiased trial-info dicts from HMM posteriors
extract_block_trial_info     – build equal/unequal trial-info dicts from prob_toRF
dpca_transform               – project data onto fitted dPCA axes, compute explained variance
create_dpca_matrix           – build trial-averaged + trial-wise data matrices
clean_dpca_data              – remove NaN-polluted timepoints (fit-safe vs. full variant)
fit_dpca_on_alignment        – fit one dPCA model on one alignment period
fit_dpca_all_alignments      – fit one model per alignment period
cross_period_projection      – for each fitted-period model, project all alignment periods
build_time_axes              – map alignment names → ms time vectors matching cleaned data
get_neuron_ids               – filter neurons by session / classification / leave-out fraction
split_trial_info_half        – 50/50 random trial split for fit-vs-test workflows

Condition dimension (first non-neuron axis)
-------------------------------------------
`create_dpca_matrix` and `split_trial_info_half` accept a generic `state_trial_info` dict:

    state_trial_info = {
        "biased":   biased_state_trial_info,   # dict[session_id → DataFrame]
        "unbiased": unbiased_state_trial_info,
    }

The keys must match `condition_dict["state_values"]`.  Swap in block-based trial dicts
(or any other grouping) without changing anything else:

    state_trial_info = {
        "equal":   equal_block_trial_info,
        "unequal": unequal_block_trial_info,
    }
"""

import numpy as np
from dPCA import dPCA as dPCAlib


# ──────────────────────────────────────────────────────────────────────────────
# Trial-info construction
# ──────────────────────────────────────────────────────────────────────────────

def extract_hmm_state_trial_info(
    session_metadata,
    glm_hmm_original,
    glm_hmm_data,
    confidence_threshold=0.8,
    compiled_dir=None,
):
    """Build biased/unbiased trial-info dicts from HMM posterior probabilities.

    Also flips stimulus/choice sign for awayRF sessions so all sessions are in
    the toRF reference frame (the convention used throughout dPCA).

    Parameters
    ----------
    session_metadata     : DataFrame with columns 'session_id', 'prior_direction'
    glm_hmm_original     : unmodified deep-copy of glm_hmm — used for inference
    glm_hmm_data         : glm_hmm["data"] — mutated in-place for awayRF sign flip
    confidence_threshold : float, default 0.8
    compiled_dir         : Path or str, optional. If provided, reaction_time is
                           loaded from each session's trial CSV and joined into
                           glm_hmm_data in-place (required by get_trial_num).

    Returns
    -------
    biased_state_trial_info   : dict[session_id → DataFrame]
    unbiased_state_trial_info : dict[session_id → DataFrame]
    state_occupancy           : dict[session_id → {"biased_state_trials": ..., "unbiased_state_trials": ...}]
    """
    import pandas as pd
    from pathlib import Path

    if compiled_dir is not None:
        compiled_dir = Path(compiled_dir)
        for session_id in glm_hmm_data:
            if "reaction_time" in glm_hmm_data[session_id].columns:
                continue
            trial_data = pd.read_csv(
                compiled_dir / session_id / f"{session_id}_trial_cleaned.csv", index_col=None
            )
            gp_trials = trial_data[trial_data.task_type == 1].reset_index(drop=True)
            valid_idx = np.where(gp_trials.outcome >= 0)[0]
            first_trial = valid_idx[0] + 1  # n_trial_back = 1
            reaction_time = np.array(gp_trials.reaction_time)[first_trial:]
            glm_hmm_data[session_id]["reaction_time"] = reaction_time

    state_occupancy           = {}
    biased_state_trial_info   = {}
    unbiased_state_trial_info = {}

    for session_id in session_metadata["session_id"]:
        model   = glm_hmm_original["model"]["models"][session_id]
        choices = glm_hmm_original["data"][session_id]["choices"].values.reshape(-1, 1)
        inputs  = np.array(
            glm_hmm_original["data"][session_id][
                ["normalized_stimulus", "bias", "prev_choice_1", "prev_target_1"]
            ]
        )
        mask = glm_hmm_original["data"][session_id]["mask"]
        mask = np.ones_like(choices, dtype=bool) if mask is None else np.array(mask)

        posterior_probs = model.expected_states(
            data=choices, input=inputs, mask=mask.reshape(-1, 1)
        )[0]
        biased_idx   = (posterior_probs[:, 0] > confidence_threshold) & mask.ravel()
        unbiased_idx = (posterior_probs[:, 1] > confidence_threshold) & mask.ravel()

        trial_nums = glm_hmm_original["data"][session_id]["trial_num"]
        state_occupancy[session_id] = {
            "biased_state_trials":   trial_nums[biased_idx],
            "unbiased_state_trials": trial_nums[unbiased_idx],
        }

        prior_dir = session_metadata.loc[
            session_metadata.session_id == session_id, "prior_direction"
        ].values[0]
        if prior_dir == "awayRF":
            d = glm_hmm_data[session_id]
            d["choices"]             = 1 - d["choices"]
            d["stimulus"]            = -d["stimulus"]
            d["normalized_stimulus"] = -d["normalized_stimulus"]
            d["prev_choice_1"]       = -d["prev_choice_1"]
            d["prev_target_1"]       = -d["prev_target_1"]

        d = glm_hmm_data[session_id]
        biased_state_trial_info[session_id]   = d.loc[d.trial_num.isin(state_occupancy[session_id]["biased_state_trials"])]
        unbiased_state_trial_info[session_id] = d.loc[d.trial_num.isin(state_occupancy[session_id]["unbiased_state_trials"])]

    return biased_state_trial_info, unbiased_state_trial_info, state_occupancy


def extract_block_trial_info(glm_hmm_data, sessions):
    """Build equal/unequal block trial-info dicts from the prob_toRF column.

    prob_toRF == 50  → equal block (symmetric prior)
    prob_toRF != 50  → unequal block (asymmetric prior)

    Parameters
    ----------
    glm_hmm_data : glm_hmm["data"] — dict[session_id → DataFrame]
    sessions     : iterable of session_id strings

    Returns
    -------
    equal_block_trial_info   : dict[session_id → DataFrame]
    unequal_block_trial_info : dict[session_id → DataFrame]
    """
    equal_block_trial_info   = {}
    unequal_block_trial_info = {}

    for session_id in sessions:
        d = glm_hmm_data[session_id]
        equal_block_trial_info[session_id]   = d[d.prob_toRF == 50].reset_index(drop=True)
        unequal_block_trial_info[session_id] = d[d.prob_toRF != 50].reset_index(drop=True)

    return equal_block_trial_info, unequal_block_trial_info


# ──────────────────────────────────────────────────────────────────────────────
# Core transform
# ──────────────────────────────────────────────────────────────────────────────

def dpca_transform(dpca, X):
    """Project X onto fitted dPCA axes and attach explained_variance_ratio_ to the model.

    Parameters
    ----------
    dpca : fitted dPCA model
    X    : ndarray, shape (n_neurons, *condition_dims, n_time)

    Returns
    -------
    dpca : same model, with .explained_variance_ratio_ updated
    Z    : dict[marginalization_key] → ndarray
    """
    X = X - np.nanmean(X.reshape((X.shape[0], -1)), 1).reshape(
        (X.shape[0],) + (len(X.shape) - 1) * (1,)
    )
    total_variance = np.nansum((X - np.nanmean(X)) ** 2)

    def marginal_variances(marginal):
        D, Xr = dpca.D[marginal], X.reshape((X.shape[0], -1))
        Xr_no_nan = np.where(np.isnan(Xr), 0, Xr)  # NaN timepoints contribute 0 after mean-centering
        return [np.sum(np.dot(D[:, k], Xr_no_nan) ** 2) / total_variance for k in range(D.shape[1])]

    Z = {}
    dpca.explained_variance_ratio_ = {}
    for key in list(dpca.marginalizations.keys()):
        Z[key] = np.dot(dpca.D[key].T, X.reshape((X.shape[0], -1))).reshape(
            (dpca.D[key].shape[1],) + X.shape[1:]
        )
        dpca.explained_variance_ratio_[key] = marginal_variances(key)
    return dpca, Z


# ──────────────────────────────────────────────────────────────────────────────
# Data matrix construction
# ──────────────────────────────────────────────────────────────────────────────

def create_dpca_matrix(
    sessions,
    condition_dict,
    session_neuron_ids,
    state_trial_info,
    neuron_metadata,
    ephys,
    ephys_config,
    condition_type="states",
):
    """Build trial-averaged and trial-wise dPCA data matrices.

    Parameters
    ----------
    sessions           : iterable of session_id strings
    condition_dict     : dict with keys:
                           'state_values' – list of condition labels along the first
                                            condition axis.
                                            Must be ["biased", "unbiased"] when
                                            condition_type="states", or any labels
                                            (e.g. ["block1", "block2"]) when
                                            condition_type="blocks".
                           'coherences'   – list of coherence values
                           'choices'      – list of choice labels
    session_neuron_ids : 1-D array of neuron_ids to include (determines output dim 0)
    state_trial_info   : dict[state_label → dict[session_id → DataFrame]]
                         Keys must match condition_dict["state_values"].
    neuron_metadata    : DataFrame with columns 'session_id', 'neuron_id'
    ephys              : dict[alignment][neuron_id] → spike data
    ephys_config       : config dict with key 'alignment_settings_GP'
    condition_type     : "states" or "blocks"
                         "states" – first condition axis is HMM states (biased/unbiased)
                         "blocks" – first condition axis is trial blocks (any labels)

    Returns
    -------
    dPCA_averaged_data   : dict[alignment] → ndarray
                             shape (n_neurons, n_states, n_coh, n_choices, n_time)
    dPCA_trial_wise_data : dict[alignment] → ndarray
                             shape (250, n_neurons, n_states, n_coh, n_choices, n_time)
    """
    if condition_type not in ("states", "blocks"):
        raise ValueError(f"condition_type must be 'states' or 'blocks', got {condition_type!r}")
    if condition_type == "states":
        expected = {"biased", "unbiased"}
        if set(condition_dict["state_values"]) != expected:
            raise ValueError(
                f"condition_type='states' requires state_values {expected}, "
                f"got {condition_dict['state_values']}"
            )
    missing = set(condition_dict["state_values"]) - set(state_trial_info.keys())
    if missing:
        raise KeyError(f"state_trial_info is missing keys: {missing}")

    from src.utils import ephys_utils

    alignment_settings = ephys_config["alignment_settings_GP"]
    n_neurons    = len(session_neuron_ids)
    n_states     = len(condition_dict["state_values"])
    n_coherences = len(condition_dict["coherences"])
    n_choices    = len(condition_dict["choices"])

    dPCA_averaged_data = {
        event: np.full(
            [n_neurons, n_states, n_coherences, n_choices,
             alignment_settings[event]["end_time_ms"] - alignment_settings[event]["start_time_ms"] + 1],
            np.nan,
        )
        for event in alignment_settings
    }
    dPCA_trial_wise_data = {
        event: np.full(
            [250, n_neurons, n_states, n_coherences, n_choices,
             alignment_settings[event]["end_time_ms"] - alignment_settings[event]["start_time_ms"] + 1],
            np.nan,
        )
        for event in alignment_settings
    }

    for alignment in alignment_settings:
        for session_id in sessions:
            neuron_ids = neuron_metadata.neuron_id[neuron_metadata.session_id == session_id].values
            neuron_ids = neuron_ids[np.isin(neuron_ids, session_neuron_ids)]
            for state_idx, state in enumerate(condition_dict["state_values"]):
                trial_info = state_trial_info[state]
                for coherence_idx, coherence in enumerate(condition_dict["coherences"]):
                    for choice_idx, choice in enumerate(condition_dict["choices"]):
                        trials = ephys_utils.get_trial_num(
                            trial_info[session_id], coherence=coherence, choice=choice_idx, outcome=1
                        )
                        for neuron_id in neuron_ids:
                            trial_wise_data = ephys_utils.get_neural_data_from_trial_num(
                                ephys[alignment][neuron_id], trials, type="convolved_spike_trains"
                            )
                            if trial_wise_data.shape[0] == 0:
                                print(f"No trials: {session_id}, {state}, {neuron_id}, {coherence, choice}")
                                continue
                            averaged_data = np.nanmean(trial_wise_data, axis=0)
                            n_idx = np.where(session_neuron_ids == neuron_id)[0]

                            if alignment == "cue":
                                non_nan_tp = np.where(
                                    np.sum(np.isnan(trial_wise_data), axis=0) / len(trials) <= 0.7
                                )[0]
                                averaged_data = averaged_data[: non_nan_tp[-1] + 1]
                                dPCA_averaged_data[alignment][
                                    n_idx, state_idx, coherence_idx, choice_idx, : len(averaged_data)
                                ] = averaged_data
                            elif alignment == "response":
                                non_nan_tp = np.where(
                                    np.sum(np.isnan(trial_wise_data), axis=0) / len(trials) <= 0.7
                                )[0]
                                averaged_data = averaged_data[non_nan_tp[0] :]
                                dPCA_averaged_data[alignment][
                                    n_idx, state_idx, coherence_idx, choice_idx, -len(averaged_data) :
                                ] = averaged_data
                            else:
                                dPCA_averaged_data[alignment][
                                    n_idx, state_idx, coherence_idx, choice_idx, :
                                ] = averaged_data

                            dPCA_trial_wise_data[alignment][
                                : trial_wise_data.shape[0],
                                n_idx,
                                state_idx,
                                coherence_idx,
                                choice_idx,
                                :,
                            ] = np.expand_dims(trial_wise_data, axis=1)

    return dPCA_averaged_data, dPCA_trial_wise_data


# ──────────────────────────────────────────────────────────────────────────────
# NaN cleaning
# ──────────────────────────────────────────────────────────────────────────────

def clean_dpca_data(averaged_data, trial_wise_data, alignments):
    """Remove NaN-polluted timepoints and return two variants of the cleaned arrays.

    fit_avg / fit_tw   – any-NaN timepoints removed; safe to pass directly to dPCA.fit()
    full_avg / full_tw – only all-NaN timepoints removed; used for cross-period projection

    Parameters
    ----------
    averaged_data   : dict[alignment] → ndarray (n_neurons, *cond_dims, n_time)
    trial_wise_data : dict[alignment] → ndarray (n_trials, n_neurons, *cond_dims, n_time)
    alignments      : iterable of alignment keys

    Returns
    -------
    fit_avg, fit_tw, full_avg, full_tw : four dicts with the same alignment keys
    """
    fit_avg  = {a: averaged_data[a].copy() for a in alignments}
    fit_tw   = {a: trial_wise_data[a].copy() for a in alignments}
    full_avg = {a: averaged_data[a].copy() for a in alignments}
    full_tw  = {a: trial_wise_data[a].copy() for a in alignments}

    for alignment in alignments:
        # fit arrays: drop timepoints where ANY condition has a NaN average
        no_any_nan = ~np.any(np.isnan(fit_avg[alignment]), axis=(0, 1, 2, 3))
        fit_avg[alignment] = fit_avg[alignment][..., no_any_nan]
        fit_tw[alignment]  = fit_tw[alignment][..., no_any_nan]

        # full arrays: drop timepoints where ALL conditions are NaN
        no_all_nan = ~np.all(np.isnan(full_avg[alignment]), axis=(0, 1, 2, 3))
        full_avg[alignment] = full_avg[alignment][..., no_all_nan]
        full_tw[alignment]  = full_tw[alignment][..., no_all_nan]

        # drop all-NaN trial rows
        nonnan_trial = ~np.all(np.isnan(fit_tw[alignment]), axis=(1, 2, 3, 4, 5))
        fit_tw[alignment] = fit_tw[alignment][nonnan_trial]

        nonnan_trial_full = ~np.all(np.isnan(full_tw[alignment]), axis=(1, 2, 3, 4, 5))
        full_tw[alignment] = full_tw[alignment][nonnan_trial_full]

    return fit_avg, fit_tw, full_avg, full_tw


# ──────────────────────────────────────────────────────────────────────────────
# dPCA fitting
# ──────────────────────────────────────────────────────────────────────────────

def fit_dpca_on_alignment(fit_averaged, fit_trial_wise, n_components=3, marginalization_keys=None):
    """Fit a single dPCA model on one alignment's cleaned data.

    Parameters
    ----------
    fit_averaged         : ndarray (n_neurons, n_states, n_coh, n_choices, n_time) — no NaNs
    fit_trial_wise       : ndarray (n_trials, n_neurons, ..., n_time)
    n_components         : int, number of components per marginalization
    marginalization_keys : list of label chars, default ['b', 's', 'c', 't']

    Returns
    -------
    dpca : fitted model with .explained_variance_ratio_ set
    Z    : dict[marginalization_key] → transformed data on the fit data
    """
    if marginalization_keys is None:
        marginalization_keys = ['b', 's', 'c', 't']
    dpca = dPCAlib.dPCA(
        n_components=n_components,
        labels=''.join(marginalization_keys),
        regularizer=0,
    )
    dpca.protect = ['t']
    dpca.fit(fit_averaged, fit_trial_wise)
    dpca, Z = dpca_transform(dpca, fit_averaged)
    return dpca, Z


def fit_dpca_all_alignments(fit_avg, fit_tw, alignments, n_components=3, marginalization_keys=None):
    """Fit one dPCA model per alignment.

    Returns
    -------
    dict[alignment] = {"model": dpca, "transformed_data": Z}
    """
    results = {}
    for alignment in alignments:
        dpca, Z = fit_dpca_on_alignment(
            fit_avg[alignment], fit_tw[alignment], n_components, marginalization_keys
        )
        results[alignment] = {"model": dpca, "transformed_data": Z}
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Cross-period projection
# ──────────────────────────────────────────────────────────────────────────────

def cross_period_projection(dpca_results, avg_data, alignments):
    """For each fitted-period model, project the average data from every alignment.

    Parameters
    ----------
    dpca_results : dict[fit_alignment] = {"model": dpca, ...}
                   Typically the output of fit_dpca_all_alignments.
    avg_data     : dict[alignment] → ndarray  (n_neurons, *cond_dims, n_time)
                   Use full_avg (all-NaN timepoints removed) so time axes differ per alignment.
    alignments   : iterable of alignment keys

    Returns
    -------
    projections : dict[fit_alignment][project_alignment] → Z dict
    """
    projections = {}
    for fit_align in alignments:
        model = dpca_results[fit_align]["model"]
        projections[fit_align] = {}
        for proj_align in alignments:
            _, Z = dpca_transform(model, avg_data[proj_align])
            projections[fit_align][proj_align] = Z
    return projections


# ──────────────────────────────────────────────────────────────────────────────
# Time axis helper
# ──────────────────────────────────────────────────────────────────────────────

def build_time_axes(avg_data, ephys_config, alignments=None):
    """Return ms time vectors matching the (already-cleaned) data for each alignment.

    Parameters
    ----------
    avg_data     : dict[alignment] → ndarray whose last dim gives n_time after cleaning
    ephys_config : config dict with key 'alignment_settings_GP'
    alignments   : alignments to build axes for; defaults to all keys in avg_data

    Returns
    -------
    dict[alignment] → 1-D ndarray of ms time points
    """
    if alignments is None:
        alignments = list(avg_data.keys())
    alignment_settings = ephys_config["alignment_settings_GP"]
    time_axes = {}
    for alignment in alignments:
        n_time = avg_data[alignment].shape[-1]
        full_range = np.arange(
            alignment_settings[alignment]["start_time_ms"],
            alignment_settings[alignment]["end_time_ms"] + 1,
        )
        time_axes[alignment] = full_range[-n_time:] if alignment == "response" else full_range[:n_time]
    return time_axes


# ──────────────────────────────────────────────────────────────────────────────
# Neuron filtering
# ──────────────────────────────────────────────────────────────────────────────

def get_neuron_ids(
    neuron_metadata,
    sessions,
    exclude_cell_types=None,
    leave_out_fraction=None,
    rng=None,
):
    """Return an array of neuron_ids for the given sessions after optional filtering.

    Parameters
    ----------
    neuron_metadata    : DataFrame with columns 'session_id', 'neuron_id', 'classification'
    sessions           : iterable of session_id strings
    exclude_cell_types : list of classification values to drop; defaults to ["trash"]
                         pass [] to include all cell types
    leave_out_fraction : float in (0, 1) — randomly remove this fraction of neurons
                         e.g. 0.1 for 10% leave-out
    rng                : np.random.Generator for reproducibility; created if None

    Returns
    -------
    neuron_ids : sorted 1-D ndarray
    """
    if exclude_cell_types is None:
        exclude_cell_types = ["trash"]
    mask = neuron_metadata.session_id.isin(sessions)
    mask &= ~neuron_metadata.classification.isin(exclude_cell_types)
    neuron_ids = neuron_metadata.neuron_id[mask].values

    if leave_out_fraction is not None and leave_out_fraction > 0:
        if rng is None:
            rng = np.random.default_rng()
        n_keep = int(np.round(len(neuron_ids) * (1 - leave_out_fraction)))
        neuron_ids = rng.choice(neuron_ids, size=n_keep, replace=False)

    return np.sort(neuron_ids)


# ──────────────────────────────────────────────────────────────────────────────
# Trial splitting
# ──────────────────────────────────────────────────────────────────────────────

def split_trial_info_half(state_trial_info, sessions, seed=None):
    """Randomly split each session's trials into two halves, for every state/condition.

    Parameters
    ----------
    state_trial_info : dict[state_label → dict[session_id → DataFrame]]
                       Same structure as the argument to create_dpca_matrix.
    sessions         : iterable of session_id strings
    seed             : int or None — passed to np.random.default_rng

    Returns
    -------
    half1, half2 : two dicts with the same structure as state_trial_info
                   Use one half to build the fit matrix and the other for held-out projection.
    """
    rng = np.random.default_rng(seed)
    half1 = {state: {} for state in state_trial_info}
    half2 = {state: {} for state in state_trial_info}

    for state, trial_info in state_trial_info.items():
        for session_id in sessions:
            df = trial_info[session_id]
            idx = rng.permutation(len(df))
            mid = len(idx) // 2
            half1[state][session_id] = df.iloc[idx[:mid]].reset_index(drop=True)
            half2[state][session_id] = df.iloc[idx[mid:]].reset_index(drop=True)

    return half1, half2
