"""
Modular dPCA utilities for fitting, cross-period projection, and neuron/trial subsetting.

Key public functions
--------------------
extract_hmm_state_trial_info          – build biased/unbiased trial-info dicts from HMM posteriors
extract_block_trial_info              – build equal/unequal trial-info dicts from prob_toRF
dpca_transform                        – project data onto fitted dPCA axes, compute explained variance
create_dpca_matrix                    – build trial-averaged + trial-wise data matrices
clean_dpca_data                       – remove NaN-polluted timepoints (fit-safe vs. full variant)
fit_dpca_on_alignment                 – fit one dPCA model on one alignment period
fit_dpca_all_alignments               – fit one model per alignment period
cross_period_projection               – for each fitted-period model, project all alignment periods
build_time_axes                       – map alignment names → ms time vectors matching cleaned data
get_neuron_ids                        – filter neurons by session / classification / leave-out fraction
split_trial_info_half                 – 50/50 random trial split for fit-vs-test workflows
dpca_significance_analysis            – nearest-centroid significance masks with shuffled null;
                                        pass key_groups={'s': [[0,1],[2,3]]} for binary low/high test

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

    if compiled_dir is None:
        raise ValueError(
            "Compiled directory is missing! Provide compiled_dir to load reaction_time from trial CSVs."
        )
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
    outcome=1,
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
            [300, n_neurons, n_states, n_coherences, n_choices,
             alignment_settings[event]["end_time_ms"] - alignment_settings[event]["start_time_ms"] + 1],
            np.nan, dtype=np.float32,
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
                    # Support grouped coherences: list/tuple pools trials from multiple values into one bin
                    coh_values = coherence if isinstance(coherence, (list, tuple)) else [coherence]
                    for choice_idx, choice in enumerate(condition_dict["choices"]):
                        trials = np.concatenate([
                            ephys_utils.get_trial_num(
                                trial_info[session_id], coherence=coh, choice=choice_idx, outcome=outcome
                            )
                            for coh in coh_values
                        ], axis=0)
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

def clean_dpca_data(averaged_data, trial_wise_data, alignments, compute_full=True):
    """Remove NaN-polluted timepoints and return two variants of the cleaned arrays.

    fit_avg / fit_tw   – any-NaN timepoints removed; safe to pass directly to dPCA.fit()
    full_avg / full_tw – only all-NaN timepoints removed; used for cross-period projection
                         (None, None when compute_full=False — saves ~2× peak memory)

    Parameters
    ----------
    averaged_data   : dict[alignment] → ndarray (n_neurons, *cond_dims, n_time)
    trial_wise_data : dict[alignment] → ndarray (n_trials, n_neurons, *cond_dims, n_time)
    alignments      : iterable of alignment keys
    compute_full    : if False, skip full_avg / full_tw; returns (fit_avg, fit_tw, None, None)

    Returns
    -------
    fit_avg, fit_tw, full_avg, full_tw : four dicts with the same alignment keys
    """
    fit_avg = {a: averaged_data[a].copy() for a in alignments}
    fit_tw  = {a: trial_wise_data[a].copy() for a in alignments}

    for alignment in alignments:
        # fit arrays: drop timepoints where ANY *present* condition has a NaN average.
        # Entirely-missing conditions (all-NaN) are excluded so they don't mask all timepoints.
        avg = fit_avg[alignment]
        present = ~np.all(np.isnan(avg), axis=(0, 4))          # (c1,c2,c3): True = has data
        has_nan = np.any(np.isnan(avg), axis=0)                 # (c1,c2,c3,n_time)
        no_any_nan = ~np.any(has_nan & present[..., np.newaxis], axis=(0, 1, 2))

        fit_avg[alignment] = avg[..., no_any_nan]
        fit_tw[alignment]  = fit_tw[alignment][..., no_any_nan]

        # drop trial rows that are all-NaN in every *present* condition
        tw = fit_tw[alignment]
        present_tw  = ~np.all(np.isnan(tw), axis=(0, 1, 5))    # (c1,c2,c3)
        row_valid   = ~np.all(np.isnan(tw), axis=(1, 5))        # (n_trials,c1,c2,c3)
        nonnan_trial = np.any(row_valid & present_tw[np.newaxis], axis=(1, 2, 3))
        fit_tw[alignment] = fit_tw[alignment][nonnan_trial]

    if not compute_full:
        return fit_avg, fit_tw, None, None

    full_avg = {a: averaged_data[a].copy() for a in alignments}
    full_tw  = {a: trial_wise_data[a].copy() for a in alignments}

    for alignment in alignments:
        # full arrays: drop timepoints where ALL *present* conditions are NaN
        favg = full_avg[alignment]
        present_f = ~np.all(np.isnan(favg), axis=(0, 4))
        all_nan_f = np.all(np.isnan(favg), axis=0)
        no_all_nan = ~np.all(all_nan_f | ~present_f[..., np.newaxis], axis=(0, 1, 2))

        full_avg[alignment] = favg[..., no_all_nan]
        full_tw[alignment]  = full_tw[alignment][..., no_all_nan]

        ftw = full_tw[alignment]
        present_ftw  = ~np.all(np.isnan(ftw), axis=(0, 1, 5))
        row_valid_f  = ~np.all(np.isnan(ftw), axis=(1, 5))
        nonnan_trial_full = np.any(row_valid_f & present_ftw[np.newaxis], axis=(1, 2, 3))
        full_tw[alignment] = full_tw[alignment][nonnan_trial_full]

    return fit_avg, fit_tw, full_avg, full_tw


# ──────────────────────────────────────────────────────────────────────────────
# dPCA fitting
# ──────────────────────────────────────────────────────────────────────────────

def fit_dpca_on_alignment(
    fit_averaged,
    fit_trial_wise,
    n_components=3,
    marginalization_keys=('b', 's', 'c', 't'),
):
    """Fit a single dPCA model on one alignment's cleaned data.

    Parameters
    ----------
    fit_averaged         : ndarray (n_neurons, n_states, n_coh, n_choices, n_time) — no NaNs
    fit_trial_wise       : ndarray (n_trials, n_neurons, ..., n_time)
    n_components         : int, number of components per marginalization
    marginalization_keys : sequence of label chars, default ('b', 's', 'c', 't')

    Returns
    -------
    dpca : fitted model with .explained_variance_ratio_ set
    Z    : dict[marginalization_key] → transformed data on the fit data
    """
    dpca = dPCAlib.dPCA(
        n_components=n_components,
        labels=''.join(marginalization_keys),
        regularizer=0,
    )
    dpca.protect = ['t']
    dpca.fit(fit_averaged, fit_trial_wise)
    dpca, Z = dpca_transform(dpca, fit_averaged)
    return dpca, Z


def fit_dpca_all_alignments(
    fit_avg,
    fit_tw,
    alignments,
    n_components=3,
    marginalization_keys=('b', 's', 'c', 't'),
):
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


# ──────────────────────────────────────────────────────────────────────────────
# Significance analysis
# ──────────────────────────────────────────────────────────────────────────────

def _flat2d(A):
    return A.reshape((A.shape[0], -1))


def _classification(class_means, test, groups=None):
    """Nearest-centroid decoder: fraction of test points assigned to correct class.

    groups : list of lists, e.g. [[0,1],[2,3]] to collapse Q conditions into 2 classes.
             Each sub-list gives the condition indices belonging to that class.
             Centroids are computed as the mean of class_means over those indices.
             If None, each condition is its own class (standard Q-class decoder).
    """
    if groups is not None:
        # Subset to only the condition indices mentioned in groups.
        # Unlisted indices (e.g. 20% coh when groups=[[0,1],[3]]) are excluded from
        # the classifier so true_group length matches the number of evaluated conditions.
        all_idx = [idx for g in groups for idx in g]
        centroids = np.stack(
            [np.nanmean(class_means[g], axis=0) for g in groups]
        )                                                                     # (n_groups, T)
        true_group = np.array([gi for gi, g in enumerate(groups) for _ in g])  # (n_mentioned,)
        distances = np.abs(test[all_idx, None, :] - centroids[None, :, :])   # (n_mentioned, n_groups, T)
        nearest = np.argmin(distances, axis=1)                                # (n_mentioned, T)
        correct = nearest == true_group[:, None]                              # (n_mentioned, T)
        performance = correct.mean(axis=0).astype(float)
        performance[np.any(np.isnan(test[all_idx]), axis=0)] = np.nan
    else:
        distances = np.abs(test[:, None, :] - class_means[None, :, :])  # (Q, Q, T)
        nearest = np.argmin(distances, axis=1)                           # (Q, T)
        correct = nearest == np.arange(class_means.shape[0])[:, None]   # (Q, T)
        performance = correct.mean(axis=0).astype(float)
        performance[np.any(np.isnan(test), axis=0)] = np.nan
    return performance


def _denoise_mask(mask, n_consecutive):
    """Zero-out runs shorter than n_consecutive in an int32 mask (in-place)."""
    subseq = 0
    N = mask.shape[0]
    for n in range(N):
        if mask[n] == 1:
            subseq += 1
        else:
            if subseq < n_consecutive:
                for k in range(n - subseq, n):
                    mask[k] = 0
            subseq = 0
    return mask


def _dpca_train_test_split(dpca, X, trialX):
    """Leave-one-out train/validation split on the trial-wise data."""
    protect = dpca.protect
    n_unprotect = len(X.shape) - len(protect) if protect is not None else len(X.shape)
    n_protect   = len(protect) if protect is not None else 0

    protected = dpca._check_protected(trialX, protect)
    if ~protected:
        axes = [dpca.labels.index(ax) + 2 for ax in protect]
        trialX = dpca._roll_back(trialX, axes)
        X = np.squeeze(dpca._roll_back(X[None, ...], axes))

    N_samples = dpca._get_n_samples(trialX, protect=dpca.protect)
    idx = (np.random.rand(*N_samples.shape) * N_samples).astype(int)

    blindX = np.empty(trialX.shape[1:])
    it = np.nditer(np.empty(N_samples.shape), flags=['multi_index'])
    while not it.finished:
        blindX[it.multi_index + (np.s_[:],) * n_protect] = trialX[
            (idx[it.multi_index],) + it.multi_index + (np.s_[:],) * n_protect
        ]
        it.iternext()

    trainX = (
        X * (N_samples / (N_samples - 1))[(np.s_[:],) * n_unprotect + (None,) * n_protect]
        - np.where(np.isnan(blindX), X, blindX)
        / (N_samples - 1)[(np.s_[:],) * n_unprotect + (None,) * n_protect]
    )

    if ~protected:
        blindX = dpca._roll_back(blindX[..., None], axes, invert=True)[..., 0]
        trainX = dpca._roll_back(trainX[..., None], axes, invert=True)[..., 0]

    trainX -= np.nanmean(_flat2d(trainX), 1)[(np.s_[:],) + (None,) * (len(X.shape) - 1)]
    blindX -= np.nanmean(_flat2d(blindX), 1)[(np.s_[:],) + (None,) * (len(X.shape) - 1)]
    return trainX, blindX


def _dpca_train_test_split_fraction(X, trialX, train_fraction=0.8):
    """Fraction-based train/test split: average train trials, average held-out trials.

    For each (neuron, condition) cell independently, that neuron's non-NaN trials are
    split into a train set (train_fraction) and a test set (1 - train_fraction). Both
    are averaged, so the test centroid is far less noisy than the single-trial LOO test
    point. The per-neuron split guarantees every neuron contributes to both trainX and
    testX (as long as it has ≥ 2 valid trials for that condition), which is important
    for pseudo-populations where neurons from different sessions have different trial counts.

    Parameters
    ----------
    X              : trial-averaged data, shape (n_neurons, *cond_dims, n_time)
    trialX         : trial-wise data,     shape (n_trials, n_neurons, *cond_dims, n_time)
    train_fraction : fraction of trials used for training (default 0.8)

    Returns
    -------
    trainX, testX : both shape (n_neurons, *cond_dims, n_time), mean-centred
    """
    import itertools
    n_neurons  = X.shape[0]
    cond_shape = X.shape[1:-1]      # (*cond_dims,)  e.g. (n_b, n_s, n_c)
    trainX = np.full_like(X, np.nan)
    testX  = np.full_like(X, np.nan)

    for idx in itertools.product(*[range(d) for d in cond_shape]):
        for n in range(n_neurons):
            # trial data for this (neuron, condition) cell: (n_trials, n_time)
            sel = trialX[(slice(None), n) + idx + (slice(None),)]
            valid = ~np.all(np.isnan(sel), axis=1)
            vi = np.where(valid)[0]
            if len(vi) == 0:
                continue
            perm    = np.random.permutation(len(vi))
            # ensure at least 1 test trial when ≥ 2 valid trials exist
            n_train = min(max(1, int(round(len(vi) * train_fraction))), len(vi) - 1)
            tr, te  = vi[perm[:n_train]], vi[perm[n_train:]]
            dest    = (n,) + idx + (slice(None),)
            x_full  = X[dest]
            if tr.size:
                tr_mean = np.nanmean(sel[tr], axis=0)
                # fall back to full mean where all train trials are NaN at a timepoint
                # (e.g. late cue timepoints trimmed by short RT across all train trials)
                tr_nan = np.isnan(tr_mean)
                tr_mean[tr_nan] = x_full[tr_nan]
                trainX[dest] = tr_mean
            if te.size:
                testX[dest] = np.nanmean(sel[te], axis=0)  # NaN only if all test trials are NaN

    trainX -= np.nanmean(_flat2d(trainX), 1)[(np.s_[:],) + (None,) * (X.ndim - 1)]
    testX  -= np.nanmean(_flat2d(testX),  1)[(np.s_[:],) + (None,) * (X.ndim - 1)]
    return trainX, testX


def _compute_mean_score(dpca, X, trialX, n_splits, keys, key_groups=None, refit=True):
    """Run n_splits train/test splits and average classification scores.

    key_groups : dict[key → groups] passed to _classification for binary grouping,
                 e.g. {'s': [[0,1],[2,3]]} to test low vs high coherence.
    refit      : if False, skip dpca.fit() and use the already-fitted axes as-is.
    """
    K = X.shape[-1]
    if isinstance(dpca.n_components, int):
        scores = {key: np.empty((dpca.n_components, n_splits, K)) for key in keys}
    else:
        scores = {key: np.empty((dpca.n_components[key], n_splits, K)) for key in keys}

    for shuffle in range(n_splits):
        # trainX, validX = _dpca_train_test_split(dpca, X, trialX)  # LOO: noisy single-trial test
        trainX, validX = _dpca_train_test_split_fraction(X, trialX, train_fraction=0.8)
        if refit:
            dpca.fit(trainX)
        dpca, trainZ = dpca_transform(dpca, trainX)
        dpca, validZ = dpca_transform(dpca, validX)

        for key in keys:
            ncomps = dpca.n_components if isinstance(dpca.n_components, int) else dpca.n_components[key]
            axset = dpca.marginalizations[key]
            axset = axset if isinstance(axset, set) else set.union(*axset)
            # Always exclude the time axis so scores are resolved over time, not collapsed to a
            # scalar. Without this exclusion, time gets averaged and the score is broadcast to a
            # flat line across all K timepoints.
            time_axis = len(X.shape) - 2
            axes = set(range(len(X.shape) - 1)) - axset - {time_axis}
            for ax in list(axes)[::-1]:
                trainZ[key] = np.nanmean(trainZ[key], axis=ax + 1)
                validZ[key] = np.nanmean(validZ[key], axis=ax + 1)
            try:
                trainZ[key] = trainZ[key].reshape((ncomps, -1, K))
                validZ[key] = validZ[key].reshape((ncomps, -1, K))
            except ValueError:
                print(f"Error occurred while reshaping for key: {key}")

        for key in keys:
            ncomps = dpca.n_components if isinstance(dpca.n_components, int) else dpca.n_components[key]
            groups = key_groups.get(key) if key_groups else None
            for comp in range(ncomps):
                scores[key][comp, shuffle] = _classification(
                    trainZ[key][comp], validZ[key][comp], groups=groups
                )

    for key in keys:
        scores[key] = np.nanmean(scores[key], axis=1)
    return scores


def _shuffle_worker(dpca, trialX, n_splits, keys, key_groups, refit=True):
    """Single shuffle iteration for parallel execution."""
    import copy
    dpca = copy.deepcopy(dpca)
    trialX_s = dpca.shuffle_labels(trialX.copy())    # .copy() required: shuffle_labels writes in-place via Numba

    # Avoid np.nanmean, which allocates ~mask (a bool array the same size as trialX_s).
    # Instead: zero NaNs in-place, count non-NaN via (N - isnan.sum), then sum/count.
    nan_mask = np.isnan(trialX_s)                    # bool, 1/4 the size of float32 trialX_s
    trialX_s[nan_mask] = 0.0
    count = nan_mask.shape[0] - nan_mask.sum(axis=0) # no ~nan_mask temporary
    X_s = trialX_s.sum(axis=0, dtype=np.float32) / np.maximum(count, 1).astype(np.float32)
    trialX_s[nan_mask] = np.nan                      # restore NaN so split functions see valid rows correctly
    X_s[count == 0] = np.nan
    del count, nan_mask

    # Exclude entirely-absent conditions (all-NaN across neurons and time) from the
    # valid-timepoint check — same logic as clean_dpca_data.  Without this, missing
    # conditions (e.g. (target=0,choice=1) absent from correct-trial subsets) mark
    # every timepoint as NaN and every shuffle returns early with NaN scores.
    missing_cond = np.all(np.isnan(X_s), axis=(0, -1), keepdims=True)  # (1,*cond_dims,1)
    no_nan = ~np.any(np.isnan(X_s) & ~missing_cond, axis=tuple(range(X_s.ndim - 1)))
    del missing_cond
    K_orig = no_nan.shape[0]
    X_s = X_s[..., no_nan]
    if no_nan.all():
        trialX_trimmed = trialX_s                    # all timepoints valid: skip the extra copy
    else:
        trialX_trimmed = trialX_s[..., no_nan]
    del trialX_s                                     # free shuffled array before compute_mean_score

    # Conditions absent from this trial subset are excluded from the valid-timepoint check
    # above.  If no valid timepoints remain even after that (truly degenerate shuffle),
    # return NaN scores so nanquantile skips this shuffle when building the null.
    if X_s.shape[-1] == 0:
        ncomps = dpca.n_components if isinstance(dpca.n_components, int) else max(dpca.n_components.values())
        return {key: np.full((ncomps, K_orig), np.nan) for key in keys}, None

    dpca, Z = dpca_transform(dpca, X_s)
    score = _compute_mean_score(dpca, X_s, trialX_trimmed, n_splits, keys, key_groups=key_groups, refit=refit)
    return score, Z


def _smooth_score_array(arr, sigma):
    """Weighted NaN-aware Gaussian smooth for a (n_comp, T) score array."""
    from scipy.ndimage import gaussian_filter1d
    out = arr.copy()
    for k in range(arr.shape[0]):
        trace = arr[k]
        nan_mask = np.isnan(trace)
        if nan_mask.all():
            continue
        filled = trace.copy()
        filled[nan_mask] = 0.0
        numerator   = gaussian_filter1d(filled,                    sigma=sigma)
        denominator = gaussian_filter1d((~nan_mask).astype(float), sigma=sigma)
        sm = numerator / np.where(denominator > 0, denominator, np.nan)
        sm[nan_mask] = np.nan
        out[k] = sm
    return out


def dpca_significance_analysis(
    dpca, X, trialX,
    n_shuffles=100, n_splits=100, n_consecutive=1,
    full=False, keys=None, n_jobs=-1,
    key_groups=None, smooth_sigma=None, refit=True,
):
    """Compute significance masks using a nearest-centroid classifier with shuffled null.

    Parameters
    ----------
    dpca          : fitted dPCA model
    X             : trial-averaged data (no NaNs), shape (n_neurons, *cond_dims, n_time)
    trialX        : trial-wise data, shape (n_trials, n_neurons, *cond_dims, n_time)
    n_shuffles    : shuffles to build null distribution
    n_splits      : train/test splits per score estimate
    n_consecutive : min consecutive significant timepoints required (ignored when smooth_sigma set)
    full          : if True return (masks, true_score, shuffled_scores, shuffled_Z)
    keys          : marginalizations to test; defaults to all non-time keys
    n_jobs        : number of parallel jobs for shuffle loop (-1 = all cores)
    key_groups    : dict[key → list of lists] for binary grouping of conditions, e.g.
                    {'s': [[0,1],[2,3]]} classifies low (0%,6%) vs high (20%,50%) coherence
                    instead of one-of-four. Data and trial structure stay unchanged;
                    only the classification boundary changes.
    smooth_sigma  : float or None. If set, apply a Gaussian filter (sigma in timepoints) to
                    true_score along the time axis before thresholding against the null
                    quantile. Smoothing removes high-frequency noise from the score trace,
                    so the resulting mask reflects sustained significance rather than isolated
                    spikes. When None, falls back to the n_consecutive denoising approach.
    refit         : if False, skip dpca.fit() on each split and use the provided axes as-is.
                    Use this when projecting new data (e.g. error trials) through axes already
                    fitted on a different dataset (e.g. correct trials).

    Returns
    -------
    masks : dict[key] → (n_components, T) bool array
    true_score : dict[key] → (n_components, T) float array  — classifier accuracy on real data
                 (smoothed when smooth_sigma is set)
    scores : dict[key] → list of (n_components, T) float arrays  — null distribution per shuffle
    shuffled_transformed_data : list  — only returned when full=True
    """
    if dpca.opt_regularizer_flag:
        print("Regularization not optimized yet; starting optimization now.")
        dpca._optimize_regularization(X, trialX)

    all_keys = list(dpca.marginalizations.keys())
    time_key = dpca.labels[-1]
    if keys is None:
        keys = [k for k in all_keys if k != time_key]

    # Cast to float32 to halve per-worker memory and allow ~2× more parallel workers.
    X = X.astype(np.float32, copy=False)
    trialX = trialX.astype(np.float32)

    from joblib import Parallel, delayed
    import os as _os

    # Cap n_jobs so peak RAM stays within 50% of available memory.
    # Peak per worker = 3× trialX: (1) .copy() for shuffle_labels, (2) _replace_nan() copy
    # inside np.nanmean when NaNs are present, (3) boolean-index copy for NaN trimming.
    # Use 0.5 (not 0.8) because available RAM reported by psutil includes OS reclaimable
    # pages that are not reliably available to new worker processes under memory pressure.
    try:
        import psutil as _psutil
        _mem_avail = _psutil.virtual_memory().available
        _mem_per_worker = 3 * trialX.nbytes
        _max_safe = max(1, int(_mem_avail * 0.5 / _mem_per_worker)) if _mem_per_worker > 0 else n_jobs
        _n_cpu = _os.cpu_count() or 1
        _requested = (_n_cpu + 1 + n_jobs) if n_jobs < 0 else n_jobs
        _effective = min(_requested, _max_safe)
        if _effective < _requested:
            print(
                f"  Capping n_jobs to {_effective} to avoid OOM "
                f"(~{_mem_per_worker/1e9:.1f} GB/worker, {_mem_avail/1e9:.1f} GB free)",
                flush=True,
            )
        n_jobs = _effective
    except ImportError:
        pass

    print(f"Computing true score ({n_splits} splits)...", flush=True)
    true_score = _compute_mean_score(dpca, X, trialX, n_splits, keys, key_groups=key_groups, refit=refit)
    print("True score done.")

    if smooth_sigma is not None:
        for key in keys:
            true_score[key] = _smooth_score_array(true_score[key], smooth_sigma)

    print(f"Running {n_shuffles} shuffles ({n_jobs} parallel jobs)...", flush=True)
    shuffle_results = Parallel(n_jobs=n_jobs, verbose=10, max_nbytes='1M')(
        delayed(_shuffle_worker)(dpca, trialX, n_splits, keys, key_groups, refit)
        for _ in range(n_shuffles)
    )
    print("Shuffles done. Computing masks...")
    scores = {key: [r[0][key] for r in shuffle_results] for key in keys}
    shuffled_transformed_data = [r[1] for r in shuffle_results]

    if smooth_sigma is not None:
        scores = {
            key: [_smooth_score_array(s, smooth_sigma) for s in scores[key]]
            for key in keys
        }

    masks = {key: np.full(true_score[key].shape, False) for key in keys}
    for key in keys:
        min_len = min(s.shape[1] for s in scores[key])
        quantile_score = np.nanquantile(
            np.dstack([s[:, :min_len] for s in scores[key]]), 0.95, axis=-1
        )
        masks[key][:, :min_len] = true_score[key][:, :min_len] >= quantile_score

    if smooth_sigma is None and n_consecutive > 1:
        # Legacy denoising: zero out runs shorter than n_consecutive in the binary mask.
        # Prefer smooth_sigma for continuous smoothing of the score trace before thresholding.
        for key in keys:
            for k in range(masks[key].shape[0]):
                masks[key][k] = _denoise_mask(masks[key][k].astype(np.int32), n_consecutive)

    if full:
        return masks, true_score, scores, shuffled_transformed_data
    return masks, true_score, scores
