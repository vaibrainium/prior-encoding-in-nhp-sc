from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import dpca_utils, ephys_utils
from config import dir_config
from dataclasses import dataclass
from typing import Callable

compiled_dir  = Path(dir_config.data.compiled)
processed_dir = Path(dir_config.data.processed)

@dataclass
class Condition:
    column: str
    values: np.ndarray
    transform: Callable | None = None

def prepare_trial_info(session_metadata, glm_hmm):
    data = glm_hmm["data"]
    biased_state_trial_info, unbiased_state_trial_info, state_occupancy = \
        dpca_utils.extract_hmm_state_trial_info(session_metadata, glm_hmm, data,
                                                compiled_dir=compiled_dir)

    assert biased_state_trial_info.keys() == unbiased_state_trial_info.keys(), \
        f"Session key mismatch: {biased_state_trial_info.keys() ^ unbiased_state_trial_info.keys()}"

    frames = []
    for session_id in biased_state_trial_info:
        biased_df   = biased_state_trial_info[session_id].copy()
        unbiased_df = unbiased_state_trial_info[session_id].copy()
        biased_df["hmm_state"]   = 1
        unbiased_df["hmm_state"] = 0
        session_df = pd.concat([biased_df, unbiased_df], ignore_index=True)
        session_df = session_df[session_df.reaction_time.notna()]

        session_trial_data = pd.read_csv(
            compiled_dir / session_id / f"{session_id}_trial_cleaned.csv", index_col=None
        )
        session_trial_data = session_trial_data.rename(columns={"trial_number": "trial_num"})
        session_df = pd.merge(
            session_df,
            session_trial_data[["trial_num", "target", "outcome"]],
            on="trial_num", how="left"
        )
        session_df = session_df.sort_values("trial_num").reset_index(drop=True)
        session_df.insert(0, "session_id", session_id)
        frames.append(session_df)

    trial_info = pd.concat(frames, ignore_index=True)
    trial_info = trial_info[
        ["session_id", "trial_num", "stimulus", "hmm_state",
         "prob_toRF", "target", "choices", "reaction_time", "outcome"]
    ].rename(columns={
        "stimulus":  "signed_coherence",
        "prob_toRF": "prior_block",
        "choices":   "choice",
    })
    trial_info["prior_block"] = (trial_info["prior_block"] != 50).astype(int)
    return trial_info

def get_label_pseudo_trials(trial_data_event: np.ndarray, label_axis: int, label_idx: int) -> np.ndarray:
    """Extract valid pseudo-trials for one label class from a trial_data tensor.

    Slices trial_data_event along label_axis at label_idx, flattens the remaining
    condition dims into the trial dim, and drops rows where ANY neuron is NaN.

    Parameters
    ----------
    trial_data_event : (max_trials, n_neurons, *cond_dims, n_timebins)
    label_axis       : axis in trial_data_event corresponding to the decode label
    label_idx        : integer index of this label class along label_axis

    Returns
    -------
    X_valid : (n_valid, n_neurons, n_timebins)
    """
    idx = [slice(None)] * trial_data_event.ndim
    idx[label_axis] = label_idx
    X = trial_data_event[tuple(idx)]                  # (max_trials, n_neurons, *other_conds, n_timebins)
    n_trials, n_neurons, n_timebins = X.shape[0], X.shape[1], X.shape[-1]
    X = X.reshape(n_trials, n_neurons, -1, n_timebins)  # fold other cond dims
    X = X.transpose(0, 2, 1, 3).reshape(-1, n_neurons, n_timebins)  # (n_trials*n_other, n_neurons, n_timebins)
    valid = ~np.isnan(X[:, :, 0]).any(axis=1)         # drop rows with any NaN neuron
    return X[valid]


def get_session_trial_data(sessions, trial_info, neuron_metadata, ephys_data, alignments):
    """Extract per-session spike trains aligned to trial_info labels.

    Each session independently contributes spike trains for all its valid trials.
    Intended for pseudo-population decoding where sessions are stacked neuron-wise
    and each session independently samples trials per label value — avoiding the
    NaN sparsity of condition-indexed tensors.

    Returns
    -------
    session_data : {event: {session_id: {
                        "spikes"    : (n_trials, n_session_neurons, n_timebins),
                        "coherence" : (n_trials,),
                        "choice"    : (n_trials,),
                        "hmm_state" : (n_trials,),
                    }}}
    neuron_positions : {session_id: ndarray}  -- global indices into sorted neuron_ids
    n_total_neurons  : int
    label_values     : {"coherence": ndarray, "choice": ndarray, "hmm_state": ndarray}
    """
    neuron_ids      = ephys_utils.get_neuron_ids(neuron_metadata, sessions)
    n_total_neurons = len(neuron_ids)

    neuron_positions = {}
    for session_id in sessions:
        s_nids = neuron_metadata.loc[neuron_metadata.session_id == session_id, "neuron_id"].values
        s_nids = s_nids[np.isin(s_nids, neuron_ids)]
        if len(s_nids):
            neuron_positions[session_id] = np.searchsorted(neuron_ids, s_nids)

    label_values = {
        "coherence": np.sort(trial_info.signed_coherence.unique()),
        "choice":    np.sort(trial_info.choice.unique()),
        "hmm_state": np.sort(trial_info.hmm_state.unique()),
    }

    session_data = {}
    for event in alignments:
        session_data[event] = {}
        for session_id in sessions:
            if session_id not in neuron_positions:
                continue
            session_ti = trial_info[trial_info.session_id == session_id].reset_index(drop=True)
            if len(session_ti) == 0:
                continue
            s_nids = neuron_metadata.loc[neuron_metadata.session_id == session_id, "neuron_id"].values
            s_nids = s_nids[np.isin(s_nids, neuron_ids)]

            ref_trials = np.array(ephys_data[event][s_nids[0]]["trial_number"])
            wanted     = session_ti.trial_num.values
            sorter     = np.argsort(ref_trials)
            pos        = np.searchsorted(ref_trials[sorter], wanted)
            pos        = np.clip(pos, 0, len(sorter) - 1)
            row_idx    = sorter[pos]
            valid      = ref_trials[row_idx] == wanted
            row_idx    = row_idx[valid]
            session_ti = session_ti[valid].reset_index(drop=True)

            if len(row_idx) == 0:
                continue

            spikes = np.stack([
                np.array(ephys_data[event][nid]["convolved_spike_trains"])[row_idx]
                for nid in s_nids
            ]).transpose(1, 0, 2)   # (n_trials, n_session_neurons, n_timebins)

            session_data[event][session_id] = {
                "spikes":    spikes,
                "coherence": session_ti.signed_coherence.values,
                "choice":    session_ti.choice.values,
                "hmm_state": session_ti.hmm_state.values,
            }

    return session_data, neuron_positions, n_total_neurons, label_values


def get_binned_counts(X_valid: np.ndarray, sample_idx: np.ndarray, t0: int, t1: int) -> np.ndarray:
    """Sum spike counts in [t0, t1) for selected pseudo-trials.

    Parameters
    ----------
    X_valid    : (n_valid, n_neurons, n_timebins)
    sample_idx : (n_samples,) integer indices into X_valid (may repeat for bootstrap)
    t0, t1     : timebin slice (index units, not ms)

    Returns
    -------
    counts : (n_samples, n_neurons)
    """
    return X_valid[sample_idx, :, t0:t1].sum(axis=-1)


def get_condition_trial_nums(trial_data, coherence=None, choice=None, target=None, outcome=None):
    trial_mask = np.ones(len(trial_data), dtype=bool)
    if coherence is not None:
        trial_mask = trial_mask & (trial_data.signed_coherence == coherence)
    if choice is not None:
        trial_mask = trial_mask & (trial_data.choice == choice)
    if target is not None:
        trial_mask = trial_mask & (trial_data.target == target)
    if outcome is not None:
        trial_mask = trial_mask & (trial_data.outcome == outcome)
    selected_trials = trial_data[trial_mask]
    return np.array(selected_trials.trial_num.values.reshape(-1, 1))
