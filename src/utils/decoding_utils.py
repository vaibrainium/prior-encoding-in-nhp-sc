"""Self-contained helpers for pseudo-population decoding (notebook 8.20).

This module is intentionally standalone: it does NOT import the decoding helpers
from ``ephys_utils`` (those don't exist on this branch and we avoid editing shared
files). Everything the decoding notebook needs to turn raw ``glm_hmm`` + neuron-wise
ephys into per-session, per-alignment spike tensors + labels lives here.

Design follows Zhang et al. (bioRxiv 2025.12.31.697231) §5.7-5.9:
  - pseudo-populations built by stacking sessions neuron-wise,
  - per-session 80/20 real-trial split, then bootstrap to a fixed count per class,
  - a single BatchNorm -> Dropout -> Linear decoder per (variable, alignment, time bin).

The sampling / model / decode loop themselves live in the notebook (analysis code).
Here we only provide the data-preparation primitives.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import dpca_utils
from config import dir_config

compiled_dir = Path(dir_config.data.compiled)
processed_dir = Path(dir_config.data.processed)


def prepare_trial_info(session_metadata, glm_hmm):
    """Build a tidy per-trial info table from the fitted GLM-HMM.

    Runs the HMM state inference (biased vs unbiased) and joins in behavioural
    columns. Sign/choice are put in the toRF reference frame by
    ``dpca_utils.extract_hmm_state_trial_info`` (which flips awayRF sessions), so
    ``choice == 1`` means **toRF** and ``choice == 0`` means **awayRF** for every
    session. The notebook asserts this against the prior bias as a guard.

    This function is idempotent: it deep-copies ``glm_hmm["data"]`` before the
    in-place awayRF sign flip, so re-running it does not double-flip.

    Returns
    -------
    trial_info : DataFrame with columns
        session_id, prior_direction ("toRF"/"awayRF"), trial_num, signed_coherence,
        hmm_state (1=biased, 0=unbiased), prior_block (1=biased block, 0=equal block),
        target, choice (1=toRF), reaction_time, outcome
    """
    glm_hmm_data = copy.deepcopy(glm_hmm["data"])  # mutated in-place by the flip; keep original clean
    biased_state_trial_info, unbiased_state_trial_info, _ = \
        dpca_utils.extract_hmm_state_trial_info(
            session_metadata, glm_hmm, glm_hmm_data, compiled_dir=compiled_dir
        )

    assert biased_state_trial_info.keys() == unbiased_state_trial_info.keys(), \
        f"Session key mismatch: {biased_state_trial_info.keys() ^ unbiased_state_trial_info.keys()}"

    frames = []
    for session_id in biased_state_trial_info:
        biased_df = biased_state_trial_info[session_id].copy()
        unbiased_df = unbiased_state_trial_info[session_id].copy()
        biased_df["hmm_state"] = 1
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
            on="trial_num", how="left",
        )
        session_df = session_df.sort_values("trial_num").reset_index(drop=True)
        session_df.insert(0, "session_id", session_id)
        frames.append(session_df)

    trial_info = pd.concat(frames, ignore_index=True)
    trial_info = trial_info[
        ["session_id", "trial_num", "stimulus", "hmm_state",
         "prob_toRF", "target", "choices", "reaction_time", "outcome"]
    ].rename(columns={
        "stimulus": "signed_coherence",
        "prob_toRF": "prior_block",
        "choices": "choice",
    })
    trial_info["prior_block"] = (trial_info["prior_block"] != 50).astype(int)
    trial_info.insert(
        1, "prior_direction",
        trial_info.session_id.map(session_metadata.set_index("session_id")["prior_direction"]),
    )
    return trial_info


def get_session_trial_data(
    sessions,
    trial_info,
    neuron_metadata,
    ephys_data,
    alignments,
    spike_type="convolved_spike_trains",
):
    """Extract per-session spike tensors + decode labels, aligned to trial_info.

    Each session independently contributes spike trains for all of its valid trials
    (trials present both in ``trial_info`` and in this session's ephys). Intended for
    pseudo-population decoding where sessions are stacked neuron-wise and each session
    independently samples trials per label value.

    Parameters
    ----------
    spike_type : which ephys array to pull. Default ``"convolved_spike_trains"``
                 (smoothed firing rate); pass ``"spike_trains"`` for raw counts.

    Returns
    -------
    session_data : {alignment: {session_id: {
                        "spikes"       : (n_trials, n_session_neurons, n_timebins),
                        "abs_coherence": (n_trials,),
                        "choice"       : (n_trials,),
                        "hmm_state"    : (n_trials,),
                   }}}
    neuron_positions : {session_id: ndarray of global column indices into sorted neuron_ids}
    n_total_neurons  : int
    label_values     : {"abs_coherence": ndarray, "choice": ndarray, "hmm_state": ndarray}
    """
    neuron_ids = dpca_utils.get_neuron_ids(neuron_metadata, sessions)
    n_total_neurons = len(neuron_ids)

    neuron_positions = {}
    for session_id in sessions:
        session_neuron_ids = neuron_metadata.loc[neuron_metadata.session_id == session_id, "neuron_id"].values
        session_neuron_ids = session_neuron_ids[np.isin(session_neuron_ids, neuron_ids)]
        if len(session_neuron_ids):
            neuron_positions[session_id] = np.searchsorted(neuron_ids, session_neuron_ids)

    label_values = {
        "abs_coherence":    np.sort(trial_info.signed_coherence.abs().unique()),
        "difficulty_level": np.array([0, 1]),   # 1=hard (|coh|<0.2: {0,0.06}) / 0=easy (|coh|>=0.2: {0.2,0.5})
        "choice":           np.sort(trial_info.choice.unique()),
        "hmm_state":        np.sort(trial_info.hmm_state.unique()),
    }

    session_data = {}
    for alignment in alignments:
        session_data[alignment] = {}
        for session_id in sessions:
            if session_id not in neuron_positions:
                continue
            session_trial_info = trial_info[trial_info.session_id == session_id].reset_index(drop=True)
            if len(session_trial_info) == 0:
                continue
            session_neuron_ids = neuron_metadata.loc[neuron_metadata.session_id == session_id, "neuron_id"].values
            session_neuron_ids = session_neuron_ids[np.isin(session_neuron_ids, neuron_ids)]

            # map wanted trial_nums to rows in this session's ephys (shared trial order across neurons)
            ref_trials = np.asarray(ephys_data[alignment][session_neuron_ids[0]]["trial_number"])
            wanted = session_trial_info.trial_num.values
            sorter = np.argsort(ref_trials)
            pos = np.searchsorted(ref_trials[sorter], wanted)
            pos = np.clip(pos, 0, len(sorter) - 1)
            row_idx = sorter[pos]
            valid = ref_trials[row_idx] == wanted          # drop trials with no ephys row
            row_idx = row_idx[valid]
            session_trial_info = session_trial_info[valid].reset_index(drop=True)
            if len(row_idx) == 0:
                continue

            spikes = np.stack([
                np.asarray(ephys_data[alignment][neuron_id][spike_type])[row_idx]
                for neuron_id in session_neuron_ids
            ]).transpose(1, 0, 2)   # (n_trials, n_session_neurons, n_timebins)

            session_data[alignment][session_id] = {
                "spikes":            spikes,
                "abs_coherence":     np.abs(session_trial_info.signed_coherence.values),
                "difficulty_level":  (np.abs(session_trial_info.signed_coherence.values) < 0.2).astype(int),
                "choice":            session_trial_info.choice.values,
                "hmm_state":         session_trial_info.hmm_state.values,
            }

    return session_data, neuron_positions, n_total_neurons, label_values
