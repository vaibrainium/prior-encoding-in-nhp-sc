import numpy as np

# def extract_neuronal_data(alignment, neuron_metadata, state_occupancy, data_type="convolved_spike_trains"):
#     neuronal_data = {
#         'biased_state': {},
#         'unbiased_state': {},
#     }

#     for idx, neuron_id in enumerate(neuron_metadata.neuron_id):
#         session_id = neuron_metadata.session_id[idx]
#         biased_trials = np.array(state_occupancy[session_id]["biased_state_trials"])
#         biased_idx = np.where(np.isin(np.array(ephys[alignment][neuron_id]["trial_number"]), biased_trials))[0]
#         unbiased_trials = np.array(state_occupancy[session_id]["unbiased_state_trials"])
#         unbiased_idx = np.where(np.isin(np.array(ephys[alignment][neuron_id]["trial_number"]), unbiased_trials))[0]

#         neuronal_data['biased_state'][neuron_id] = np.array(ephys[alignment][neuron_id][data_type][biased_idx])
#         neuronal_data['unbiased_state'][neuron_id] = np.array(ephys[alignment][neuron_id][data_type][unbiased_idx])

#     return neuronal_data


def get_trial_num(trial_data, coherence, choice, outcome=None):
    trial_data = trial_data[~np.isnan(trial_data.reaction_time)]
    if coherence == 0 or outcome is None:
        idx = (np.abs(trial_data.stimulus) == coherence) & (trial_data.choices == choice)
    elif outcome == 1:
        idx = (np.abs(trial_data.stimulus) == coherence) & (trial_data.choices == choice) & (trial_data["stimulus"] * (trial_data["choices"] * 2 - 1) > 0)
    elif outcome == 0:
        idx = (np.abs(trial_data.stimulus) == coherence) & (trial_data.choices == choice) & (trial_data["stimulus"] * (trial_data["choices"] * 2 - 1) < 0)
    return np.array(trial_data["trial_num"][idx].values.reshape(-1, 1))


def get_neural_data_from_trial_num(neuronal_data, trial_num, type="convolved_spike_trains"):
    index = np.where(np.isin(np.array(neuronal_data["trial_number"]), trial_num))[0]
    if type == "spike_trains":
        return np.array(neuronal_data["spike_trains"][index])
    elif type == "convolved_spike_trains":
        return np.array(neuronal_data["convolved_spike_trains"][index])
    raise ValueError(f"Unknown type: {type!r}")


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
    else:
        exclude_cell_types = exclude_cell_types.append("trash") if "trash" not in exclude_cell_types else exclude_cell_types
    mask = neuron_metadata.session_id.isin(sessions)
    mask &= ~neuron_metadata.classification.isin(exclude_cell_types)
    neuron_ids = neuron_metadata.neuron_id[mask].values

    if leave_out_fraction is not None and leave_out_fraction > 0:
        if rng is None:
            rng = np.random.default_rng()
        n_keep = int(np.round(len(neuron_ids) * (1 - leave_out_fraction)))
        neuron_ids = rng.choice(neuron_ids, size=n_keep, replace=False)

    return np.sort(neuron_ids)


def get_windowed_spike_count(neuronal_data, trial_nums, t_start_idx, t_end_idx):
    """Sum spike_trains over [t_start_idx, t_end_idx) for given trials. Returns (n_trials,).
    Preserves duplicates in trial_nums (bootstrap with replacement)."""
    trial_number = np.array(neuronal_data["trial_number"])
    spike_trains = np.array(neuronal_data["spike_trains"])
    sorter = np.argsort(trial_number)
    positions = np.searchsorted(trial_number[sorter], trial_nums)
    positions = np.clip(positions, 0, len(sorter) - 1)
    idx = sorter[positions]
    return np.nansum(spike_trains[idx, t_start_idx:t_end_idx], axis=1)
