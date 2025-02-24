import numpy as np
from sklearn.preprocessing import StandardScaler

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
    if coherence == 0 or outcome is None:
        idx = (np.abs(trial_data.stimulus) == coherence) & (trial_data.choices == choice)
    elif outcome == 1:
        idx = (np.abs(trial_data.stimulus) == coherence) & (trial_data.choices == choice) & (trial_data["stimulus"] * (trial_data["choices"]*2-1) > 0)
    elif outcome == 0:
        idx = (np.abs(trial_data.stimulus) == coherence) & (trial_data.choices == choice) & (trial_data["stimulus"] * (trial_data["choices"]*2-1) < 0)
    return np.array(trial_data["trial_num"][idx].values.reshape(-1, 1))

def get_neural_data_from_trial_num(neuronal_data, trial_num, type = "convolved_spike_trains"):
    index = np.where(np.isin(np.array(neuronal_data["trial_number"]), trial_num))[0]
    if type == "spike_trains":
        return np.array(neuronal_data["spike_trains"][index])
    elif type == "convolved_spike_trains":
        return np.array(neuronal_data["convolved_spike_trains"][index])