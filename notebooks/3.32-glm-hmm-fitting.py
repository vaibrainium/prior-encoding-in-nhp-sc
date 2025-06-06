#!/usr/bin/env python
# coding: utf-8

# In[47]:
from pathlib import Path
import numpy as np
import numpy.random as npr
import pandas as pd
from sklearn import preprocessing

from notebooks.imports import *
from config import dir_config, main_config
from src.utils.glm_hmm_utils import *
import pickle
import copy


# In[ ]:


compiled_dir = Path(dir_config.data.compiled)
processed_dir = Path(dir_config.data.processed)


# In[ ]:
session_metadata = pd.read_csv(processed_dir / 'sessions_metadata.csv')

def extract_previous_data(trial_data, valid_idx, first_trial, n_trial_back=3):
	np.random.seed(1)

	# remove trials before the first valid trial or first three trials, whichever is larger
	prev_choice = np.empty((len(trial_data) - first_trial, n_trial_back), dtype=int)
	prev_target = np.empty((len(trial_data) - first_trial, n_trial_back), dtype=int)

	# Loop to populate prev_choice and prev_target with the last n_trial_back values for each trial
	for i in range(first_trial, len(trial_data)):
		# Get the valid indices for the last n_trial_back trials
		valid_indices = valid_idx[valid_idx < i][-n_trial_back:]  # Ensure we get the last n_trial_back valid trials

		prev_choice[i - first_trial] = trial_data.choice[valid_indices] * 2 - 1  # Convert choice to -1/1
		prev_target[i - first_trial] = trial_data.target[valid_indices] * 2 - 1  # Convert target to -1/1


	return prev_choice, prev_target

def prepare_input_data(data, input_dim, valid_idx, first_trial):
	if "no_bias" in _TRIALS:
		current_trial_param = 1
	else:
		current_trial_param = 2
	n_trial_back=(input_dim - current_trial_param) // 2
	X = np.ones((1, data.shape[0] - first_trial, input_dim))

	current_stimulus = data.coherence * (2 * data.target - 1)
	current_stimulus = current_stimulus / 100

	X[0, :, 0] = current_stimulus[first_trial:]  # current stimulus
	X[0, :, current_trial_param:current_trial_param+n_trial_back], X[0, :, current_trial_param+n_trial_back: current_trial_param+2*n_trial_back] = extract_previous_data(data, valid_idx, first_trial, n_trial_back=n_trial_back)
	return list(X)


# In[ ]:



for _TRIALS in ["all_trials", "all_trials_no_bias", "all_trials_eq_prior"]:
	n_states = 2  # number of discrete states
	obs_dim = 1  # number of observed dimensions: choice(toRF/awayRF)
	num_categories = 2  # number of categories for output

	if "no_bias" in _TRIALS:
		current_trial_param = 1
	else:
		current_trial_param = 2

	n_trial_back = 1

	input_dim = current_trial_param + 2*n_trial_back  # input dimensions: current signed coherence, 1(bias), previous choice(toRF/awayRF), previous target side(toRF/awayRF)

	# Pre-allocate lists for session data
	inputs_session_wise = []
	choices_session_wise = []
	invalid_idx_session_wise = []
	masks_session_wise = []
	GP_trial_num_session_wise = []
	prob_toRF_session_wise = []

	# Pre-build a mapping from session_id to prior_direction for efficient lookup
	prior_direction_map = session_metadata.set_index("session_id")["prior_direction"].to_dict()

	# Process each session
	for session_id in session_metadata["session_id"]:

		# Read trial data for each session
		trial_data = pd.read_csv(Path(compiled_dir, session_id, f"{session_id}_trial.csv"), index_col=None)
		GP_trial_data = trial_data[trial_data.task_type == 1].reset_index()

		if "eq_prior" in _TRIALS:
			GP_trial_data = GP_trial_data[GP_trial_data.prob_toRF == 50].reset_index()

		# Fill missing values for important columns
		GP_trial_data['choice'] = GP_trial_data.choice.fillna(-1)
		GP_trial_data['target'] = GP_trial_data.target.fillna(-1)
		GP_trial_data['outcome'] = GP_trial_data.outcome.fillna(-1)

		# Get valid indices based on outcomes
		valid_idx = np.where(GP_trial_data.outcome >= 0)[0]

		# First valid trial considering n_trial_back
		first_trial = valid_idx[n_trial_back - 1] + 1

		# Prepare inputs and choices
		inputs = prepare_input_data(GP_trial_data, input_dim, valid_idx, first_trial)
		choices = GP_trial_data.choice.values.reshape(-1, 1).astype("int")
		choices = choices[first_trial:]

		# Adjust invalid_idx and prepare mask
		invalid_idx = np.where(choices == -1)[0]

		if "all_trials" in _TRIALS:
			# For training, replace -1 with a random sample from 0,1
			choices[choices == -1] = np.random.choice(2, invalid_idx.shape[0])

			# Prepare mask
			mask = np.ones_like(choices, dtype=bool)
			mask[invalid_idx] = 0

			# Get trial numbers and prob_toRF for the cropped session
			GP_trial_num = np.array(GP_trial_data.trial_number)[first_trial:]
			prob_toRF = np.array(GP_trial_data.prob_toRF)[first_trial:]
		else:
			assert "all_trials" in _TRIALS, "Invalid trials option"

		# Check prior_direction for the current session and adjust inputs and choices
		prior_direction = prior_direction_map.get(session_id, 'awayRF')
		if prior_direction == 'awayRF':
			inputs[0][:, 0] = -inputs[0][:, 0]  # Flip the direction for input features
			inputs[0][:, 2:] = -inputs[0][:, 2:]
			choices = 1-choices  # Flip the choices

		assert len(choices) == len(inputs[0]), f"Length mismatch: {len(choices)} vs {len(inputs[0])}"
		assert len(mask) == len(inputs[0]), f"Length mismatch: {len(mask)} vs {len(inputs[0])}"
		assert len(GP_trial_num) == len(inputs[0]), f"Length mismatch: {len(GP_trial_num)} vs {len(inputs[0])}"
		assert len(prob_toRF) == len(inputs[0]), f"Length mismatch: {len(prob_toRF)} vs {len(inputs[0])}"


		# Append session-wise data to corresponding lists
		masks_session_wise.append(mask)
		inputs_session_wise += inputs
		choices_session_wise.append(choices)
		GP_trial_num_session_wise.append(GP_trial_num)
		prob_toRF_session_wise.append(prob_toRF)


	# In[ ]:


	unnormalized_inputs_session_wise = copy.deepcopy(inputs_session_wise)
	# scaling all input variables
	for idx_session in range(len(session_metadata)):
		mask = masks_session_wise[idx_session][:, 0]
		inputs_session_wise[idx_session][mask, 0] = preprocessing.scale(inputs_session_wise[idx_session][mask, 0], axis=0)
		inputs_session_wise[idx_session][mask, 2:] = preprocessing.scale(inputs_session_wise[idx_session][mask, 2:], axis=0)


	# In[ ]:


	models_glm_hmm, fit_lls_glm_hmm = global_fit(choices_session_wise, inputs_session_wise, state_range=np.arange(1, 6), masks=masks_session_wise, n_iters=2500, n_initializations=20)


	# In[ ]:
	# get best model of 20 initializations for each state
	init_params = {"glm_weights": {}, "transition_matrices": {}}
	for n_states in np.arange(1, 6):
		best_idx = fit_lls_glm_hmm[n_states].index(max(fit_lls_glm_hmm[n_states]))
		init_params["glm_weights"][n_states] = models_glm_hmm[n_states][best_idx].observations.params
		init_params["transition_matrices"][n_states] = models_glm_hmm[n_states][best_idx].transitions.params


	# In[ ]:


	# session-wise fitting with 5 fold cross-validation
	models_session_state_fold, train_ll_session, test_ll_session = session_wise_fit_cv(choices_session_wise, inputs_session_wise, masks=masks_session_wise, n_sessions=len((session_metadata["session_id"])), init_params=init_params, state_range=np.arange(1, 6), n_iters=1000)


	# store data and models for aggregated
	global_fits = {"models": models_glm_hmm, "fits_lls_glm_hmm": fit_lls_glm_hmm, "init_params": init_params}
	session_wise_fits = {
		"models": models_session_state_fold,
		"train_ll": train_ll_session,
		"test_ll": test_ll_session,
	}
	# store data and models for session-wise
	session_data = {}
	for idx, session_id in enumerate(session_metadata["session_id"]):
		inputs = inputs_session_wise[idx]
		df = {
			"choices": choices_session_wise[idx].ravel(),
			"stimulus": unnormalized_inputs_session_wise[idx][:, 0],
			"normalized_stimulus": inputs[:, 0],
			"bias": inputs[:, 1],
			"mask": masks_session_wise[idx].ravel(),
			"trial_num": GP_trial_num_session_wise[idx].ravel(),
			"prob_toRF": prob_toRF_session_wise[idx].ravel(),
		}


		for t in range(n_trial_back):
			df[f"prev_choice_{t+1}"] = inputs[:, current_trial_param + t]
			df[f"prev_target_{t+1}"] = inputs[:, current_trial_param + n_trial_back + t]

		session_data[session_id] = pd.DataFrame(df)


	models_and_data = {
		"global": global_fits,
		"session_wise":session_wise_fits,
		"data": session_data,
	}

	with open(Path(processed_dir, f"global_glm_hmm_{_TRIALS}.pkl"), "wb") as f:
		pickle.dump(models_and_data, f)
