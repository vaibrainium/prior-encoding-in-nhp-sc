import numpy as np
import numpy.random as npr
import ssm
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from tqdm import tqdm


def split_non_continuous(arr):
    arr = np.array(arr)
    # Find where the difference between consecutive elements is not 1
    breaks = np.where(np.diff(arr) != 1)[0] + 1
    # Use np.split to divide the array at those indices
    return np.split(arr, breaks)

def cross_validation_split(session_length, idx_split=0, n_sub_block = 4, k_folds=5):
	assert 0 <= idx_split < 5, "Number of splits must be between [0,5)!"
	block_length = session_length // n_sub_block // k_folds

	test_idx = []
	for i in range(n_sub_block):
		if i == n_sub_block - 1 & idx_split == k_folds - 1:
			end_idx = session_length
			start_idx = (i * k_folds + idx_split) * block_length
		else:
			end_idx = (i * k_folds + idx_split + 1) * block_length
			start_idx = (i * k_folds + idx_split) * block_length
		test_idx.append(np.arange(start_idx, end_idx))

	train_idx = np.setdiff1d(np.arange(session_length), np.concatenate(test_idx))
	train_idx = split_non_continuous(train_idx)
	return train_idx, test_idx


def session_wise_fit_cv(observations, inputs, masks, n_sessions, init_params, k_folds=5, state_range=np.arange(2, 6), fitting_method="em", n_iters=200, tolerance=10**-4, n_jobs=-1):
	"""
	Optimized version of session-wise GLM-HMM fitting with k-fold cross-validation using parallelization.
	"""
	masks = [np.ones_like(arr) for arr in observations] if masks is None else masks
	assert len(observations) == n_sessions, "Observations are not compatible with number of sessions!"
	assert len(inputs) == n_sessions, "Inputs are not compatible with number of sessions!"
	assert len(masks) == n_sessions, "Masks are not compatible with number of sessions!"
	assert "transition_matrices" in init_params.keys() and "glm_weights" in init_params.keys(), "Initial parameters not provided correctly!"

	def fit_model_on_fold(idx_split, glm_hmm, idx_session):
		"""
		Fit a GLM-HMM on one fold and compute training and testing log-likelihoods.
		"""
		session_length = observations[idx_session].shape[0]
		train_idx, test_idx = cross_validation_split(session_length, idx_split, k_folds=k_folds)
		train_obs = [observations[idx_session][train] for train in tran_idx]
		test_obs = [observations[idx_session][test] for test in test_idx]
		train_masks = [masks[idx_session][train] for train in train_idx]
		test_masks = [masks[idx_session][test] for test in test_idx]
		train_inputs = [inputs[idx_session][train] for train in train_idx]
		test_inputs = [inputs[idx_session][test] for test in test_idx]
		# Fit the model on the training data
		train_ll = glm_hmm.fit(train_obs, inputs=train_inputs, masks=train_masks, method=fitting_method, num_iters=n_iters, initialize=False, tolerance=tolerance)
		test_ll = glm_hmm.log_likelihood(test_obs, inputs=test_inputs, masks=test_masks)
		return glm_hmm, train_ll, test_ll

	def process_session_state_fold(idx_session, n_states):
		"""
		Fit a GLM-HMM for a specific session and state with cross-validation.
		"""
		glm_hmm = ssm.HMM(n_states, observations[0].shape[1], inputs[0].shape[1], observations="input_driven_obs", observation_kwargs=dict(C=len(np.unique(observations[0]))), transitions="standard")
		glm_hmm.observations.params = init_params["glm_weights"][n_states]
		glm_hmm.transitions.params = init_params["transition_matrices"][n_states]

		# Collecting results across all folds
		results = Parallel(n_jobs=n_jobs)(delayed(fit_model_on_fold)(idx_split, glm_hmm, idx_session) for idx_split in np.arange(k_folds))
		# Unzip results and ensure consistent shape for log-likelihoods
		models, train_lls, test_lls = zip(*results)

		# Convert train_lls and test_lls into lists of lists to handle varying lengths
		train_ll_list = [ll for ll in train_lls]  # List of lists for train log-likelihoods
		test_ll_list = [ll for ll in test_lls]  # List of lists for test log-likelihoods

		# Return the models and the lists of log-likelihoods
		return models, train_ll_list, test_ll_list

	# Initialize the dictionary before using it
	models_session_state_fold = {}
	# Initialize arrays to hold the log-likelihoods (train and test)
	train_ll = np.full((n_sessions, len(state_range), k_folds), np.nan)
	test_ll = np.full((n_sessions, len(state_range), k_folds), np.nan)

	for idx_session in range(n_sessions):
		models_session_state_fold[idx_session] = {}
		print(f"Fitting session {idx_session}...")
		for state_idx, n_states in enumerate(state_range):
			print(f"Fitting {n_states} states...")
			models, train_lls, test_lls = process_session_state_fold(idx_session, n_states)
			models_session_state_fold[idx_session][n_states] = models

			# Convert the list of log-likelihoods to arrays
			for fold_idx in range(k_folds):
				train_ll[idx_session, state_idx, fold_idx] = np.max(train_lls[fold_idx])
				test_ll[idx_session, state_idx, fold_idx] = np.max(test_lls[fold_idx])

	return models_session_state_fold, train_ll, test_ll


# def session_wise_fit(observations, inputs, masks, n_sessions, init_params, n_states, fitting_method="em", n_iters=200, tolerance=10**-4, n_jobs=-1):
# 	"""
# 	Optimized version of session-wise GLM-HMM fitting with parallel processing and progress tracking.
# 	"""
# 	masks = [np.ones_like(arr) for arr in observations] if masks is None else masks
# 	assert len(observations) == n_sessions, "Observations are not compatible with number of sessions!"
# 	assert len(inputs) == n_sessions, "Inputs are not compatible with number of sessions!"
# 	assert len(masks) == n_sessions, "Masks are not compatible with number of sessions!"
# 	assert "transition_matrices" in init_params and "glm_weights" in init_params, "Initial parameters not provided correctly!"

# 	def process_session(idx_session):
# 		"""
# 		Fit a GLM-HMM for a specific session.
# 		"""
# 		glm_hmm = ssm.HMM(n_states, observations[0].shape[1], inputs[0].shape[1], observations="input_driven_obs", observation_kwargs=dict(C=len(np.unique(observations[0]))), transitions="standard")
# 		glm_hmm.observations.params = init_params["glm_weights"][idx_session]
# 		glm_hmm.transitions.params = init_params["transition_matrices"][idx_session]

# 		fit_ll = glm_hmm.fit(observations[idx_session], inputs=inputs[idx_session], masks=masks[idx_session], method=fitting_method, num_iters=n_iters, initialize=False, tolerance=tolerance)
# 		return idx_session, glm_hmm, fit_ll

# 	results = []
# 	for result in tqdm(Parallel(n_jobs=n_jobs)(delayed(process_session)(idx_session) for idx_session in range(n_sessions)), total=n_sessions, desc="Fitting sessions"):
# 		results.append(result)

# 	models_session = {idx_session: model for idx_session, model, _ in results}
# 	fit_ll_session = {idx_session: fit_ll for idx_session, _, fit_ll in results}

# 	return models_session, fit_ll_session
