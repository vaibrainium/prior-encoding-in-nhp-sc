import numpy as np
import numpy.random as npr
import ssm
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from tqdm import tqdm


def global_fit(observations, inputs, masks, state_range=np.arange(2, 6), n_initializations=20, fitting_method="em", n_iters=200, tolerance=10**-4, n_jobs=-1):
	"""
	Optimized version of global GLM-HMM fitting with parallelization over initializations.
	"""
	print("Fitting GLM globally...")
	glm = ssm.HMM(1, observations[0].shape[1], inputs[0].shape[1], observations="input_driven_obs", observation_kwargs=dict(C=len(np.unique(observations[0]))), transitions="standard")

	glm.fit(observations, inputs=inputs, masks=masks, method=fitting_method, num_iters=n_iters, tolerance=tolerance)
	glm_weights = glm.observations.params

	def fit_single_initialization(n_states, init_num):
		"""
		Fit GLM-HMM with a single initialization.
		"""
		npr.seed(init_num * n_states)  # Set seed for reproducibility
		glm_hmm = ssm.HMM(n_states, observations[0].shape[1], inputs[0].shape[1], observations="input_driven_obs", observation_kwargs=dict(C=len(np.unique(observations[0]))), transitions="standard")

		# Initialize weights and transition matrix
		glm_hmm.observations.params = glm_weights + np.random.normal(0, 0.2, (n_states, 1, inputs[0].shape[1]))
		transition_matrix = 0.9 * np.eye(n_states) + np.random.multivariate_normal(mean=np.zeros(n_states), cov=0.05 * np.eye(n_states), size=n_states)
		transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
		glm_hmm.transitions.params = [transition_matrix]
		fit_ll = glm_hmm.fit(observations, inputs=inputs, masks=masks, method=fitting_method, num_iters=n_iters, initialize=False, tolerance=tolerance)
		return glm_hmm, fit_ll

	models_glm_hmm = {}
	fit_lls_glm_hmm = {}

	for n_states in state_range:
		print(f"Fitting {n_states} states...")

		results = Parallel(n_jobs=n_jobs)(delayed(fit_single_initialization)(n_states, init_num) for init_num in range(n_initializations))
		models, fit_lls = zip(*results)

		# Store results in the dictionaries
		models_glm_hmm[n_states] = list(models)
		fit_lls_glm_hmm[n_states] = list(fit_lls)
	return models_glm_hmm, fit_lls_glm_hmm



def session_wise_fit(observations, inputs, masks, n_sessions, init_params, n_states, fitting_method="em", n_iters=200, tolerance=10**-4, n_jobs=-1):
	"""
	Optimized version of session-wise GLM-HMM fitting with parallel processing and progress tracking.
	"""
	masks = [np.ones_like(arr) for arr in observations] if masks is None else masks
	assert len(observations) == n_sessions, "Observations are not compatible with number of sessions!"
	assert len(inputs) == n_sessions, "Inputs are not compatible with number of sessions!"
	assert len(masks) == n_sessions, "Masks are not compatible with number of sessions!"
	assert "transition_matrices" in init_params and "glm_weights" in init_params, "Initial parameters not provided correctly!"

	def process_session(idx_session):
		"""
		Fit a GLM-HMM for a specific session.
		"""
		glm_hmm = ssm.HMM(n_states, observations[0].shape[1], inputs[0].shape[1], observations="input_driven_obs", observation_kwargs=dict(C=len(np.unique(observations[0]))), transitions="standard")
		glm_hmm.observations.params = init_params["glm_weights"][idx_session]
		glm_hmm.transitions.params = init_params["transition_matrices"][idx_session]

		fit_ll = glm_hmm.fit(observations[idx_session], inputs=inputs[idx_session], masks=masks[idx_session], method=fitting_method, num_iters=n_iters, initialize=False, tolerance=tolerance)
		return idx_session, glm_hmm, fit_ll

	results = []
	for result in tqdm(Parallel(n_jobs=n_jobs)(delayed(process_session)(idx_session) for idx_session in range(n_sessions)), total=n_sessions, desc="Fitting sessions"):
		results.append(result)

	models_session = {idx_session: model for idx_session, model, _ in results}
	fit_ll_session = {idx_session: fit_ll for idx_session, _, fit_ll in results}

	return models_session, fit_ll_session
