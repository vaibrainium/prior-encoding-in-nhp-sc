import numpy as np
import numpy.random as npr
import ssm
from sklearn.model_selection import KFold
from joblib import Parallel, delayed


def global_fit(observations, inputs, masks, state_range=np.arange(2, 6), n_initializations=20,
               fitting_method='em', n_iters=200, tolerance=10**-4, n_jobs=-1):
    """
    Optimized version of global GLM-HMM fitting with parallelization over initializations.
    """
    print('Fitting GLM globally...')
    glm = ssm.HMM(
        1, observations[0].shape[1], inputs[0].shape[1],
        observations="input_driven_obs",
        observation_kwargs=dict(C=len(np.unique(observations[0]))),
        transitions="standard"
    )
    glm.fit(observations, inputs=inputs, masks=masks, method=fitting_method, num_iters=n_iters, tolerance=tolerance)
    glm_weights = glm.observations.params

    def fit_single_initialization(n_states, init_num):
        """
        Fit GLM-HMM with a single initialization.
        """
        npr.seed(init_num*n_states) # Set seed for reproducibility
        
        glm_hmm = ssm.HMM(
            n_states, observations[0].shape[1], inputs[0].shape[1],
            observations="input_driven_obs",
            observation_kwargs=dict(C=len(np.unique(observations[0]))),
            transitions="standard"
        )
        # Initialize weights and transition matrix
        glm_hmm.observations.params = glm_weights + np.random.normal(0, 0.2, (n_states, 1, inputs[0].shape[1]))
        transition_matrix = 0.95 * np.eye(n_states) + np.random.multivariate_normal(
            mean=np.zeros(n_states), cov=0.05 * np.eye(n_states), size=n_states
        )
        transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
        glm_hmm.transitions.params = [transition_matrix]

        fit_ll = glm_hmm.fit(observations, inputs=inputs, masks=masks, method=fitting_method,
                             num_iters=n_iters, initialize=False, tolerance=tolerance)
        return glm_hmm, fit_ll

    models_glm_hmm = {}
    fit_lls_glm_hmm = {}
    for n_states in state_range:
        print(f'Fitting {n_states} states...')

        
        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_single_initialization)(n_states, init_num)
            for init_num in range(n_initializations)
        )
        models, fit_lls = zip(*results)
        
        # Store results in the dictionaries
        models_glm_hmm[n_states] = list(models)
        fit_lls_glm_hmm[n_states] = list(fit_lls) 

    return models_glm_hmm, fit_lls_glm_hmm


def session_wise_fit_cv(observations, inputs, masks, n_sessions, init_params, k_folds=5, state_range=np.arange(2, 6),
                        fitting_method='em', n_iters=200, tolerance=10**-4, n_jobs=-1):
    """
    Optimized version of session-wise GLM-HMM fitting with k-fold cross-validation using parallelization.
    """
    assert len(observations) == n_sessions, "Observations are not compatible with number of sessions!"
    assert len(inputs) == n_sessions, "Inputs are not compatible with number of sessions!"
    assert len(masks) == n_sessions, "Masks are not compatible with number of sessions!"
    assert "transition_matrices" in init_params.keys() and "glm_weights" in init_params.keys(), "Initial parameters not provided correctly!"

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=1)

    def fit_model_on_fold(train_idx, test_idx, glm_hmm, idx_session):
        """
        Fit a GLM-HMM on one fold and compute training and testing log-likelihoods.
        """
        train_ll = glm_hmm.fit(
            observations[idx_session][train_idx],
            inputs=inputs[idx_session][train_idx],
            masks=masks[idx_session][train_idx],
            method=fitting_method,
            num_iters=n_iters,
            initialize=False,
            tolerance=tolerance
        )
        test_ll = glm_hmm.log_likelihood(
            observations[idx_session][test_idx],
            inputs=inputs[idx_session][test_idx],
            masks=masks[idx_session][test_idx]
        )
        return glm_hmm, train_ll, test_ll

    def process_session_state_fold(idx_session, n_states):
        """
        Fit a GLM-HMM for a specific session and state with cross-validation.
        """
        glm_hmm = ssm.HMM(
            n_states, observations[0].shape[1], inputs[0].shape[1],
            observations="input_driven_obs",
            observation_kwargs=dict(C=len(np.unique(observations[0]))),
            transitions="standard"
        )
        glm_hmm.observations.params = init_params['glm_weights'][n_states]
        glm_hmm.transitions.params = init_params['transition_matrices'][n_states]

        # Collecting results across all folds
        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_model_on_fold)(train_idx, test_idx, glm_hmm, idx_session)
            for train_idx, test_idx in kf.split(np.arange(observations[idx_session].shape[0]))
        )

        # Unzip results and ensure consistent shape for log-likelihoods
        models, train_lls, test_lls = zip(*results)

        # Convert train_lls and test_lls into lists of lists to handle varying lengths
        train_ll_list = [ll for ll in train_lls]  # List of lists for train log-likelihoods
        test_ll_list = [ll for ll in test_lls]    # List of lists for test log-likelihoods

        # Return the models and the lists of log-likelihoods
        return models, train_ll_list, test_ll_list

    # Initialize the dictionary before using it
    models_session_state_fold = {}

    # Initialize arrays to hold the log-likelihoods (train and test)
    train_ll = np.full((n_sessions, len(state_range), k_folds), np.nan)
    test_ll = np.full((n_sessions, len(state_range), k_folds), np.nan)

    for idx_session in range(n_sessions):
        models_session_state_fold[idx_session] = {}
        print(f'Fitting session {idx_session}...')
        for state_idx, n_states in enumerate(state_range):
            print(f'Fitting {n_states} states...')
            models, train_lls, test_lls = process_session_state_fold(idx_session, n_states)
            models_session_state_fold[idx_session][n_states] = models
            # Convert the list of log-likelihoods to arrays
            for fold_idx in range(k_folds):
                train_ll[idx_session, state_idx, fold_idx] = np.max(train_lls[fold_idx])
                test_ll[idx_session, state_idx, fold_idx] = np.max(test_lls[fold_idx])

    return models_session_state_fold, train_ll, test_ll