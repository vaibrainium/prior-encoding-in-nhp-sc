import numpy as np
import ssm
from sklearn.model_selection import KFold



def session_wise_fit_cv(observations, inputs, masks, n_sessions, init_params, k_folds=5, state_range=np.arange(2,6), fitting_method='em', n_iters=200, tolerance=10**-4):
    """
    Fit n-state GLM_HMM for every session with provided initialization parameters and k-fold cross validation
    Args:
        observations (list): (n_sessions x n_trials_per_session x obs_dim)
        inputs (list): (n_sessions x n_trials_per_session x input_dim)
        masks (list): (n_sessions x n_trials_per_session x 1), Boolean(1 for included trial, 0 for excluded trials)
        n_sessions (int): 
        init_params (dict): keys are "glm_weights" and "transition matrics", values contain dict of intial parameter for each state
        k_folds (int),*default=5 : k-fold for cross validation
        state_range (1d array/list), *default = np.arange(2,6): list of number of states in HMM
        fitting_method (str), *default='em : em, map...
        n_iters (int), * default=200: maximum number of EM iterations. 
        tolerance (float), *default=10e-4: fitting will stop earlier if increase in LL is below tolerance specified by tolerance parameter
    
    Returns:
        models_session_state_fold (dict): GLM-HMM models saved with levels: sessions -> number of states in HMM -> k-fold
        train_ll (ndarray): (n_sessions x num_all_fitted_states x k_folds), log likelihood of GLM-HMM on training set
        test_ll (ndarray): (n_sessions x num_all_fitted_states x k_folds), log likelihood of GLM-HMM on testing set
    """
    assert len(observations) == n_sessions, "Observations are not compatible with number of sessions!"
    assert len(inputs) == n_sessions, "Inputs are not compatible with number of sessions!"
    assert len(masks) == n_sessions, "Masks are not compatible with number of sessions!"
    assert "transition_matrices" in init_params.keys() & "glm_weights" in init_params.keys(), "Initial parameters not provided correctly!"

    kf = KFold(n_splits=k_folds,shuffle=True,random_state=1)

    def cross_validation(k_fold_indices,idx_session,glm_hmm):
        models_glm_hmm, train_ll, test_ll, test_accuracy = {}, [], [], {}

        for idx,(train_index, test_index) in enumerate(k_fold_indices):
            train_ll.append(glm_hmm.fit(observations[idx_session][train_index,:], inputs=inputs[idx_session][train_index,:], 
                                 masks = masks[idx_session][train_index,:], method=fitting_method, num_iters=n_iters, initialize=False, tolerance=tolerance))
            test_ll.append(glm_hmm.log_likelihood(observations[idx_session][test_index,:],
                                          inputs=inputs[idx_session][test_index,:], masks=masks[idx_session][test_index,:]))
            models_glm_hmm[idx] = glm_hmm
        return models_glm_hmm, train_ll, test_ll, test_accuracy


    models_session_state_fold = {}
    train_ll, test_ll = np.zeros([n_sessions,len(state_range),k_folds]) * np.nan, np.zeros([n_sessions,len(state_range),k_folds]) * np.nan

    for idx_session in range(n_sessions):
        k_fold_indices = kf.split(np.arange(observations[idx_session].shape[0]))
        models_session_state_fold[idx_session] = {}

        for state_idx, n_states in enumerate(state_range):

            glm_hmm = ssm.HMM(n_states, observations[0].shape[1] , inputs[0].shape[1], observations="input_driven_obs", 
                    observation_kwargs=dict(C=len(np.unique(observations[0]))), transitions="standard")
            glm_hmm.observations.params =init_params['glm_weights'][n_states]
            glm_hmm.transitions.params =init_params['transition_matrices'][n_states]

            models_session_state_fold[idx_session][n_states], train_ll[idx_session,state_idx,:], test_ll[idx_session,state_idx,:], _ = cross_validation(k_fold_indices,idx_session,glm_hmm)


    return models_session_state_fold, train_ll, test_ll


def global_fit(observations, inputs, masks, state_range=np.arange(2,6), n_initializations=20, fitting_method='em', n_iters=200, tolerance=10**-4):
    """
    Fit data aggregated from all sessions with GLM-HMM with a list of states(n_initializations per state). All GLM-HMM models are initialized by noise plus weights of best GLM model fitted on whole data 
    Args:
        observations (list): 1 x n_trials x obs_dim
        inputs (list): 1 x n_trials x input_dim
        masks (list): 1 x n_trials x 1, Boolean(1 for included trial, 0 for excluded trials)
        state_range (1d array/list), *default = np.arange(2,6): list of number of states in HMM
        n_intializations (int), *default=20: number of times to fit glm-hmm with different parameter initializations
        fitting_method (str), *default='em : em, map...
        n_iters (int), * default=200: maximum number of EM iterations. 
        tolerance (float), *default=10e-4: fitting will stop earlier if increase in LL is below tolerance specified by tolerance parameter
    
    Returns:
        models_glm_hmm (dict): keys: number of states n, values: list of GLM-HMM models with the n states and n_initializations
        fit_lls_glm_hmm (dict): keys: number of states n, values: list of fitted log likelihood of GLM-HMM models with the n states and n_initializations
    """
    print('fitting GLM globally.....')
    glm = ssm.HMM(1,observations[0].shape[1] , inputs[0].shape[1], observations="input_driven_obs", 
                   observation_kwargs=dict(C=len(np.unique(observations[0]))), transitions="standard")
    glm.fit(observations, inputs=inputs, masks = masks,method=fitting_method, num_iters=n_iters, tolerance=tolerance)
    glm_weights = glm.observations.params

    models_glm_hmm = {}
    fit_lls_glm_hmm = {}

    print('fitting GLM-HMM globally.....')
    for n_states in state_range:
        k_state_model = []
        k_state_fit_ll = []
        print(f'fitting {n_states} states.....')
        for init_num in range(n_initializations):
            print('Initialization  ',str(init_num+1))
            glm_hmm = ssm.HMM(n_states, observations[0].shape[1] , inputs[0].shape[1], observations="input_driven_obs", 
                   observation_kwargs=dict(C=len(np.unique(observations[0]))), transitions="standard")
            
            # intialize glm-hmm with glm weights+noise and default transition matrix
            glm_hmm.observations.params = glm_weights + np.random.normal(0,0.2,(n_states,1,inputs[0].shape[1]))
            transition_matrix = 0.95 * np.eye(n_states) + np.random.multivariate_normal(
                        mean=np.zeros(n_states), cov=0.05*np.eye(n_states), size=n_states)
            transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
            glm_hmm.transitions.params = [transition_matrix]

            fit_ll = glm_hmm.fit(observations, inputs=inputs, masks = masks,method=fitting_method, num_iters=n_iters, initialize=False, tolerance=tolerance)
            
            k_state_fit_ll.append(fit_ll)
            k_state_model.append(glm_hmm)

        fit_lls_glm_hmm[n_states] = k_state_fit_ll 
        models_glm_hmm[n_states] = k_state_model

    return models_glm_hmm, fit_lls_glm_hmm
