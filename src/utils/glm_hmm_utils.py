import numpy as np
import ssm
from sklearn.model_selection import KFold



def session_wise_fit_cv(observations, inputs, masks, n_sessions, init_params, k_folds=5, state_range=np.arange(2,6), method='em', n_iters=200, tolerance=10**-4):
    assert len(observations) == n_sessions, "Observations are not compatible with number of sessions!"
    assert len(inputs) == n_sessions, "Inputs are not compatible with number of sessions!"
    assert len(masks) == n_sessions, "Masks are not compatible with number of sessions!"
    assert "transition_matrices" in init_params.keys() & "glm_weights" in init_params.keys(), "Initial parameters not provided correctly!"

    kf = KFold(n_splits=k_folds,shuffle=True,random_state=1)

    def cross_validation(k_fold_indices,idx_session,glm_hmm):
        models_glm_hmm, train_ll, test_ll, test_accuracy = {}, [], [], {}

        for idx,(train_index, test_index) in enumerate(k_fold_indices):
            train_ll.append(glm_hmm.fit(observations[idx_session][train_index,:], inputs=inputs[idx_session][train_index,:], 
                                 masks = masks[idx_session][train_index,:], method=method, num_iters=n_iters, initialize=False, tolerance=tolerance))
            test_ll.append(glm_hmm.log_likelihood(observations[idx_session][test_index,:],
                                          inputs=inputs[idx_session][test_index,:], masks=masks[idx_session][test_index,:]))
            models_glm_hmm[idx] = glm_hmm
        return models_glm_hmm, train_ll, test_ll, test_accuracy


    models_session_state_fold = {}
    train_ll, test_ll = np.zeros([n_sessions,len(state_range),k_folds]) * np.nan, np.zeros([n_sessions,len(state_range),k_folds]) * np.nan

    for idx_session in range(n_sessions):
        k_fold_indices = kf.split(np.arange(observations[idx_session].shape[0]))
        models_session_state_fold[idx_session] = {}

        for n_states in state_range:

            glm_hmm = ssm.HMM(n_states, observations[0].shape[1] , inputs[0].shape[1], observations="input_driven_obs", 
                    observation_kwargs=dict(C=len(np.unique(observations[0]))), transitions="standard")
            glm_hmm.observations.params =init_params['glm_weights'][n_states]
            glm_hmm.transitions.params =init_params['transition_matrices'][n_states]

            models_session_state_fold[idx_session][n_states], train_ll[idx_session,n_states,:], test_ll[idx_session,n_states,:], _ = cross_validation(k_fold_indices,idx_session,glm_hmm)


    return models_session_state_fold, train_ll, test_ll


def global_fit(observations, inputs, masks, state_range=np.arange(2,6), n_initializations=20, method='em', n_iters=200, tolerance=10**-4):

    print('fitting GLM globally.....')
    glm = ssm.HMM(1,observations[0].shape[1] , inputs[0].shape[1], observations="input_driven_obs", 
                   observation_kwargs=dict(C=len(np.unique(observations[0]))), transitions="standard")
    glm.fit(observations, inputs=inputs, masks = masks,method=method, num_iters=n_iters, tolerance=tolerance)
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
            
            glm_hmm.observations.params = glm_weights + np.random.normal(0,0.2,(n_states,1,inputs[0].shape[1]))
            transition_matrix = 0.95 * np.eye(n_states) + np.random.multivariate_normal(
                        mean=np.zeros(n_states), cov=0.05*np.eye(n_states), size=n_states)
            transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
            glm_hmm.transitions.params = [transition_matrix]

            fit_ll = glm_hmm.fit(observations, inputs=inputs, masks = masks,method=method, num_iters=n_iters, initialize=False, tolerance=tolerance)
            
            k_state_fit_ll.append(fit_ll)
            k_state_model.append(glm_hmm)

        fit_lls_glm_hmm[n_states] = k_state_fit_ll 
        models_glm_hmm[n_states] = k_state_model

    return models_glm_hmm, fit_lls_glm_hmm
