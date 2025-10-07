class plot_utils:
	from .plotting import (figure_by_height, figure_by_width,
	                       figure_with_cbar_by_height, grid_by_height,
	                       grid_by_width, plot_errorbar, plot_line, plot_scatter,
	                       plot_x_errorbar)


class pmf_utils:
	from .pmf_utils import (fit_psychometric_function, get_chronometric_data,
	                        get_psychometric_data)


# Create a limited interface for glm_hmm_utils
class glm_hmm_utils:
	from .glm_hmm_utils import global_fit, session_wise_fit
	from .glm_hmm_utils_cv import session_wise_fit_cv


class ephys_utils:
	from .ephys_utils import get_neural_data_from_trial_num, get_trial_num

class poisson_glm_utils:
    from .poisson_glm_utils import (convolve_with_basis,
                                    create_post_spike_history_matrix,
                                    make_post_spike_history_basis,
                                    make_smooth_temporal_basis,
                                    reconstruct_kernels_from_weights)

__all__ = ["plot_utils", "pmf_utils", "glm_hmm_utils", "ephys_utils", "poisson_glm_utils"]
