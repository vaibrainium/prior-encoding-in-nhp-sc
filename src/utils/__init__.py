
class plot_utils:
    from .plotting import (
        figure_by_height,
        figure_by_width,
        figure_with_cbar_by_height,
        grid_by_height,
        grid_by_width,
        plot_scatter,
        plot_line,
        plot_errorbar,
        plot_x_errorbar,
    )

class pmf_utils:
    from .PMF_utils import (
        fit_psychometric_function,
        get_psychometric_data,
        get_chronometric_data,
    )

# Create a limited interface for glm_hmm_utils
class glm_hmm_utils:
    from .glm_hmm_utils import global_fit, session_wise_fit_cv, session_wise_fit

import ephys_utils

__all__ = ["plot_utils", "pmf_utils", "glm_hmm_utils", "ephys_utils"]
