"""

class for fitting Generalized Linear Models (GLMs).

"""


from warnings import simplefilter

import autograd.numpy as np
from autograd import hessian, value_and_grad
from scipy import optimize
