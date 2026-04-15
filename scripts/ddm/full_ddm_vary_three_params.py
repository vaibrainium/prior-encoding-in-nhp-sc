"""
Optimized Decision Models for Drift Diffusion Model Simulation and Fitting

This module provides highly efficient implementations of drift diffusion models
with CUDA support, streamlined likelihood calculations, and robust parameter fitting.
"""

import logging
import warnings

import numpy as np
import torch
from numba import jit, prange
from scipy.optimize import differential_evolution

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@jit(nopython=True, parallel=True)
def _simulate_ddm_trials_numba(stimulus: np.ndarray, drift_gain: float, drift_offset: float, a: float, z: float, ndt: float, dt: float, variance: float, leak_rate: float, time_constant: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated DDM simulation for maximum CPU performance.
    """
    n_trials, n_timepoints = stimulus.shape
    rt = np.full(n_trials, np.nan, dtype=np.float32)
    choice = np.full(n_trials, np.nan, dtype=np.float32)

    starting_point = z * a
    noise_std = np.sqrt(variance * dt)

    for trial in prange(n_trials):
        evidence = starting_point

        for t in range(1, n_timepoints):
            if np.isnan(stimulus[trial, t]):
                break

            # Evidence update
            drift = (drift_gain * stimulus[trial, t - 1] + drift_offset) * dt
            noise = np.random.normal(0, noise_std)
            leak = leak_rate * (evidence - starting_point) * dt

            evidence += drift + noise - leak

            # Apply urgency (t * dt converts step index to seconds; time_constant in 1/s)
            decision_var = evidence
            if time_constant != 0:
                urgency_factor = 1 + t * dt * time_constant
                decision_var = starting_point + (evidence - starting_point) * urgency_factor

            # Check boundaries
            if decision_var >= a:
                rt[trial] = t * dt + ndt
                choice[trial] = 1.0
                break
            if decision_var <= 0:
                rt[trial] = t * dt + ndt
                choice[trial] = 0.0
                break

    return rt, choice


class DriftDiffusionSimulator:
    """
    Highly optimized DDM simulator using Numba for CPU acceleration.
    """

    def __init__(self, leak: bool = True, time_dependence: bool = True):
        # Core parameters
        self.ndt = 0.1
        self.a = 2.0
        self.z = 0.5
        self.drift_gain = 7.0
        self.drift_offset = 0.0
        self.variance = 1.0
        self.dt = 0.001
        self.leak_rate = 0.01 if leak else 0.0
        self.time_constant = 0

        self._validate_parameters()

    def _validate_parameters(self):
        """Validate parameter ranges for numerical stability."""
        assert 0 < self.a < 10, f"Invalid boundary separation: {self.a}"
        assert 0 < self.z < 1, f"Invalid starting point: {self.z}"
        assert 0 < self.dt < 0.01, f"Invalid time step: {self.dt}"
        assert self.ndt >= 0, f"Invalid non-decision time: {self.ndt}"

    def simulate_trials(self, stimulus: np.ndarray) -> tuple[np.ndarray, np.ndarray, None]:
        """
        Simulate DDM trials using optimized Numba implementation.

        Args:
            stimulus: Shape (n_trials, n_timepoints) stimulus array

        Returns:
            Tuple of (reaction_times, choices, None)

        """
        if stimulus.size == 0:
            return np.array([]), np.array([]), None

        stimulus = np.asarray(stimulus, dtype=np.float32)

        rt, choice = _simulate_ddm_trials_numba(stimulus, self.drift_gain, self.drift_offset, self.a, self.z, self.ndt, self.dt, self.variance, self.leak_rate, self.time_constant)

        return rt, choice, None


class DriftDiffusionSimulatorCUDA:
    """
    CUDA-accelerated DDM simulator for GPU computation.
    """

    def __init__(self, leak: bool = True, time_dependence: bool = True, device: str | None = None):
        # Set device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Using device: {self.device}")

        # Parameters as tensors
        self.ndt = torch.tensor(0.1, device=self.device, dtype=torch.float32)
        self.a = torch.tensor(2.0, device=self.device, dtype=torch.float32)
        self.z = torch.tensor(0.5, device=self.device, dtype=torch.float32)
        self.drift_gain = torch.tensor(7.0, device=self.device, dtype=torch.float32)
        self.drift_offset = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.variance = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        self.dt = torch.tensor(0.001, device=self.device, dtype=torch.float32)
        self.leak_rate = torch.tensor(0.01 if leak else 0.0, device=self.device, dtype=torch.float32)
        self.time_constant = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # Precomputed values
        self._noise_std = torch.sqrt(self.variance * self.dt)

    def simulate_trials(self, stimulus: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate trials using CUDA acceleration.

        Args:
            stimulus: Input stimulus array

        Returns:
            Tuple of (reaction_times, choices, decision_variables)

        """
        # Convert to tensor if needed
        if not isinstance(stimulus, torch.Tensor):
            stimulus = torch.tensor(stimulus, device=self.device, dtype=torch.float32)
        else:
            stimulus = stimulus.to(self.device, dtype=torch.float32)

        n_trials, n_timepoints = stimulus.shape
        starting_point = self.z * self.a

        # Initialize arrays
        rt = torch.full((n_trials,), float("nan"), device=self.device, dtype=torch.float32)
        choice = torch.full((n_trials,), float("nan"), device=self.device, dtype=torch.float32)
        evidence = torch.full((n_trials,), starting_point, device=self.device, dtype=torch.float32)

        # Precompute drift rates
        drift_rates = self.drift_gain * stimulus + self.drift_offset

        # Track active trials
        active = torch.ones(n_trials, dtype=torch.bool, device=self.device)

        for t in range(1, n_timepoints):
            if not active.any():
                break

            # Get active trial indices
            active_idx = active.nonzero(as_tuple=True)[0]

            # Check for NaN stimuli
            valid_stim = ~torch.isnan(stimulus[active_idx, t])
            if not valid_stim.any():
                continue

            active_idx = active_idx[valid_stim]
            if len(active_idx) == 0:
                continue

            # Vectorized evidence update
            noise = torch.randn(len(active_idx), device=self.device, dtype=torch.float32) * self._noise_std
            drift = drift_rates[active_idx, t - 1] * self.dt
            leak = self.leak_rate * (evidence[active_idx] - starting_point) * self.dt

            evidence[active_idx] += drift + noise - leak

            # Apply urgency signal (t * dt converts step index to seconds; time_constant in 1/s)
            decision_var = evidence[active_idx]
            if self.time_constant != 0:
                urgency_factor = 1 + t * self.dt * self.time_constant
                decision_var = starting_point + (evidence[active_idx] - starting_point) * urgency_factor

            # Boundary detection
            hit_upper = decision_var >= self.a
            hit_lower = decision_var <= 0
            crossed = hit_upper | hit_lower

            if crossed.any():
                crossed_idx = active_idx[crossed]

                # Set reaction times and choices
                rt[crossed_idx] = t * self.dt + self.ndt
                choice[crossed_idx] = hit_upper[crossed].float()

                # Mark as inactive
                active[crossed_idx] = False

        # Return numpy arrays
        return rt.cpu().numpy(), choice.cpu().numpy(), evidence.cpu().numpy()
        ##### only latest evidence is returned


class LikelihoodCalculator:
    """
    Optimized likelihood calculator using simplified quantile-based approach.
    """

    def __init__(self, nbins: int = 5, rt_weight: float = 1.0):
        self.nbins = nbins
        self.rt_weight = rt_weight
        self.eps = 1e-12

        # Simple cache with size limit
        self._cache = {}
        self._cache_limit = 100

    def _get_quantiles(self, rt_data: np.ndarray) -> np.ndarray:
        """Get quantiles with caching."""
        if len(rt_data) < self.nbins:
            return np.array([rt_data.min(), rt_data.max()])
        # Simple cache key (avoid expensive tuple conversion)
        cache_key = (len(rt_data), rt_data.min(), rt_data.max(), rt_data.mean())

        if cache_key in self._cache:
            return self._cache[cache_key]

        quantiles = np.linspace(0.1, 0.9, self.nbins)
        values = np.quantile(rt_data, quantiles)

        # Update cache with size limit
        if len(self._cache) >= self._cache_limit:
            self._cache.clear()
        self._cache[cache_key] = values

        return values

    def calculate_likelihood(self, rt_pred: np.ndarray, choice_pred: np.ndarray, rt_data: np.ndarray, choice_data: np.ndarray, coherences_pred: np.ndarray, coherences_data: np.ndarray) -> float:
        """
        Calculate total likelihood using streamlined approach.

        Args:
            rt_pred, choice_pred: Model predictions
            rt_data, choice_data: Empirical data
            coherences_pred, coherences_data: Coherence values

        Returns:
            Negative log likelihood

        """
        if len(rt_pred) == 0 or len(rt_data) == 0:
            return 1e6

        # Remove invalid data
        valid_pred = (~np.isnan(rt_pred)) & (~np.isnan(choice_pred))
        valid_data = (~np.isnan(rt_data)) & (~np.isnan(choice_data))

        if not (valid_pred.any() and valid_data.any()):
            return 1e6

        rt_pred = rt_pred[valid_pred]
        choice_pred = choice_pred[valid_pred]
        coherences_pred = coherences_pred[valid_pred]

        rt_data = rt_data[valid_data]
        choice_data = choice_data[valid_data]
        coherences_data = coherences_data[valid_data]

        total_nllh = 0.0
        unique_cohs = np.unique(coherences_data)
        unique_choices = np.unique(choice_data)

        for coh in unique_cohs:
            data_mask = coherences_data == coh
            pred_mask = coherences_pred == coh

            if not (data_mask.any() and pred_mask.any()):
                continue

            # Choice likelihood
            for choice_val in unique_choices:
                n_data = np.sum(choice_data[data_mask] == choice_val)
                if n_data == 0:
                    continue

                p_pred = np.mean(choice_pred[pred_mask] == choice_val)
                total_nllh -= n_data * np.log(p_pred + self.eps)

            # RT likelihood using quantile matching
            for choice_val in unique_choices:
                rt_data_sub = rt_data[data_mask & (choice_data == choice_val)]
                rt_pred_sub = rt_pred[pred_mask & (choice_pred == choice_val)]

                if len(rt_data_sub) < 3 or len(rt_pred_sub) < 3:
                    continue

                # Quantile-based likelihood (fast approximation)
                data_quantiles = self._get_quantiles(rt_data_sub)
                pred_quantiles = np.quantile(rt_pred_sub, np.linspace(0.1, 0.9, len(data_quantiles)))

                # Penalize quantile deviations
                quantile_error = np.mean((data_quantiles - pred_quantiles) ** 2)  ### different
                total_nllh += quantile_error * self.rt_weight * len(rt_data_sub)

        return total_nllh if np.isfinite(total_nllh) else 1e6


class DecisionModel:
    """
    Main class for optimized decision model fitting and simulation.
    """

    def __init__(self, model_name: str = "DDM", enable_leak: bool = True, enable_time_dependence: bool = True, device: str | None = None, likelihood_params: dict | None = None):
        self.model_name = model_name.upper()
        self.enable_leak = enable_leak
        self.enable_time_dependence = enable_time_dependence

        # Initialize simulator
        if self.model_name == "DDM":
            if device == "cuda":
                self.simulator = DriftDiffusionSimulatorCUDA(leak=enable_leak, time_dependence=enable_time_dependence, device=device)
            else:
                self.simulator = DriftDiffusionSimulator(leak=enable_leak, time_dependence=enable_time_dependence)
        else:
            raise NotImplementedError(f"Model '{model_name}' not implemented")

        # Initialize likelihood calculator
        if likelihood_params is None:
            likelihood_params = {"nbins": 5, "rt_weight": 1.0}

        self.likelihood_calc = LikelihoodCalculator(**likelihood_params)

        # Optimization settings
        self._param_bounds = self._setup_parameter_bounds()

        logger.info(f"Initialized {self.model_name} model with device: {getattr(self.simulator, 'device', 'CPU')}")

    def _setup_parameter_bounds(self) -> dict[str, tuple[float, tuple[float, float]]]:
        """Setup parameter bounds based on model configuration."""
        bounds = {
            "ndt": (0.2, (0.1, 0.5)),
            "drift_gain": (7.0, (1.0, 20.0)),
            "variance": (1.0, (0.1, 5.0)),
            # Prior condition 1 (equal) parameters
            "a_1": (2.0, (0.8, 6.0)),
            "z_1": (0.5, (0.1, 0.9)),
            "drift_offset_1": (0.0, (-5.0, 5.0)),
            # Prior condition 2 (unequal) parameters
            "a_2": (2.0, (0.8, 6.0)),
            "z_2": (0.5, (0.1, 0.9)),
            "drift_offset_2": (0.0, (-5.0, 5.0)),
        }

        if self.enable_leak:
            bounds["leak_rate"] = (0.1, (0.0, 1.0))

        if self.enable_time_dependence:
            bounds["time_constant"] = (0.0, (-5.0, 5.0))

        return bounds

    def _update_parameters(self, param_values: np.ndarray, prior_idx: int | None = None):
        """
        Update simulator parameters with support for both single and dual-prior conditions.

        Args:
            param_values: Array of parameter values
            prior_idx: Optional prior index (0 for equal, 1 for unequal) for dual-prior conditions

        """
        param_names = list(self._param_bounds.keys())

        for name, value in zip(param_names, param_values, strict=False):
            # Handle prior-specific parameters when prior_idx is specified
            if prior_idx is not None and name.endswith(f"_{prior_idx + 1}"):
                # Remove the _1 or _2 suffix to get the base parameter name
                base_name = name[:-2]  # removes _1 or _2
                setattr(self.simulator, base_name, value)
            elif prior_idx is None and ("_1" in name or "_2" in name):
                # For final parameter update after optimization, strip suffix
                base_name = name.split("_")[0] if "_" in name else name
                setattr(self.simulator, base_name, value)
            elif "_" not in name or not name[-1].isdigit():
                # Global parameters (no suffix)
                setattr(self.simulator, name, value)

    def _objective_function(self, params: np.ndarray, data: dict, stimulus: np.ndarray, n_reps: int = 5, seed: int = 42, l1_weight: float = 0.01) -> float:
        """
        Objective function for parameter optimization with dual-prior support.

        Args:
            params: Parameter values to evaluate
            data: Empirical data dictionary
            stimulus: Stimulus array
            n_reps: Number of simulation repetitions
            seed: Random seed
            l1_weight: L1 regularization weight

        Returns:
            Negative log likelihood

        """
        # Set seeds for reproducibility
        np.random.seed(seed)
        if hasattr(torch, "manual_seed"):
            torch.manual_seed(seed)

        total_nllh = 0.0
        param_names = list(self._param_bounds.keys())

        # Check if we have prior_block in data (for dual-prior conditions)
        has_prior_blocks = "prior_block" in data and len(np.unique(data["prior_block"])) > 1

        if has_prior_blocks:
            # Handle dual-prior conditions
            for idx_prior, prior in enumerate(["equal", "unequal"]):
                prior_mask = data["prior_block"] == prior
                if not np.any(prior_mask):
                    continue

                # Update parameters for this prior condition
                self._update_parameters(params, prior_idx=idx_prior)

                # Simulate for this prior condition
                all_rt, all_choice, all_coh = [], [], []

                for rep in range(n_reps):
                    try:
                        rt, choice, _ = self.simulator.simulate_trials(stimulus[prior_mask])

                        # Filter valid trials
                        valid = (~np.isnan(rt)) & (~np.isnan(choice))
                        if valid.sum() < 5:  # Need minimum valid trials
                            continue

                        all_rt.append(rt[valid])
                        all_choice.append(choice[valid])
                        all_coh.append(stimulus[prior_mask][valid, 0])

                    except Exception as e:
                        logger.warning(f"Simulation failed for prior {prior}: {e}")
                        return 1e6

                if not all_rt:
                    continue

                # Combine repetitions
                rt_pred = np.concatenate(all_rt)
                choice_pred = np.concatenate(all_choice)
                coh_pred = np.concatenate(all_coh)
                ## did not average repetitions

                # Prepare data for likelihood calculation
                prior_data = {k: v[prior_mask] for k, v in data.items() if k != "prior_block"}

                # Calculate likelihood for this prior condition
                nllh = self.likelihood_calc.calculate_likelihood(
                    rt_pred,
                    choice_pred,
                    prior_data["rt"],
                    prior_data["choice"],
                    coh_pred,
                    prior_data["signed_coherence"],
                )
                total_nllh += nllh
        else:
            # Single condition - use unified parameter update
            self._update_parameters(params)

            all_rt, all_choice, all_coh = [], [], []

            for rep in range(n_reps):
                try:
                    rt, choice, _ = self.simulator.simulate_trials(stimulus)

                    # Filter valid trials
                    valid = (~np.isnan(rt)) & (~np.isnan(choice))
                    if valid.sum() < 5:  # Need minimum valid trials
                        continue

                    all_rt.append(rt[valid])
                    all_choice.append(choice[valid])
                    all_coh.append(stimulus[valid, 0])  # Assuming first column is coherence

                except Exception as e:
                    logger.warning(f"Simulation failed: {e}")
                    return 1e6

            if not all_rt:
                return 1e6

            # Combine all repetitions
            rt_pred = np.concatenate(all_rt)
            choice_pred = np.concatenate(all_choice)
            coh_pred = np.concatenate(all_coh)

            # Calculate likelihood
            nllh = self.likelihood_calc.calculate_likelihood(
                rt_pred,
                choice_pred,
                data["rt"],
                data["choice"],
                coh_pred,
                data["signed_coherence"],
            )
            total_nllh += nllh

        # Add L1 regularization
        total_nllh += l1_weight * np.sum(np.abs(params))

        return total_nllh if np.isfinite(total_nllh) else 1e6

    def fit(self, data: dict[str, np.ndarray], stimulus: np.ndarray, max_iterations: int = 100, n_reps: int = 5, seed: int = 42, l1_weight: float = 0.01, verbose: bool = True) -> dict:
        """
        Fit model parameters to empirical data.

        Args:
            data: Dictionary with keys 'signed_coherence', 'choice', 'rt'
            stimulus: Stimulus array (n_trials, n_timepoints)
            max_iterations: Maximum optimization iterations
            n_reps: Number of simulation repetitions per evaluation
            seed: Random seed for reproducibility
            l1_weight: L1 regularization weight
            verbose: Whether to print progress

        Returns:
            Optimization result dictionary

        """
        if verbose:
            logger.info("Starting parameter optimization...")

        # Setup parameter bounds
        bounds = [self._param_bounds[name][1] for name in self._param_bounds.keys()]
        initial_guess = [self._param_bounds[name][0] for name in self._param_bounds.keys()]

        # Set random seeds
        np.random.seed(seed)

        def objective(params):
            return self._objective_function(params, data, stimulus, n_reps, seed, l1_weight)

        # Use differential evolution for global optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = differential_evolution(objective, bounds=bounds, maxiter=max_iterations, popsize=10, seed=seed, polish=True, disp=verbose)

        if verbose:
            logger.info(f"Optimization completed. Final cost: {result.fun:.2f}")
            logger.info("Best-fit parameters:")
            for name, value in zip(self._param_bounds.keys(), result.x, strict=False):
                logger.info(f"  {name} = {value:.4f}")

        # Return comprehensive results
        return {"success": result.success, "parameters": dict(zip(self._param_bounds.keys(), result.x, strict=False)), "likelihood": result.fun, "n_iterations": result.nit, "optimization_result": result}

    def simulate(self, stimulus: np.ndarray, n_reps: int = 1) -> dict[str, np.ndarray]:
        """
        Simulate model responses for given stimulus.

        Args:
            stimulus: Stimulus array (n_trials, n_timepoints)
            n_reps: Number of repetitions

        Returns:
            Dictionary with simulation results

        """
        all_rt, all_choice = [], []

        for rep in range(n_reps):
            rt, choice, _ = self.simulator.simulate_trials(stimulus)
            all_rt.append(rt)
            all_choice.append(choice)

        return {"rt": np.concatenate(all_rt) if n_reps > 1 else all_rt[0], "choice": np.concatenate(all_choice) if n_reps > 1 else all_choice[0], "coherence": np.tile(stimulus[:, 0], n_reps) if n_reps > 1 else stimulus[:, 0]}


if __name__ == "__main__":
    # Quick demonstration
    logger.info("Drift Diffusion Model - Optimized Implementation")

    # Test basic functionality
    n_trials, n_timepoints = 100, 1000
    stimulus = np.random.randn(n_trials, n_timepoints) * 0.1

    # Test CPU simulator
    model_cpu = DecisionModel(device=None)
    rt, choice, _ = model_cpu.simulator.simulate_trials(stimulus)
    logger.info(f"CPU simulation: {np.sum(~np.isnan(rt))} valid trials")

    # Test CUDA if available
    if torch.cuda.is_available():
        model_cuda = DecisionModel(device="cuda")
        rt_cuda, choice_cuda, _ = model_cuda.simulator.simulate_trials(stimulus)
        logger.info(f"CUDA simulation: {np.sum(~np.isnan(rt_cuda))} valid trials")

    logger.info("All tests passed!")
