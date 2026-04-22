"""
ddm_base.py

Pure DDM fitting engine. This module knows nothing about which parameters
exist, how many conditions there are, or which parameters are shared vs.
condition-specific. All of that is the responsibility of concrete model
classes in ddm_model.py.

What lives here
---------------
DDMParams                    -- flat container of all simulator inputs
ParamSpec                    -- (default, bounds) descriptor for a free parameter
SimResult                    -- typed outcome of _simulate_condition
DriftDiffusionSimulator      -- CPU (Numba) backend
DriftDiffusionSimulatorCUDA  -- GPU/CPU (PyTorch) backend
LikelihoodCalculator         -- quantile-based NLL
DecisionModel                -- abstract fitting engine

What does NOT live here
-----------------------
- Which parameters are free, fixed, or shared across conditions
- Parameter naming conventions (suffixes, per-condition keys)
- Any specific DDMParams field referenced by name outside of DDMParams itself
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import torch
from numba import jit, prange
from scipy.optimize import differential_evolution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Column index of coherence in the stimulus array.
COH_COL: int = 0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DDMParams:
    """
    Flat container of all simulator inputs.

    Passed by value into simulate_trials() so that parameter updates are
    atomic and validate() can run before any simulation starts.
    """
    ndt: float = 0.1
    a: float = 2.0
    z: float = 0.5
    drift_gain: float = 7.0
    drift_offset: float = 0.0
    variance: float = 1.0
    dt: float = 0.001
    leak_rate: float = 0.0
    time_constant: float = 0.0

    def validate(self) -> None:
        if not (0 < self.a < 10):
            raise ValueError(f"Invalid boundary separation: {self.a}")
        if not (0 < self.z < 1):
            raise ValueError(f"Invalid starting point: {self.z}")
        if not (0 < self.dt < 0.01):
            raise ValueError(f"Invalid time step: {self.dt}")
        if self.ndt < 0:
            raise ValueError(f"Invalid non-decision time: {self.ndt}")


@dataclass
class ParamSpec:
    """
    Descriptor for a single free parameter: its default value and (lo, hi) bounds.

    Lives in base so the fitting engine can read bounds generically without
    knowing what the parameters mean. The actual specs dicts mapping names to
    ParamSpec instances are defined in ddm_model.py. Callers can freely
    subclass or instantiate ParamSpec to override defaults or bounds.
    """
    default: float
    bounds: tuple[float, float]


@dataclass
class SimResult:
    """
    Typed outcome of _simulate_condition.

    Exactly one of the three states is active:
      crashed=True          -- simulator raised an exception; caller returns 1e6
      no_valid_trials=True  -- all reps produced <5 valid trials; caller may skip
      rt/choice/coherence   -- valid arrays ready for likelihood calculation
    """
    crashed: bool = False
    no_valid_trials: bool = False
    rt: Optional[np.ndarray] = None
    choice: Optional[np.ndarray] = None
    coherence: Optional[np.ndarray] = None

    @property
    def ok(self) -> bool:
        return not self.crashed and not self.no_valid_trials


# ---------------------------------------------------------------------------
# Simulator protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class DDMSimulator(Protocol):
    """Common interface for all DDM simulator backends."""

    def simulate_trials(
        self, stimulus: np.ndarray, params: DDMParams
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (reaction_times, choices), both shape (n_trials,)."""
        ...


# ---------------------------------------------------------------------------
# Numba (CPU) simulator
# ---------------------------------------------------------------------------

@jit(nopython=True, parallel=True)
def _simulate_ddm_trials_numba(
    stimulus: np.ndarray,
    drift_gain: float,
    drift_offset: float,
    a: float,
    z: float,
    ndt: float,
    dt: float,
    variance: float,
    leak_rate: float,
    time_constant: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Numba-accelerated DDM simulation (CPU parallel)."""
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

            drift = (drift_gain * stimulus[trial, t - 1] + drift_offset) * dt
            noise = np.random.normal(0, noise_std)
            leak  = leak_rate * (evidence - starting_point) * dt
            evidence += drift + noise - leak

            if time_constant != 0.0:
                urgency_factor = 1.0 + t * dt * time_constant
                decision_var = starting_point + (evidence - starting_point) * urgency_factor
            else:
                decision_var = evidence

            if decision_var >= a:
                rt[trial] = t * dt + ndt
                choice[trial] = 1.0
                break
            if decision_var <= 0.0:
                rt[trial] = t * dt + ndt
                choice[trial] = 0.0
                break

    return rt, choice


class DriftDiffusionSimulator:
    """DDM simulator using Numba for CPU parallelism."""

    def simulate_trials(
        self, stimulus: np.ndarray, params: DDMParams
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate DDM trials on CPU.

        Args:
            stimulus: (n_trials, n_timepoints) float32 array.
            params:   Validated DDMParams instance.

        Returns:
            (rt, choice) arrays of shape (n_trials,).
        """
        if stimulus.size == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        stim = np.asarray(stimulus, dtype=np.float32)
        return _simulate_ddm_trials_numba(
            stim,
            params.drift_gain, params.drift_offset,
            params.a, params.z, params.ndt, params.dt,
            params.variance, params.leak_rate, params.time_constant,
        )


# ---------------------------------------------------------------------------
# PyTorch (CUDA/CPU) simulator
# ---------------------------------------------------------------------------

class DriftDiffusionSimulatorCUDA:
    """DDM simulator using PyTorch for vectorised GPU (or CPU) computation."""

    def __init__(self, device: str | None = None):
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable, falling back to CPU.")
            device = "cpu"
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info("Using device: %s", self.device)

    def simulate_trials(
        self, stimulus: np.ndarray | torch.Tensor, params: DDMParams
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate DDM trials on GPU or CPU via PyTorch.

        Urgency tensor is precomputed once before the time loop.
        All noise is drawn in a single kernel call.

        Args:
            stimulus: (n_trials, n_timepoints) array or Tensor.
            params:   Validated DDMParams instance.

        Returns:
            (rt, choice) as numpy arrays of shape (n_trials,).
        """
        dev = self.device

        if not isinstance(stimulus, torch.Tensor):
            stim = torch.tensor(stimulus, device=dev, dtype=torch.float32)
        else:
            stim = stimulus.to(dev, dtype=torch.float32)

        n_trials, n_timepoints = stim.shape

        a             = torch.tensor(params.a,             device=dev, dtype=torch.float32)
        z             = torch.tensor(params.z,             device=dev, dtype=torch.float32)
        ndt           = torch.tensor(params.ndt,           device=dev, dtype=torch.float32)
        drift_gain    = torch.tensor(params.drift_gain,    device=dev, dtype=torch.float32)
        drift_offset  = torch.tensor(params.drift_offset,  device=dev, dtype=torch.float32)
        variance      = torch.tensor(params.variance,      device=dev, dtype=torch.float32)
        dt            = torch.tensor(params.dt,            device=dev, dtype=torch.float32)
        leak_rate     = torch.tensor(params.leak_rate,     device=dev, dtype=torch.float32)
        time_constant = torch.tensor(params.time_constant, device=dev, dtype=torch.float32)

        starting_point = z * a
        noise_std = torch.sqrt(variance * dt)

        rt       = torch.full((n_trials,), float("nan"), device=dev, dtype=torch.float32)
        choice   = torch.full((n_trials,), float("nan"), device=dev, dtype=torch.float32)
        evidence = torch.full((n_trials,), starting_point.item(), device=dev, dtype=torch.float32)
        active   = torch.ones(n_trials, dtype=torch.bool, device=dev)

        drift_rates = drift_gain * stim + drift_offset  # (N, T)
        noise = torch.randn(n_trials, n_timepoints, device=dev, dtype=torch.float32) * noise_std

        use_urgency = params.time_constant != 0.0
        if use_urgency:
            t_idx = torch.arange(n_timepoints, device=dev, dtype=torch.float32)
            urgency_factors = 1.0 + t_idx * dt * time_constant  # (T,)

        for t in range(1, n_timepoints):
            if not active.any():
                break

            drift = drift_rates[:, t - 1] * dt
            leak  = leak_rate * (evidence - starting_point) * dt
            evidence = evidence + drift + noise[:, t] - leak

            if use_urgency:
                decision_var = starting_point + (evidence - starting_point) * urgency_factors[t]
            else:
                decision_var = evidence

            hit_upper = (decision_var >= a) & active
            hit_lower = (decision_var <= 0) & active
            crossed   = hit_upper | hit_lower

            if crossed.any():
                rt[crossed]     = t * dt + ndt
                choice[crossed] = hit_upper[crossed].float()
                active[crossed] = False

        return rt.cpu().numpy(), choice.cpu().numpy()

class LikelihoodCalculator:
    """
    Computes a combined negative log-likelihood from choice proportions and an
    RT term that adapts based on how many predicted trials are available per cell.

    RT term:
      - >= sparse_threshold predicted trials: KL-divergence multinomial NLL
        (Heathcote, Brown & Mewhort 2002). Bins both observed and predicted RTs
        using data-quantile boundaries, then computes KL(obs || pred) * n_obs.
        This equals zero when pred matches obs perfectly and is scale-consistent
        with the choice NLL (both in nats). No rt_weight needed.
      - < sparse_threshold predicted trials: quantile MSE fallback, scaled by
        rt_weight * n_obs. rt_weight is a no-op on the KL path.

    The threshold is on rt_pred (not rt_data) because sparsity on the predicted
    side is what destabilises bin probability estimates.

    Args:
        nbins:            Number of interior quantile boundaries, giving nbins+1
                          RT bins with equal probability mass under the data.
        rt_weight:        Scalar applied to the MSE fallback term only.
        sparse_threshold: Minimum rt_pred trials required for the KL path.
        p_min:            Choice probabilities are clipped to [p_min, 1-p_min]
                          before the log. Prevents near-zero NLL from degenerate
                          parameter combinations that predict p~1 for all choices.
                          Physiologically motivated: real DDMs always produce some
                          errors even at the highest coherence (~5%).
    """

    def __init__(
        self,
        nbins: int = 5,
        rt_weight: float = 1.0,
        sparse_threshold: int = 2,
        p_min: float = 0.05,
    ):
        self.nbins            = nbins
        self.rt_weight        = rt_weight
        self.sparse_threshold = sparse_threshold
        self.p_min            = p_min
        self.eps              = 1e-12
        # Equal-probability quantile grid: nbins interior boundaries divide
        # the RT distribution into nbins+1 bins with equal mass under the data.
        # Using linspace(1/(nbins+1), nbins/(nbins+1), nbins) rather than
        # linspace(0.1, 0.9, nbins) ensures no overflow artifact in the outer
        # bins -- every bin captures exactly 1/(nbins+1) of the data.
        self._q_grid = np.linspace(1 / (nbins + 1), nbins / (nbins + 1), nbins)
        # Separate finer grid for the MSE fallback (interior quantiles only).
        self._mse_grid = np.linspace(0.1, 0.9, nbins)

    def _data_boundaries(self, rt: np.ndarray) -> np.ndarray:
        """
        nbins interior RT boundaries derived from the observed distribution.
        Bins via np.searchsorted on these boundaries gives nbins+1 equal-mass bins.
        """
        if rt.size < self.nbins:
            return np.array([rt.min(), rt.max()])
        return np.quantile(rt, self._q_grid)

    def _rt_quantile_nll(self, rt_data: np.ndarray, rt_pred: np.ndarray) -> float:
        """
        RT NLL term, path selected by predicted sample size.

        KL path (rt_pred.size >= sparse_threshold):
            KL(obs || pred) * n_obs. Laplace smoothing prevents log(0).
            Exactly zero when pred matches obs; positive otherwise.

        MSE fallback (rt_pred.size < sparse_threshold):
            Quantile MSE scaled by rt_weight * n_obs.
        """
        n_bins     = self.nbins + 1
        boundaries = self._data_boundaries(rt_data)

        if rt_pred.size >= self.sparse_threshold:
            obs_counts  = np.bincount(
                np.searchsorted(boundaries, rt_data), minlength=n_bins
            ).astype(float)
            pred_counts = np.bincount(
                np.searchsorted(boundaries, rt_pred), minlength=n_bins
            ).astype(float)
            obs_props  = (obs_counts  + self.eps) / (obs_counts.sum()  + self.eps * n_bins)
            pred_props = (pred_counts + self.eps) / (pred_counts.sum() + self.eps * n_bins)
            return float(np.sum(obs_counts * (np.log(obs_props) - np.log(pred_props))))

        # MSE fallback
        data_q = np.quantile(rt_data, self._mse_grid)
        pred_q = np.quantile(rt_pred, self._mse_grid)
        return self.rt_weight * rt_data.size * float(np.mean((data_q - pred_q) ** 2))

    def calculate_likelihood(
        self,
        rt_pred: np.ndarray,
        choice_pred: np.ndarray,
        rt_data: np.ndarray,
        choice_data: np.ndarray,
        coherences_pred: np.ndarray,
        coherences_data: np.ndarray,
    ) -> float:

        if rt_pred.size == 0 or rt_data.size == 0:
            return 1e6

        vp = ~(np.isnan(rt_pred)   | np.isnan(choice_pred))
        vd = ~(np.isnan(rt_data)   | np.isnan(choice_data))
        if not (vp.any() and vd.any()):
            return 1e6

        rt_pred, choice_pred, coh_pred = rt_pred[vp], choice_pred[vp], coherences_pred[vp]
        rt_data, choice_data, coh_data = rt_data[vd], choice_data[vd], coherences_data[vd]

        total = 0.0
        for coh in np.unique(coh_data):
            dm = coh_data == coh
            pm = coh_pred == coh
            if not (dm.any() and pm.any()):
                continue

            for cv in np.unique(choice_data):
                n_obs = int(np.sum(choice_data[dm] == cv))
                if n_obs == 0:
                    continue

                # Choice NLL -- clipped to prevent near-zero from p_pred ~ 1.
                p_pred = float(np.mean(choice_pred[pm] == cv))
                p_pred = np.clip(p_pred, self.p_min, 1.0 - self.p_min)
                total -= n_obs * np.log(p_pred)

                # RT term
                rt_d = rt_data[dm & (choice_data == cv)]
                rt_p = rt_pred[pm & (choice_pred == cv)]
                if rt_d.size < 3 or rt_p.size < 3:
                    continue
                total += self._rt_quantile_nll(rt_d, rt_p)

        return float(total) if np.isfinite(total) else 1e6

# ---------------------------------------------------------------------------
# Abstract fitting engine
# ---------------------------------------------------------------------------

class DecisionModel:
    """
    Abstract DDM fitting engine.

    This class knows how to run differential evolution and call the simulator,
    but has zero knowledge of parameter names, conditions, or model structure.

    Subclass responsibilities (implemented in ddm_model.py)
    --------------------------------------------------------
    param_specs : property -> dict[str, ParamSpec]
        Ordered mapping of parameter key -> ParamSpec(default, bounds).
        Keys are arbitrary; the base class treats them as opaque identifiers.
        Defines the length and order of the flat optimiser vector.

    _build_params(values, condition_idx=None) -> DDMParams
        Translates a flat optimiser vector into a validated DDMParams.
        condition_idx is an optional integer hint for multi-condition models.

    _objective_function(values, data, stimulus, n_reps, seed, l1_weight) -> float
        Returns negative log-likelihood plus any regularisation.
    """

    def __init__(
        self,
        device: str | None = None,
        likelihood_params: dict | None = None,
    ):
        if device == "cuda":
            self.simulator: DriftDiffusionSimulator | DriftDiffusionSimulatorCUDA = (
                DriftDiffusionSimulatorCUDA(device=device)
            )
        else:
            self.simulator = DriftDiffusionSimulator()

        lp = likelihood_params or {}
        self.likelihood_calc = LikelihoodCalculator(
            nbins=lp.get("nbins", 5),
            rt_weight=lp.get("rt_weight", 1.0),
        )
        logger.info(
            "Initialized %s | device=%s",
            type(self).__name__,
            getattr(self.simulator, "device", "CPU"),
        )

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @property
    def param_specs(self) -> dict[str, ParamSpec]:
        raise NotImplementedError

    def _build_params(
        self, values: np.ndarray, condition_idx: int | None = None
    ) -> DDMParams:
        raise NotImplementedError

    def _objective_function(
        self,
        values: np.ndarray,
        data: dict,
        stimulus: np.ndarray,
        n_reps: int,
        seed: int,
        l1_weight: float,
    ) -> float:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Simulation helper (shared infrastructure)
    # ------------------------------------------------------------------

    def _simulate_condition(
        self,
        stimulus: np.ndarray,
        params: DDMParams,
        n_reps: int,
    ) -> SimResult:
        """
        Run n_reps simulations on stimulus with the given params.

        Returns a SimResult with one of three states:
          crashed=True         -- simulator raised an exception
          no_valid_trials=True -- all reps produced <5 valid trials
          ok                   -- rt / choice / coherence arrays populated
        """
        all_rt, all_choice, all_coh = [], [], []
        for _ in range(n_reps):
            try:
                rt, choice = self.simulator.simulate_trials(stimulus, params)
            except Exception as exc:
                logger.warning("Simulation failed: %s", exc)
                return SimResult(crashed=True)
            valid = (~np.isnan(rt)) & (~np.isnan(choice))
            if valid.sum() < 5:
                continue
            all_rt.append(rt[valid])
            all_choice.append(choice[valid])
            all_coh.append(stimulus[valid, COH_COL])

        if not all_rt:
            return SimResult(no_valid_trials=True)

        return SimResult(
            rt=np.concatenate(all_rt),
            choice=np.concatenate(all_choice),
            coherence=np.concatenate(all_coh),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        data: dict[str, np.ndarray],
        stimulus: np.ndarray,
        max_iterations: int = 100,
        n_reps: int = 5,
        seed: int = 42,
        l1_weight: float = 0.01,
        verbose: bool = True,
    ) -> dict:
        """
        Fit model parameters to empirical data via differential evolution.

        Args:
            data:           Dict with at least 'signed_coherence', 'choice', 'rt'.
                            Multi-condition models may require additional keys.
            stimulus:       (n_trials, n_timepoints) array; column 0 is coherence.
            max_iterations: DE max generations.
            n_reps:         Simulation repetitions per objective evaluation.
            seed:           RNG seed (applied once here, not per evaluation).
            l1_weight:      L1 regularisation coefficient.
            verbose:        Log progress.

        Returns:
            Dict with keys 'success', 'parameters', 'likelihood', 'n_iterations',
            'optimization_result'.
        """
        if verbose:
            logger.info("Starting optimisation (%s)...", type(self).__name__)

        specs  = self.param_specs
        bounds = [s.bounds for s in specs.values()]

        np.random.seed(seed)
        torch.manual_seed(seed)

        best_nll = [np.inf]
        best_vals = [None]

        def objective(values: np.ndarray) -> float:
            nll = self._objective_function(values, data, stimulus, n_reps, seed, l1_weight)
            if nll < best_nll[0]:
                best_nll[0] = nll
                best_vals[0] = values.copy()
            return nll

        iteration = [0]

        def callback(xk: np.ndarray, convergence: float) -> None:
            iteration[0] += 1
            params_str = "  ".join(f"{k}={v:.4f}" for k, v in zip(specs.keys(), best_vals[0]))
            logger.info("Iter %3d | NLL=%.4f | convergence=%.4f | %s", iteration[0], best_nll[0], convergence, params_str)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=max_iterations,
                popsize=10,
                seed=seed,
                polish=True,
                disp=False,
                callback=callback if verbose else None,
            )

        best = dict(zip(specs.keys(), result.x))

        if verbose:
            logger.info("Optimisation done. Cost: %.4f", result.fun)
            for name, val in best.items():
                logger.info("  %s = %.4f", name, val)

        return {
            "success":              result.success,
            "parameters":          best,
            "likelihood":          result.fun,
            "n_iterations":        result.nit,
            "optimization_result": result,
        }

    def simulate(
        self,
        stimulus: np.ndarray,
        params: DDMParams | None = None,
        n_reps: int = 1,
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Forward pass: generate simulated responses for a given stimulus.

        Args:
            stimulus: (n_trials, n_timepoints) array.
            params:   DDMParams to use. Defaults to DDMParams() if None.
            n_reps:   Independent repetitions to concatenate.
            seed:     Optional RNG seed for reproducibility.

        Returns:
            Dict with keys 'rt', 'choice', 'coherence'.
        """
        if params is None:
            params = DDMParams()
        params.validate()

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        all_rt, all_choice = [], []
        for _ in range(n_reps):
            rt, choice = self.simulator.simulate_trials(stimulus, params)
            all_rt.append(rt)
            all_choice.append(choice)

        return {
            "rt":        np.concatenate(all_rt)     if n_reps > 1 else all_rt[0],
            "choice":    np.concatenate(all_choice) if n_reps > 1 else all_choice[0],
            "coherence": np.tile(stimulus[:, COH_COL], n_reps) if n_reps > 1 else stimulus[:, COH_COL],
        }
