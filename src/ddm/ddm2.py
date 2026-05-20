"""
ddm_simple.py — Simplified DDM simulation and fitting engine.

Classes
-------
ParamSpec            (default, bounds) for a free parameter
LikelihoodCalculator quantile-based NLL
DecisionModel        abstract fitting engine (subclass in ddm_model.py)

Parameters are passed as plain dicts. Use DEFAULT_PARAMS as a base and
override keys as needed. validate_params() checks required fields.
"""

from __future__ import annotations
import logging
import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from scipy.optimize import differential_evolution, minimize

from .simulator import DriftDiffusionSimulator
from .likelihood_calculator_working import LikelihoodCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SIGNED_COH_COL : int = 0  # column index of coherence in the stimulus array

REQUIRED = {
    "ndt",
    "a",
    "z",
    "drift_gain",
    "drift_offset",
    "variance",
    "dt",
    "leak_rate",
    "time_constant",
}

# ---------------------------------------------------------------------------
# Parameters as plain dict
# ---------------------------------------------------------------------------

def validate_params(p: dict) -> None:
    for k in REQUIRED:
        if k not in p:
            raise ValueError(f"Missing {k}")
    if not (0 < p["a"] < 10):
        raise ValueError(f"Invalid boundary separation: {p['a']}")
    if not (0 < p["z"] < 1):
        raise ValueError(f"Invalid starting point: {p['z']}")
    if not (0 < p["dt"] < 0.01):
        raise ValueError(f"Invalid time step: {p['dt']}")
    if p["ndt"] < 0:
        raise ValueError(f"Invalid non-decision time: {p['ndt']}")
    if not (0 < p["variance"]):
        raise ValueError(f"Invalid variance: {p['variance']}")

@dataclass
class FreeParam:
    """Default value and (lo, hi) bounds for a single free parameter."""
    lower_bound: float
    upper_bound: float

@dataclass
class FixedParam:
    value: float



# ---------------------------------------------------------------------------
# Abstract fitting engine
# ---------------------------------------------------------------------------

class DecisionModel:
    """
    Abstract DDM fitting engine. Subclass and implement:
      - param_specs property  -> dict[str, ParamSpec]
      - _build_params(values, condition_idx) -> DDMParams
      - _objective_function(values, data, stimulus, n_reps, seed, l1_weight) -> float
    """

    def __init__(
        self,
        free_params: dict[str, FreeParam] | None = None,
        fixed_params: dict[str, FixedParam] | None = None,
        device: str | None = None,
        likelihood_params: dict | None = None,
        seed: int = 42,
    ):
        self.seed = seed
        self._unit_noise: np.ndarray | None = None
        self.simulator = DriftDiffusionSimulator(device=device)

        self._free_params = free_params or {}
        self.fit_keys = list(self._free_params.keys())
        self.fixed_params = fixed_params or {}
        self._validate_configuration()

        lp = likelihood_params or {}
        self.likelihood_calc = LikelihoodCalculator(
            nbins=lp.get("nbins", 5),
            caf_weight=lp.get("caf_weight", 1.0),
            caf_bins=lp.get("caf_bins", 5),
            rt_var_weight=lp.get("rt_var_weight", 1.0),
        )
        logger.info(
            "Initialized %s | device=%s",
            type(self).__name__,
            getattr(self.simulator, "device", "CPU"),
        )

    def _validate_configuration(self):
        for k, v in self._free_params.items():
            if not isinstance(v, FreeParam):
                raise ValueError(
                    f"free_params['{k}'] must be a FreeParam, got {type(v).__name__}. "
                    "Did you swap free_params and fixed_params?"
                )
        for k, v in self.fixed_params.items():
            if not isinstance(v, FixedParam):
                raise ValueError(
                    f"fixed_params['{k}'] must be a FixedParam, got {type(v).__name__}. "
                    "Did you swap free_params and fixed_params?"
                )
        overlap = set(self._free_params) & set(self.fixed_params)
        if overlap:
            raise ValueError(f"Parameters both fixed and fitted: {overlap}")
        missing = REQUIRED - set(self._free_params) - set(self.fixed_params)
        if missing:
            raise ValueError(f"Missing parameters (not fixed or fitted): {missing}")

    @property
    def fit_param_specs(self):
        return self._free_params

    def _build_params(self, values):

        if len(values) != len(self.fit_keys):
            raise ValueError(
                f"Expected {len(self.fit_keys)} values, got {len(values)}"
            )

        p = {}
        for key, val in self.fixed_params.items():
            p[key] = val.value if isinstance(val, FixedParam) else val
        for key, val in zip(self.fit_keys, values):
            p[key] = val

        missing = REQUIRED - set(p.keys())
        if missing:
            raise ValueError(f"Incomplete parameter set: {missing}")

        validate_params(p)

        return p

    def _objective_function(
        self,
        values: np.ndarray,
        data: dict,
        stimulus: np.ndarray,
        n_reps: int,
        l1_weight: float,
    ) -> float:
        raise NotImplementedError

    @staticmethod
    def _set_seed(seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _simulate_condition(
        self,
        stimulus: np.ndarray,
        params: dict,
        n_reps: int,
    ) -> pd.DataFrame | None:
        """Run repeated simulations and concatenate valid trials."""
        simulations = []

        for rep in range(n_reps):
            rep_noise = self._unit_noise[rep] if self._unit_noise is not None else None
            try:
                sim = self.simulator.simulate_trials(stimulus, params, unit_noise=rep_noise)
            except Exception as exc:
                logger.warning("Simulation failed: %s", exc)
                return None

            valid = np.isfinite(sim["rt"]) & np.isfinite(sim["choice"])
            sim = sim.loc[valid]

            if len(sim) < 5:
                continue

            simulations.append(sim)

        if not simulations:
            return None

        return pd.concat(simulations, ignore_index=True)

    def fit(
        self,
        data: dict[str, np.ndarray],
        stimulus: np.ndarray,
        max_iterations: int = 100,
        n_reps: int = 5,
        l1_weight: float = 0.01,
        verbose: bool = True,
    ) -> dict:
        """
        Fit parameters via differential evolution.

        Returns dict with keys: success, parameters, likelihood,
        n_iterations, optimization_result.
        """
        if verbose:
            logger.info("Starting optimization (%s)...", type(self).__name__)

        if not self.fit_keys:
            raise ValueError("No fitted parameters specified.")

        specs  = self.fit_param_specs
        bounds = [(s.lower_bound, s.upper_bound) for s in specs.values()]

        self._set_seed(self.seed)

        # Precompute unit Gaussian noise once so every objective evaluation
        # sees the same random draws (Common Random Numbers).
        n_trials, n_timepoints = stimulus.shape
        self._unit_noise = self.simulator.precompute_noise(n_trials, n_timepoints, n_reps=n_reps)

        best_nll  = [np.inf]
        best_vals = [None]

        def objective(values: np.ndarray) -> float:
            nll = self._objective_function(values, data, stimulus, n_reps, l1_weight)
            if nll < best_nll[0]:
                best_nll[0] = nll
                best_vals[0] = values.copy()
            return nll

        iteration = [0]

        def callback(xk: np.ndarray, convergence: float) -> None:
            iteration[0] += 1
            params_str = "  ".join(f"{k}={v:.4f}" for k, v in zip(self.fit_keys, best_vals[0]))
            logger.info(
                "Iter %3d | NLL=%.4f | convergence=%.4f | %s",
                iteration[0], best_nll[0], convergence, params_str,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            de_result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=max_iterations,
                popsize=10,
                seed=self.seed,
                polish=True,
                disp=False,
                callback=callback if verbose else None,
            )

            result = minimize(
                objective,
                x0=de_result.x,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": max_iterations, "disp": True},
            )

        best = self._build_params(result.x)

        if verbose:
            logger.info("Optimization done. Cost: %.4f", result.fun)
            for name, val in best.items():
                logger.info("  %s = %.4f", name, val)

        if result.success:
            logger.info("Optimization successful.")
            self.fitted_params_ = best
            self.optimization_result_ = result

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
        params: dict | None = None,
        n_reps: int = 1,
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Forward pass: returns dict with rt, choice, coherence arrays."""
        if params is None:
            raise ValueError("Must provide params for simulation.")
        validate_params(params)

        self._set_seed(seed if seed is not None else self.seed)

        all_coherences, all_rt, all_choice = [], [], []
        for _ in range(n_reps):
            simulation = self.simulator.simulate_trials(stimulus, params)
            all_rt.append(simulation["rt"])
            all_choice.append(simulation["choice"])
            all_coherences.append(simulation["signed_coherence"])


        return pd.DataFrame({
            "rt":        np.concatenate(all_rt)     if n_reps > 1 else all_rt[0],
            "choice":    np.concatenate(all_choice) if n_reps > 1 else all_choice[0],
            "signed_coherence": np.concatenate(all_coherences) if n_reps > 1 else all_coherences[0],
        })
