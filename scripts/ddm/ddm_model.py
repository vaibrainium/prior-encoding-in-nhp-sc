"""
ddm_model.py

DecisionModel base class and two concrete variants:

  AllParamsModel   -- all parameters (a, z, drift_offset, ndt, drift_gain,
                      variance, leak_rate, time_constant) optimised freely
                      for a single condition.

  ThreeParamsModel -- a, z, drift_offset vary per prior condition
                      (equal / unequal); ndt, drift_gain, variance,
                      leak_rate, time_constant are shared across conditions.

Usage
-----
    from ddm_model import AllParamsModel, ThreeParamsModel

    model = AllParamsModel(device="cuda")
    result = model.fit(data, stimulus, max_iterations=500)

    model = ThreeParamsModel(device="cuda")
    result = model.fit(data, stimulus, max_iterations=500)
        # data must contain "prior_block" column with values "equal"/"unequal"
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import torch
from scipy.optimize import differential_evolution

from ddm_base import (
    DriftDiffusionSimulator,
    DriftDiffusionSimulatorCUDA,
    LikelihoodCalculator,
)

logger = logging.getLogger(__name__)

PRIOR_CONDITIONS = ("equal", "unequal")


# ---------------------------------------------------------------------------
# Base class — common __init__, fit, simulate
# ---------------------------------------------------------------------------

class DecisionModel:
    """
    Abstract base class for DDM variants.

    Subclasses must implement:
      - _setup_parameter_bounds() -> dict[str, tuple[default, (lo, hi)]]
      - _update_parameters(params, prior_idx=None)
      - _objective_function(params, data, stimulus, n_reps, seed, l1_weight)
    """

    def __init__(
        self,
        model_name: str = "DDM",
        enable_leak: bool = True,
        enable_time_dependence: bool = True,
        device: str | None = None,
        likelihood_params: dict | None = None,
    ):
        if model_name.upper() != "DDM":
            raise NotImplementedError(f"Model '{model_name}' not implemented.")

        self.model_name = model_name.upper()
        self.enable_leak = enable_leak
        self.enable_time_dependence = enable_time_dependence

        if device == "cuda":
            self.simulator = DriftDiffusionSimulatorCUDA(
                leak=enable_leak, time_dependence=enable_time_dependence, device=device
            )
        else:
            self.simulator = DriftDiffusionSimulator(
                leak=enable_leak, time_dependence=enable_time_dependence
            )

        lp = likelihood_params or {}
        self.likelihood_calc = LikelihoodCalculator(
            nbins=lp.get("nbins", 5),
            rt_weight=lp.get("rt_weight", 1.0),
        )

        self._param_bounds = self._setup_parameter_bounds()
        logger.info(
            "Initialized %s (%s) | device=%s | leak=%s | time_dep=%s",
            type(self).__name__, self.model_name,
            getattr(self.simulator, "device", "CPU"),
            enable_leak, enable_time_dependence,
        )

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def _setup_parameter_bounds(self) -> dict[str, tuple[float, tuple[float, float]]]:
        raise NotImplementedError

    def _update_parameters(self, params: np.ndarray, prior_idx: int | None = None):
        raise NotImplementedError

    def _objective_function(
        self,
        params: np.ndarray,
        data: dict,
        stimulus: np.ndarray,
        n_reps: int,
        seed: int,
        l1_weight: float,
    ) -> float:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _simulate_condition(
        self,
        stimulus: np.ndarray,
        n_reps: int,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray] | None, bool]:
        """
        Run n_reps simulations on stimulus, concatenate valid trials.

        Returns:
            (result, crashed) where:
              result  -- (rt, choice, coherence) arrays on success, None otherwise
              crashed -- True if a simulation exception occurred (→ caller returns 1e6)
                         False if trials simply produced no valid responses (→ caller skips)
        """
        all_rt, all_choice, all_coh = [], [], []
        for _ in range(n_reps):
            try:
                rt, choice, _ = self.simulator.simulate_trials(stimulus)
            except Exception as e:
                logger.warning("Simulation failed: %s", e)
                return None, True
            valid = (~np.isnan(rt)) & (~np.isnan(choice))
            if valid.sum() < 5:
                continue
            all_rt.append(rt[valid])
            all_choice.append(choice[valid])
            all_coh.append(stimulus[valid, 0])

        if not all_rt:
            return None, False
        return (
            np.concatenate(all_rt),
            np.concatenate(all_choice),
            np.concatenate(all_coh),
        ), False

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
            data: Dict with keys 'signed_coherence', 'choice', 'rt', and
                optionally 'prior_block' (values 'equal'/'unequal') for
                ThreeParamsModel.
            stimulus: (n_trials, n_timepoints) array — first column is coherence.
            max_iterations: DE max generations.
            n_reps: Simulation repetitions per objective evaluation.
            seed: RNG seed.
            l1_weight: L1 regularisation coefficient.
            verbose: Log progress.

        Returns:
            Dict with keys 'success', 'parameters', 'likelihood', 'n_iterations'.
        """
        if verbose:
            logger.info("Starting parameter optimisation (%s)...", type(self).__name__)

        bounds = [self._param_bounds[n][1] for n in self._param_bounds]
        np.random.seed(seed)

        def objective(params: np.ndarray) -> float:
            return self._objective_function(params, data, stimulus, n_reps, seed, l1_weight)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=max_iterations,
                popsize=10,
                seed=seed,
                polish=True,
                disp=verbose,
            )

        best = dict(zip(self._param_bounds.keys(), result.x))

        if verbose:
            logger.info("Optimisation done. Cost: %.4f", result.fun)
            for name, val in best.items():
                logger.info("  %s = %.4f", name, val)

        return {
            "success": result.success,
            "parameters": best,
            "likelihood": result.fun,
            "n_iterations": result.nit,
            "optimization_result": result,
        }

    def simulate(self, stimulus: np.ndarray, n_reps: int = 1) -> dict[str, np.ndarray]:
        """
        Generate simulated responses for a given stimulus.

        Args:
            stimulus: (n_trials, n_timepoints) array.
            n_reps: Independent repeats to concatenate.

        Returns:
            Dict with 'rt', 'choice', 'coherence'.
        """
        all_rt, all_choice = [], []
        for _ in range(n_reps):
            rt, choice, _ = self.simulator.simulate_trials(stimulus)
            all_rt.append(rt)
            all_choice.append(choice)

        return {
            "rt": np.concatenate(all_rt) if n_reps > 1 else all_rt[0],
            "choice": np.concatenate(all_choice) if n_reps > 1 else all_choice[0],
            "coherence": np.tile(stimulus[:, 0], n_reps) if n_reps > 1 else stimulus[:, 0],
        }


# ---------------------------------------------------------------------------
# AllParamsModel — single condition, all parameters free
# ---------------------------------------------------------------------------

class AllParamsModel(DecisionModel):
    """
    All DDM parameters optimised freely for a single condition.

    Free parameters: a, z, drift_offset, ndt, drift_gain, variance,
                     [leak_rate], [time_constant]
    """

    def _setup_parameter_bounds(self) -> dict[str, tuple[float, tuple[float, float]]]:
        bounds: dict[str, tuple[float, tuple[float, float]]] = {
            "a":            (2.0,  (0.8,   6.0)),
            "z":            (0.5,  (0.1,   0.9)),
            "drift_offset": (0.0,  (-5.0,  5.0)),
            "ndt":          (0.2,  (0.1,   0.5)),
            "drift_gain":   (7.0,  (1.0,  20.0)),
            "variance":     (1.0,  (0.1,   5.0)),
        }
        if self.enable_leak:
            bounds["leak_rate"] = (0.1, (0.0, 1.0))
        if self.enable_time_dependence:
            bounds["time_constant"] = (0.0, (-5.0, 5.0))
        return bounds

    def _update_parameters(self, params: np.ndarray, prior_idx: int | None = None):
        """Set all simulator attributes directly from the flat params vector."""
        for name, value in zip(self._param_bounds.keys(), params):
            setattr(self.simulator, name, float(value))

    def _objective_function(
        self,
        params: np.ndarray,
        data: dict,
        stimulus: np.ndarray,
        n_reps: int = 5,
        seed: int = 42,
        l1_weight: float = 0.01,
    ) -> float:
        np.random.seed(seed)
        torch.manual_seed(seed)

        self._update_parameters(params)
        result, _ = self._simulate_condition(stimulus, n_reps)
        if result is None:   # crashed or no valid trials — both penalised
            return 1e6

        rt_p, ch_p, coh_p = result
        nllh = self.likelihood_calc.calculate_likelihood(
            rt_p, ch_p,
            data["rt"], data["choice"],
            coh_p, data["signed_coherence"],
        )
        nllh += l1_weight * float(np.sum(np.abs(params)))
        return float(nllh) if np.isfinite(nllh) else 1e6


# ---------------------------------------------------------------------------
# ThreeParamsModel — a / z / drift_offset vary per prior condition
# ---------------------------------------------------------------------------

class ThreeParamsModel(DecisionModel):
    """
    a, z, and drift_offset each have separate values for the equal-prior and
    unequal-prior conditions (suffixed _1 and _2 respectively).
    All other parameters (ndt, drift_gain, variance, leak_rate, time_constant)
    are shared across conditions.

    Expects data["prior_block"] with string values "equal" and "unequal".
    """

    def _setup_parameter_bounds(self) -> dict[str, tuple[float, tuple[float, float]]]:
        bounds: dict[str, tuple[float, tuple[float, float]]] = {
            "ndt":            (0.2, (0.1,  0.5)),
            "drift_gain":     (7.0, (1.0, 20.0)),
            "variance":       (1.0, (0.1,  5.0)),
            # Equal-prior condition
            "a_1":            (2.0, (0.8,  6.0)),
            "z_1":            (0.5, (0.1,  0.9)),
            "drift_offset_1": (0.0, (-5.0, 5.0)),
            # Unequal-prior condition
            "a_2":            (2.0, (0.8,  6.0)),
            "z_2":            (0.5, (0.1,  0.9)),
            "drift_offset_2": (0.0, (-5.0, 5.0)),
        }
        if self.enable_leak:
            bounds["leak_rate"] = (0.1, (0.0, 1.0))
        if self.enable_time_dependence:
            bounds["time_constant"] = (0.0, (-5.0, 5.0))
        return bounds

    def _update_parameters(self, params: np.ndarray, prior_idx: int | None = None):
        """
        Apply params to the simulator.

        When prior_idx is given (0 = equal, 1 = unequal):
          - Parameters ending in _{prior_idx+1} are stripped of their suffix
            and applied (e.g. drift_offset_1 → simulator.drift_offset).
          - Global parameters (no numeric suffix) are always applied.
          - Parameters for the other condition are skipped.

        When prior_idx is None (not used in practice for this model):
          - Suffixed params are applied under their base name using condition 1.
          - Global params are applied directly.
        """
        suffix = f"_{prior_idx + 1}" if prior_idx is not None else "_1"

        for name, value in zip(self._param_bounds.keys(), params):
            if name[-1].isdigit():
                # Per-condition parameter: apply only if suffix matches
                if name.endswith(suffix):
                    base_name = name.rsplit("_", 1)[0]
                    setattr(self.simulator, base_name, float(value))
            else:
                # Global parameter: always apply
                setattr(self.simulator, name, float(value))

    def _objective_function(
        self,
        params: np.ndarray,
        data: dict,
        stimulus: np.ndarray,
        n_reps: int = 5,
        seed: int = 42,
        l1_weight: float = 0.01,
    ) -> float:
        np.random.seed(seed)
        torch.manual_seed(seed)

        has_prior = "prior_block" in data and np.unique(data["prior_block"]).size > 1
        if not has_prior:
            logger.warning(
                "ThreeParamsModel expects dual-prior data with 'prior_block' column. "
                "Falling back to single-condition evaluation using equal-prior params."
            )
            self._update_parameters(params, prior_idx=0)
            result, _ = self._simulate_condition(stimulus, n_reps)
            if result is None:
                return 1e6
            rt_p, ch_p, coh_p = result
            nllh = self.likelihood_calc.calculate_likelihood(
                rt_p, ch_p,
                data["rt"], data["choice"],
                coh_p, data["signed_coherence"],
            )
            nllh += l1_weight * float(np.sum(np.abs(params)))
            return float(nllh) if np.isfinite(nllh) else 1e6

        total_nllh = 0.0
        for idx, condition in enumerate(PRIOR_CONDITIONS):
            mask = data["prior_block"] == condition
            if not mask.any():
                continue

            self._update_parameters(params, prior_idx=idx)
            result, crashed = self._simulate_condition(stimulus[mask], n_reps)
            if crashed:
                return 1e6     # simulation exception → penalise
            if result is None:
                continue       # no valid trials for this condition → skip

            rt_p, ch_p, coh_p = result
            cond_data = {k: v[mask] for k, v in data.items() if k != "prior_block"}
            total_nllh += self.likelihood_calc.calculate_likelihood(
                rt_p, ch_p,
                cond_data["rt"], cond_data["choice"],
                coh_p, cond_data["signed_coherence"],
            )

        total_nllh += l1_weight * float(np.sum(np.abs(params)))
        return float(total_nllh) if np.isfinite(total_nllh) else 1e6


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    stim = rng.standard_normal((200, 500)).astype(np.float32) * 0.1

    model_all = AllParamsModel(device=None)
    out = model_all.simulate(stim, n_reps=2)
    logger.info("AllParamsModel smoke test: %d valid trials", int((~np.isnan(out["rt"])).sum()))

    # Dual-prior data for ThreeParamsModel
    blocks = np.array(["equal"] * 100 + ["unequal"] * 100)
    data_dummy = {
        "signed_coherence": stim[:, 0],
        "choice": rng.integers(0, 2, 200).astype(float),
        "rt": rng.uniform(0.3, 1.5, 200),
        "prior_block": blocks,
    }
    model_three = ThreeParamsModel(device=None)
    out3 = model_three.simulate(stim, n_reps=1)
    logger.info("ThreeParamsModel smoke test: %d valid trials", int((~np.isnan(out3["rt"])).sum()))

    logger.info("Done.")