"""
ddm_base.py

Shared infrastructure for all DDM model variants:
  - Data-extraction helpers (psychometric, chronometric)
  - Numba (CPU) and PyTorch (CUDA/CPU) simulators
  - Quantile-based likelihood calculator
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from numba import jit, prange
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def get_psychometric_data(
    data: dict[str, np.ndarray],
    positive_direction: str = "right",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (coherences, proportion_positive_choice) aligned arrays.

    NaN choices are excluded trial-wise before computing proportions.
    """
    unique_coh = np.unique(data["signed_coherence"])
    x_data = np.where(positive_direction == "left", -unique_coh, unique_coh)

    valid_mask = ~np.isnan(data["choice"])
    if not np.any(valid_mask):
        return np.array([]), np.array([])

    y_data = []
    for coh in unique_coh:
        mask = (data["signed_coherence"] == coh) & valid_mask
        if not mask.any():
            continue
        y_data.append(np.mean(data["choice"][mask] == (positive_direction == "right")))

    return np.array(x_data[: len(y_data)]), np.array(y_data)


def get_chronometric_data(
    data: dict[str, np.ndarray],
    positive_direction: str = "right",
) -> tuple[np.ndarray, ...]:
    """
    Return (coherences, rt_median, rt_mean, rt_std, rt_sem) sorted by coherence.
    """
    unique_coh = np.unique(data["signed_coherence"])
    rows = []
    for coh in unique_coh:
        rts = data["rt"][data["signed_coherence"] == coh]
        rts = rts[~np.isnan(rts)]
        if rts.size == 0:
            continue
        c = -coh if positive_direction == "left" else coh
        rows.append((c, np.median(rts), np.mean(rts), np.std(rts), stats.sem(rts)))

    if not rows:
        return tuple(np.array([]) for _ in range(5))  # type: ignore[return-value]

    rows.sort(key=lambda r: r[0])
    return tuple(map(np.array, zip(*rows)))  # type: ignore[return-value]


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
    """Numba-accelerated DDM simulation for CPU parallelism."""
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
            leak = leak_rate * (evidence - starting_point) * dt
            evidence += drift + noise - leak

            # Urgency: t * dt converts step index to seconds; time_constant in 1/s
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

    def __init__(self, leak: bool = True, time_dependence: bool = True):
        self.ndt = 0.1
        self.a = 2.0
        self.z = 0.5
        self.drift_gain = 7.0
        self.drift_offset = 0.0
        self.variance = 1.0
        self.dt = 0.001
        self.leak_rate = 0.01 if leak else 0.0
        self.time_constant = 0.0
        self._validate_parameters()

    def _validate_parameters(self):
        assert 0 < self.a < 10, f"Invalid boundary separation: {self.a}"
        assert 0 < self.z < 1, f"Invalid starting point: {self.z}"
        assert 0 < self.dt < 0.01, f"Invalid time step: {self.dt}"
        assert self.ndt >= 0, f"Invalid non-decision time: {self.ndt}"

    def simulate_trials(self, stimulus: np.ndarray) -> tuple[np.ndarray, np.ndarray, None]:
        if stimulus.size == 0:
            return np.array([]), np.array([]), None
        stim = np.asarray(stimulus, dtype=np.float32)
        rt, choice = _simulate_ddm_trials_numba(
            stim,
            self.drift_gain, self.drift_offset,
            self.a, self.z, self.ndt, self.dt,
            self.variance, self.leak_rate, self.time_constant,
        )
        return rt, choice, None


# ---------------------------------------------------------------------------
# PyTorch (CUDA/CPU) simulator
# ---------------------------------------------------------------------------

class DriftDiffusionSimulatorCUDA:
    """DDM simulator using PyTorch for vectorised GPU (or CPU) computation."""

    def __init__(self, leak: bool = True, time_dependence: bool = True, device: str | None = None):
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable, falling back to CPU.")
            device = "cpu"
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"Using device: {self.device}")

        self.ndt = torch.tensor(0.1, device=self.device, dtype=torch.float32)
        self.a = torch.tensor(2.0, device=self.device, dtype=torch.float32)
        self.z = torch.tensor(0.5, device=self.device, dtype=torch.float32)
        self.drift_gain = torch.tensor(7.0, device=self.device, dtype=torch.float32)
        self.drift_offset = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.variance = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        self.dt = torch.tensor(0.001, device=self.device, dtype=torch.float32)
        self.leak_rate = torch.tensor(0.01 if leak else 0.0, device=self.device, dtype=torch.float32)
        self.time_constant = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        self._noise_std = torch.sqrt(self.variance * self.dt)

    def simulate_trials(
        self, stimulus: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(stimulus, torch.Tensor):
            stim = torch.tensor(stimulus, device=self.device, dtype=torch.float32)
        else:
            stim = stimulus.to(self.device, dtype=torch.float32)

        n_trials, n_timepoints = stim.shape
        starting_point = self.z * self.a

        rt = torch.full((n_trials,), float("nan"), device=self.device, dtype=torch.float32)
        choice = torch.full((n_trials,), float("nan"), device=self.device, dtype=torch.float32)
        evidence = torch.full((n_trials,), starting_point, device=self.device, dtype=torch.float32)
        active = torch.ones(n_trials, dtype=torch.bool, device=self.device)

        drift_rates = self.drift_gain * stim + self.drift_offset

        for t in range(1, n_timepoints):
            if not active.any():
                break

            active_idx = active.nonzero(as_tuple=True)[0]
            valid_stim = ~torch.isnan(stim[active_idx, t])
            if not valid_stim.any():
                continue
            active_idx = active_idx[valid_stim]
            if len(active_idx) == 0:
                continue

            noise = torch.randn(len(active_idx), device=self.device, dtype=torch.float32) * self._noise_std
            drift = drift_rates[active_idx, t - 1] * self.dt
            leak = self.leak_rate * (evidence[active_idx] - starting_point) * self.dt
            evidence[active_idx] += drift + noise - leak

            # Urgency: t * dt converts step index to seconds; time_constant in 1/s
            if self.time_constant != 0:
                urgency_factor = 1 + t * self.dt * self.time_constant
                decision_var = starting_point + (evidence[active_idx] - starting_point) * urgency_factor
            else:
                decision_var = evidence[active_idx]

            hit_upper = decision_var >= self.a
            hit_lower = decision_var <= 0
            crossed = hit_upper | hit_lower

            if crossed.any():
                cidx = active_idx[crossed]
                rt[cidx] = t * self.dt + self.ndt
                choice[cidx] = hit_upper[crossed].float()
                active[cidx] = False

        return rt.cpu().numpy(), choice.cpu().numpy(), evidence.cpu().numpy()


# ---------------------------------------------------------------------------
# Likelihood calculator
# ---------------------------------------------------------------------------

class LikelihoodCalculator:
    """
    Quantile-based negative log-likelihood combining choice proportions and RT.

    RT term: penalises deviation between observed and predicted RT quantiles,
    scaled by rt_weight * n_obs.  Both data and predicted distributions are
    evaluated at the same linspace(0.1, 0.9, nbins) quantile grid.
    """

    def __init__(self, nbins: int = 5, rt_weight: float = 1.0):
        self.nbins = nbins
        self.rt_weight = rt_weight
        self.eps = 1e-12
        self._cache: dict = {}
        self._cache_limit = 100

    def _get_quantiles(self, rt_data: np.ndarray) -> np.ndarray:
        if rt_data.size < self.nbins:
            return np.array([rt_data.min(), rt_data.max()])
        cache_key = (rt_data.size, float(rt_data.min()), float(rt_data.max()), float(rt_data.mean()))
        if cache_key in self._cache:
            return self._cache[cache_key]
        values = np.quantile(rt_data, np.linspace(0.1, 0.9, self.nbins))
        if len(self._cache) >= self._cache_limit:
            self._cache.clear()
        self._cache[cache_key] = values
        return values

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

        vp = ~(np.isnan(rt_pred) | np.isnan(choice_pred))
        vd = ~(np.isnan(rt_data) | np.isnan(choice_data))
        if not (vp.any() and vd.any()):
            return 1e6

        rt_pred, choice_pred, coh_pred = rt_pred[vp], choice_pred[vp], coherences_pred[vp]
        rt_data, choice_data, coh_data = rt_data[vd], choice_data[vd], coherences_data[vd]

        total = 0.0
        unique_choices = np.unique(choice_data)

        for coh in np.unique(coh_data):
            dm = coh_data == coh
            pm = coh_pred == coh
            if not (dm.any() and pm.any()):
                continue

            for cv in unique_choices:
                # Choice NLL
                n_obs = int(np.sum(choice_data[dm] == cv))
                if n_obs == 0:
                    continue
                p_pred = float(np.mean(choice_pred[pm] == cv))
                total -= n_obs * np.log(p_pred + self.eps)

                # RT quantile MSE
                rt_d = rt_data[dm & (choice_data == cv)]
                rt_p = rt_pred[pm & (choice_pred == cv)]
                if rt_d.size < 3 or rt_p.size < 3:
                    continue
                data_q = self._get_quantiles(rt_d)
                pred_q = np.quantile(rt_p, np.linspace(0.1, 0.9, len(data_q)))
                total += float(np.mean((data_q - pred_q) ** 2)) * self.rt_weight * rt_d.size

        return float(total) if np.isfinite(total) else 1e6