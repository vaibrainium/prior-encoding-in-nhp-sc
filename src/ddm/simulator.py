from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import torch
from numba import jit, prange
from scipy.optimize import differential_evolution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private backends
# ---------------------------------------------------------------------------

@jit(nopython=True, parallel=True)
def _simulate_ddm_trials_numba(
    stimulus: np.ndarray,
    noise: np.ndarray,
    sv_draws: np.ndarray,   # (n_trials,) per-trial drift offset ~ N(0, sv)
    z_offsets: np.ndarray,  # (n_trials,) per-trial z offset ~ Uniform(-sz/2, sz/2)
    drift_gain: float,
    drift_offset: float,
    a: float,
    z: float,
    ndt: float,
    dt: float,
    leak_rate: float,
    time_constant: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_trials, n_timepoints = stimulus.shape
    rt = np.full(n_trials, np.nan, dtype=np.float32)
    choice = np.full(n_trials, np.nan, dtype=np.float32)

    for trial in prange(n_trials):
        z_trial = min(0.99, max(0.01, z + z_offsets[trial]))
        starting_point = z_trial * a
        evidence = starting_point
        trial_drift_offset = drift_offset + sv_draws[trial]

        for t in range(1, n_timepoints):
            if np.isnan(stimulus[trial, t]):
                break

            drift = (drift_gain * stimulus[trial, t - 1] + trial_drift_offset) * dt
            leak  = leak_rate * (evidence - starting_point) * dt
            evidence += drift + noise[trial, t - 1] - leak

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


class _NumbaSimulator:
    """Numba JIT backend — fastest on CPU due to per-trial early termination."""

    def simulate_trials(
        self, stimulus: np.ndarray, params: dict, unit_noise: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if stimulus.size == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        stim = np.asarray(stimulus, dtype=np.float32)
        n_trials, n_timepoints = stim.shape
        noise_std = np.sqrt(params["variance"] * params["dt"])
        if unit_noise is not None:
            noise = (unit_noise * noise_std).astype(np.float32)
        else:
            noise = np.random.normal(0.0, noise_std, (n_trials, n_timepoints - 1)).astype(np.float32)

        sv = float(params.get("sv", 0.0))
        sz = float(params.get("sz", 0.0))
        sv_draws  = np.random.normal(0.0, sv,  n_trials).astype(np.float32) if sv > 0.0 else np.zeros(n_trials, dtype=np.float32)
        z_offsets = (np.random.uniform(-0.5, 0.5, n_trials) * sz).astype(np.float32) if sz > 0.0 else np.zeros(n_trials, dtype=np.float32)

        return _simulate_ddm_trials_numba(
            stim, noise, sv_draws, z_offsets,
            params["drift_gain"], params["drift_offset"],
            params["a"], params["z"], params["ndt"], params["dt"],
            params["leak_rate"], params["time_constant"],
        )


class _TorchSimulator:
    """
    PyTorch GPU/CPU backend.

    No-leak path: fully vectorized via cumsum — no Python loop over timesteps.
    Leak path: closed-form FFT convolution.
    Fastest when CUDA is available; on CPU, _NumbaSimulator wins due to early
    termination (most trials cross long before the stimulus end).
    """

    def __init__(self, device: str | None = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def simulate_trials(
        self, stimulus: np.ndarray | torch.Tensor, params: dict,
        unit_noise: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        dev = self.device
        if isinstance(stimulus, torch.Tensor):
            stim = stimulus.to(dev, dtype=torch.float32)
        else:
            # Cache the GPU tensor: during fitting the same numpy array is passed
            # thousands of times. Re-uploading it on every call is the main GPU
            # bottleneck. We key the cache on the array's data pointer + shape.
            ptr = stimulus.__array_interface__["data"][0]
            key = (ptr, stimulus.shape)
            if getattr(self, "_stim_cache_key", None) != key:
                self._stim_cache_key = key
                self._stim_cache     = torch.tensor(stimulus, device=dev, dtype=torch.float32)
            stim = self._stim_cache

        if stim.numel() == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        n_trials, n_timepoints = stim.shape
        T = n_timepoints - 1

        a             = float(params["a"])
        z             = float(params["z"])
        ndt           = float(params["ndt"])
        drift_gain    = float(params["drift_gain"])
        drift_offset  = float(params["drift_offset"])
        variance      = float(params["variance"])
        dt            = float(params["dt"])
        leak_rate     = float(params["leak_rate"])
        time_constant = float(params["time_constant"])
        sv            = float(params.get("sv", 0.0))
        sz            = float(params.get("sz", 0.0))

        noise_std = (variance * dt) ** 0.5

        valid     = ~torch.isnan(stim[:, 1:])                                          # (N, T)
        drift_inc = torch.nan_to_num((drift_gain * stim[:, :-1] + drift_offset) * dt)  # (N, T)

        # sv: per-trial drift variability — add a constant offset to each trial's drift
        if sv > 0.0:
            sv_draws   = torch.randn(n_trials, 1, device=dev) * sv                    # (N, 1)
            drift_inc  = drift_inc + sv_draws * dt                                    # broadcast over T

        # sz: per-trial starting point variability — Uniform(z - sz/2, z + sz/2)
        if sz > 0.0:
            z_offsets = (torch.rand(n_trials, device=dev) - 0.5) * sz
            z_trials  = torch.clamp(z + z_offsets, 0.01, 0.99)
            sp        = (z_trials * a).unsqueeze(1)                                   # (N, 1)
        else:
            sp = z * a                                                                 # scalar

        if unit_noise is not None:
            # Cache the per-rep noise slice on GPU to avoid repeated CPU→GPU transfers.
            # unit_noise is a 2-D view into the precomputed array; the data pointer is
            # stable across all optimizer iterations so we get a cache hit after rep 1.
            ptr = unit_noise.__array_interface__["data"][0]
            nkey = (ptr, unit_noise.shape)
            if getattr(self, "_noise_cache_key", None) != nkey:
                self._noise_cache_key = nkey
                self._noise_cache = torch.tensor(unit_noise, device=dev, dtype=torch.float32)
            noise_inc = self._noise_cache * noise_std                                  # (N, T)
        else:
            noise_inc = torch.randn(n_trials, T, device=dev) * noise_std              # (N, T)

        if leak_rate == 0.0:
            evidence = sp + torch.cumsum((drift_inc + noise_inc) * valid, dim=1)
        else:
            # ------------------------------------------------------------------
            # Leak: e[t] = α·e[t-1] + b[t],  α = 1 − leak·dt
            #
            # Closed-form solution via FFT convolution (no Python time loop):
            #   e[t] = αᵗ·sp  +  Σ_{j=0}^{t-1} α^{t-1-j} · b[j]
            #        = αᵗ·sp  +  (b ★ w)[t-1]   where w[k] = αᵏ
            # ------------------------------------------------------------------
            α   = 1.0 - leak_rate * dt
            b   = (drift_inc + noise_inc + sp * leak_rate * dt) * valid       # (N, T)

            fft_n = 1 << (2 * T - 1).bit_length()   # next power-of-2 ≥ 2T-1
            k     = torch.arange(T, device=dev, dtype=torch.float32)
            w     = torch.tensor(α, device=dev, dtype=torch.float32).pow(k)   # (T,)

            particular = torch.fft.irfft(
                torch.fft.rfft(b, n=fft_n, dim=1) * torch.fft.rfft(w, n=fft_n),
                n=fft_n, dim=1,
            )[:, :T]                                                           # (N, T)

            t_exp    = torch.arange(1, T + 1, device=dev, dtype=torch.float32)
            evidence = torch.tensor(α, device=dev).pow(t_exp) * sp + particular  # (N, T)

        if time_constant != 0.0:
            t_idx        = torch.arange(1, n_timepoints, device=dev, dtype=torch.float32)
            decision_var = sp + (evidence - sp) * (1.0 + t_idx * dt * time_constant)
        else:
            decision_var = evidence

        upper   = (decision_var >= a) & valid
        lower   = (decision_var <= 0) & valid
        crossed = upper | lower

        rt     = torch.full((n_trials,), float("nan"), device=dev, dtype=torch.float32)
        choice = torch.full((n_trials,), float("nan"), device=dev, dtype=torch.float32)

        has_crossed = crossed.any(dim=1)
        if has_crossed.any():
            first_idx       = crossed[has_crossed].int().argmax(dim=1)
            rt[has_crossed] = (first_idx.float() + 1) * dt + ndt
            choice[has_crossed] = (
                upper[has_crossed]
                .gather(1, first_idx.unsqueeze(1))
                .squeeze(1)
                .float()
            )

        return rt.cpu().numpy(), choice.cpu().numpy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DriftDiffusionSimulator:
    """
    Auto-selecting DDM simulator.

    Uses the PyTorch GPU backend when CUDA is available (vectorized, no
    Python loop over timesteps). Falls back to the Numba JIT backend on
    CPU-only machines, which is faster there due to per-trial early
    termination.

    Pass device="cuda" to force GPU (warns and falls back if unavailable).
    Pass device="cpu" to force Numba regardless of GPU availability.
    """

    def __init__(self, device: str | None = None):
        if device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but unavailable, falling back to Numba CPU.")
                self._backend = _NumbaSimulator()
                self.device = torch.device("cpu")
            else:
                self._backend = _TorchSimulator("cuda")
                self.device = self._backend.device
        elif torch.cuda.is_available() and device != "cpu":
            self._backend = _TorchSimulator()
            self.device = self._backend.device
        else:
            self._backend = _NumbaSimulator()
            self.device = torch.device("cpu")

        logger.info(
            "DriftDiffusionSimulator using %s backend on %s",
            type(self._backend).__name__,
            self.device,
        )

    def precompute_noise(
        self,
        n_trials: int,
        n_timepoints: int,
        n_reps: int = 1,
    ) -> np.ndarray:
        """Return unit Gaussian noise of shape (n_reps, n_trials, n_timepoints-1).

        Call this once before an optimization run (after setting the global seed)
        and pass the result to simulate_trials via unit_noise to enable CRN.
        """
        return np.random.standard_normal(
            (n_reps, n_trials, n_timepoints - 1)
        ).astype(np.float32)

    def simulate_trials(
        self, stimulus: np.ndarray | torch.Tensor, params: dict,
        unit_noise: np.ndarray | None = None,
    ) -> pd.DataFrame:
        rt, choice = self._backend.simulate_trials(stimulus, params, unit_noise=unit_noise)

        return pd.DataFrame({"signed_coherence": stimulus[:, 0], "rt": rt, "choice": choice})
