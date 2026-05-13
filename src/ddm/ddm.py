"""
ddm_simple.py — Simplified DDM simulation and fitting engine.

Classes
-------
ParamSpec                   (default, bounds) for a free parameter
DriftDiffusionSimulator     CPU backend (Numba)
DriftDiffusionSimulatorCUDA GPU/CPU backend (PyTorch)
LikelihoodCalculator        quantile-based NLL
DecisionModel               abstract fitting engine (subclass in ddm_model.py)

Parameters are passed as plain dicts. Use DEFAULT_PARAMS as a base and
override keys as needed. validate_params() checks required fields.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass  # still used by ParamSpec

import numpy as np
import torch
from numba import jit, prange
from scipy.optimize import differential_evolution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COH_COL: int = 0  # column index of coherence in the stimulus array


# ---------------------------------------------------------------------------
# Parameters as plain dict
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: dict = {
    "ndt":          0.1,
    "a":            2.0,
    "z":            0.5,
    "drift_gain":   7.0,
    "drift_offset": 0.0,
    "variance":     1.0,
    "dt":           0.001,
    "leak_rate":    0.0,
    "time_constant": 0.0,
}


def validate_params(p: dict) -> None:
    if not (0 < p["a"] < 10):
        raise ValueError(f"Invalid boundary separation: {p['a']}")
    if not (0 < p["z"] < 1):
        raise ValueError(f"Invalid starting point: {p['z']}")
    if not (0 < p["dt"] < 0.01):
        raise ValueError(f"Invalid time step: {p['dt']}")
    if p["ndt"] < 0:
        raise ValueError(f"Invalid non-decision time: {p['ndt']}")


@dataclass
class ParamSpec:
    """Default value and (lo, hi) bounds for a single free parameter."""
    default: float
    bounds: tuple[float, float]


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
    """Numba CPU simulator."""

    def simulate_trials(
        self, stimulus: np.ndarray, params: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        if stimulus.size == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        stim = np.asarray(stimulus, dtype=np.float32)
        return _simulate_ddm_trials_numba(
            stim,
            params["drift_gain"], params["drift_offset"],
            params["a"], params["z"], params["ndt"], params["dt"],
            params["variance"], params["leak_rate"], params["time_constant"],
        )


# ---------------------------------------------------------------------------
# PyTorch (CUDA/CPU) simulator
# ---------------------------------------------------------------------------

class DriftDiffusionSimulatorCUDA:
    """
    PyTorch GPU/CPU simulator.

    No-leak path: fully vectorized — cumsum builds the entire evidence
    trajectory in one shot, argmax finds the first boundary crossing.
    No Python loop over timesteps.

    Leak path: falls back to the sequential loop (leak couples timesteps,
    breaking the cumsum decomposition).
    """

    def __init__(self, device: str | None = None):
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable, falling back to CPU.")
            device = "cpu"
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info("Using device: %s", self.device)

    def simulate_trials(
        self, stimulus: np.ndarray | torch.Tensor, params: dict
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

        sp        = z * a
        noise_std = (variance * dt) ** 0.5

        # valid[i]: step t=i+1 is live (stim not NaN)
        valid     = ~torch.isnan(stim[:, 1:])                                 # (N, T)
        drift_inc = torch.nan_to_num((drift_gain * stim[:, :-1] + drift_offset) * dt)  # (N, T)
        noise_inc = torch.randn(n_trials, T, device=dev) * noise_std          # (N, T)

        if leak_rate == 0.0:
            # ------------------------------------------------------------------
            # No-leak: evidence is a simple cumsum of increments
            # ------------------------------------------------------------------
            evidence = sp + torch.cumsum((drift_inc + noise_inc) * valid, dim=1)
        else:
            # ------------------------------------------------------------------
            # Leak: e[t] = α·e[t-1] + b[t],  α = 1 − leak·dt
            #
            # Closed-form solution via FFT convolution (no Python time loop):
            #   e[t] = αᵗ·sp  +  Σ_{j=0}^{t-1} α^{t-1-j} · b[j]
            #        = αᵗ·sp  +  (b ★ w)[t-1]   where w[k] = αᵏ
            #
            # Crossing detection uses valid mask so the decaying evidence
            # after a NaN stimulus never triggers a false crossing.
            # ------------------------------------------------------------------
            α   = 1.0 - leak_rate * dt
            # forcing term includes the leak's pull toward sp
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
# Likelihood calculator
# ---------------------------------------------------------------------------
class LikelihoodCalculator:
    """
    Joint multinomial NLL (KL divergence over choice × RT bins per coherence)
    plus an optional Conditional Accuracy Function (CAF) term.

    CAF term: pools all non-zero coherence trials, bins by RT quantile,
    computes accuracy per bin, and penalises MSE between data and prediction.
    This directly constrains leak vs urgency, which are otherwise hard to
    distinguish from per-coherence summaries alone.
    """

    def __init__(
        self,
        nbins: int = 5,
        p_min: float = 1e-6,
        caf_weight: float = 1.0,
        caf_bins: int = 5,
        rt_var_weight: float = 0.0,   # <-- add this
        rt_weight=None,           # legacy (ignored)
        sparse_threshold=None,    # legacy (ignored)
        **kwargs
    ):
        self.nbins      = nbins
        self.p_min      = p_min
        self.caf_weight = caf_weight
        self.caf_bins   = caf_bins
        self.rt_var_weight = rt_var_weight
        self.eps        = 1e-12
        self._q_grid    = np.linspace(1/(nbins+1), nbins/(nbins+1), nbins)

        if rt_weight is not None or sparse_threshold is not None or kwargs:
            warnings.warn(
                "Deprecated likelihood parameters (rt_weight, sparse_threshold, etc.) "
                "are ignored in JointLikelihoodCalculator.",
                RuntimeWarning
            )

    # ----------------------------
    # Helpers
    # ----------------------------
    def _valid_mask(self, rt, choice):
        return np.isfinite(rt) & np.isfinite(choice)

    def _caf_nll(self, rt_data, ch_data, coh_data, rt_pred, ch_pred, coh_pred):
        """
        MSE between data and predicted CAF, scaled by n_obs × caf_weight.

        RT quantile boundaries are derived from the data and applied to both
        data and predictions, so the bins are always comparable.
        Coherence=0 trials are excluded (no ground-truth correct answer).
        """
        def _filter(rt, ch, coh):
            mask = (np.round(coh, 2) != 0) & np.isfinite(rt) & np.isfinite(ch)
            return rt[mask], ch[mask], coh[mask]

        rt_d, ch_d, coh_d = _filter(rt_data, ch_data, coh_data)
        rt_p, ch_p, coh_p = _filter(rt_pred,  ch_pred,  coh_pred)

        if len(rt_d) < self.caf_bins or len(rt_p) < self.caf_bins:
            return 0.0

        # correct = choice aligned with coherence sign
        correct_d = ((coh_d > 0) & (ch_d == 1)) | ((coh_d < 0) & (ch_d == 0))
        correct_p = ((coh_p > 0) & (ch_p == 1)) | ((coh_p < 0) & (ch_p == 0))

        # data-derived RT quantile boundaries (equal-mass bins under data)
        boundaries = np.quantile(rt_d, np.linspace(0, 1, self.caf_bins + 1))
        boundaries[-1] += 1e-9  # include max RT

        caf_d, caf_p = [], []
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            bd = (rt_d >= lo) & (rt_d < hi)
            bp = (rt_p >= lo) & (rt_p < hi)
            if not bd.any() or not bp.any():
                continue
            caf_d.append(correct_d[bd].mean())
            caf_p.append(correct_p[bp].mean())

        if not caf_d:
            return 0.0

        mse = float(np.mean((np.array(caf_d) - np.array(caf_p)) ** 2))
        return self.caf_weight * len(rt_d) * mse

    def _joint_counts(self, rt, choice, boundaries, choice_vals):
        n_rt_bins = self.nbins + 1

        # bin RTs
        rt_bins = np.searchsorted(boundaries, rt)

        # map choices → indices (fixed from data)
        choice_map = {c: i for i, c in enumerate(choice_vals)}
        counts = np.zeros((len(choice_vals), n_rt_bins), dtype=float)

        for c, b in zip(choice, rt_bins):
            if not np.isfinite(c):
                continue
            if c not in choice_map:
                continue
            counts[choice_map[c], b] += 1

        return counts.reshape(-1)

    def _variance_nll(self, rt_data, coh_data, rt_pred, coh_pred):
        """
        Penalizes mismatch in RT variance per coherence level.
        Leak inflates RT variance at low coherence disproportionately,
        giving the optimizer a direct signal to separate leak from urgency.
        Uses log-ratio so the penalty is scale-invariant.
        """
        coh_data = np.round(coh_data.astype(float), 2)
        coh_pred = np.round(coh_pred.astype(float), 2)

        total = 0.0
        n_bins = 0

        for coh in np.unique(coh_data):
            rd = rt_data[coh_data == coh]
            rp = rt_pred[coh_pred == coh]

            if len(rd) < 5 or len(rp) < 5:
                continue

            var_d = np.var(rd)
            var_p = np.var(rp)

            # log-ratio penalty: symmetric, scale-invariant
            total += (np.log(var_p + 1e-6) - np.log(var_d + 1e-6)) ** 2
            n_bins += 1

        if n_bins == 0:
            return 0.0

        # normalize by n_bins so weight is interpretable regardless of n coherences
        return self.rt_var_weight * len(rt_data) * (total / n_bins)

    # ----------------------------
    # Main likelihood
    # ----------------------------
    def calculate_likelihood(
        self,
        rt_pred, choice_pred,
        rt_data, choice_data,
        coh_pred, coh_data
    ):
        try:
            # convert to arrays
            rt_pred = np.asarray(rt_pred)
            rt_data = np.asarray(rt_data)
            choice_pred = np.asarray(choice_pred)
            choice_data = np.asarray(choice_data)
            coh_pred = np.asarray(coh_pred)
            coh_data = np.asarray(coh_data)

            # basic check
            if rt_pred.size == 0 or rt_data.size == 0:
                return 1e6

            # Normalise coherence dtype: float32(-0.18) != float64(-0.18) so
            # the == comparisons below silently drop those coherence bins.
            coh_pred = np.round(coh_pred.astype(float), 2)
            coh_data = np.round(coh_data.astype(float), 2)

            # remove NaNs / infs
            vp = self._valid_mask(rt_pred, choice_pred)
            vd = self._valid_mask(rt_data, choice_data)

            if not (vp.any() and vd.any()):
                return 1e6

            rt_pred, choice_pred, coh_pred = rt_pred[vp], choice_pred[vp], coh_pred[vp]
            rt_data, choice_data, coh_data = rt_data[vd], choice_data[vd], coh_data[vd]

            total = 0.0

            # loop over coherence conditions
            for coh in np.unique(coh_data):
                dm = coh_data == coh
                pm = coh_pred == coh

                if not (dm.any() and pm.any()):
                    continue

                rt_d, ch_d = rt_data[dm], choice_data[dm]
                rt_p, ch_p = rt_pred[pm], choice_pred[pm]

                # guard against tiny samples
                if len(rt_d) < 5 or len(rt_p) < 5:
                    continue

                # skip degenerate RT distributions
                if np.all(rt_d == rt_d[0]):
                    continue

                # compute quantile bins safely
                try:
                    boundaries = np.quantile(rt_d, self._q_grid)
                except Exception:
                    continue

                # FIXED: use data-defined choice set
                choice_vals = np.unique(ch_d)

                obs_counts = self._joint_counts(rt_d, ch_d, boundaries, choice_vals)
                pred_counts = self._joint_counts(rt_p, ch_p, boundaries, choice_vals)

                if obs_counts.sum() == 0 or pred_counts.sum() == 0:
                    continue

                K = len(obs_counts)

                # smooth probabilities
                obs_p  = (obs_counts  + self.eps) / (obs_counts.sum()  + self.eps * K)
                pred_p = (pred_counts + self.eps) / (pred_counts.sum() + self.eps * K)

                # avoid log(0)
                pred_p = np.clip(pred_p, self.p_min, 1.0)

                # KL divergence (scaled by counts)
                kl = np.sum(obs_counts * (np.log(obs_p) - np.log(pred_p)))

                if not np.isfinite(kl):
                    return 1e6

                total += kl

            # CAF term: pooled across coherences, constrains leak vs urgency
            if self.caf_weight > 0:
                total += self._caf_nll(
                    rt_data, choice_data, coh_data,
                    rt_pred, choice_pred, coh_pred,
                )

            # RT variance term: penalizes mismatch in RT variance per coherence level
            if self.rt_var_weight > 0:
                total += self._variance_nll(
                    rt_data, coh_data,
                    rt_pred, coh_pred,
                )

            return float(total) if np.isfinite(total) else 1e6

        except Exception:
            # NEVER let optimizer crash
            return 1e6

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
        device: str | None = None,
        likelihood_params: dict | None = None,
    ):
        # Use Numba (fastest on CPU) unless CUDA is actually available.
        if device == "cuda" and torch.cuda.is_available():
            self.simulator = DriftDiffusionSimulatorCUDA(device="cuda")
        else:
            if device == "cuda":
                logger.warning("CUDA requested but unavailable, using Numba CPU.")
            self.simulator = DriftDiffusionSimulator()

        lp = likelihood_params or {}
        self.likelihood_calc = LikelihoodCalculator(
            nbins=lp.get("nbins", 5),
            caf_weight=lp.get("caf_weight", 0.0),
            caf_bins=lp.get("caf_bins", 5),
        )
        logger.info(
            "Initialized %s | device=%s",
            type(self).__name__,
            getattr(self.simulator, "device", "CPU"),
        )

    @property
    def param_specs(self) -> dict[str, ParamSpec]:
        raise NotImplementedError

    def _build_params(self, values: np.ndarray, condition_idx: int | None = None) -> dict:
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

    def _simulate_condition(
        self,
        stimulus: np.ndarray,
        params: dict,
        n_reps: int,
    ) -> dict | None:
        """
        Run n_reps simulations. Returns dict with rt/choice/coherence arrays,
        or None if the simulation crashed or produced no valid trials.
        """
        all_rt, all_choice, all_coh = [], [], []
        for _ in range(n_reps):
            try:
                rt, choice = self.simulator.simulate_trials(stimulus, params)
            except Exception as exc:
                logger.warning("Simulation failed: %s", exc)
                return None
            valid = (~np.isnan(rt)) & (~np.isnan(choice))
            if valid.sum() < 5:
                continue
            all_rt.append(rt[valid])
            all_choice.append(choice[valid])
            all_coh.append(stimulus[valid, COH_COL])

        if not all_rt:
            return None

        return {
            "rt":        np.concatenate(all_rt),
            "choice":    np.concatenate(all_choice),
            "coherence": np.concatenate(all_coh),
        }

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
        Fit parameters via differential evolution.

        Returns dict with keys: success, parameters, likelihood,
        n_iterations, optimization_result.
        """
        if verbose:
            logger.info("Starting optimization (%s)...", type(self).__name__)

        specs  = self.param_specs
        bounds = [s.bounds for s in specs.values()]

        np.random.seed(seed)
        torch.manual_seed(seed)

        best_nll  = [np.inf]
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
            logger.info(
                "Iter %3d | NLL=%.4f | convergence=%.4f | %s",
                iteration[0], best_nll[0], convergence, params_str,
            )

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
            logger.info("Optimization done. Cost: %.4f", result.fun)
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
        params: dict | None = None,
        n_reps: int = 1,
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Forward pass: returns dict with rt, choice, coherence arrays."""
        if params is None:
            params = dict(DEFAULT_PARAMS)
        validate_params(params)

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
