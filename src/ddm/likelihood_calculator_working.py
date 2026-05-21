
from __future__ import annotations
import logging
import warnings
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
from numba import jit, prange
from scipy.optimize import differential_evolution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LikelihoodCalculator:
    """
    Joint NLL combining three complementary signals:

    1. Per-coherence KL divergence over (choice × RT-bin) joint distribution —
       matches the full marginal shape at each coherence level.

    2. Choice-conditional chronometric matching (per coherence, per boundary) —
       directly penalizes mean RT mismatch for correct *and* error responses
       separately. This is the primary fix for the chronometric V-shape issue:
       the KL alone gives low weight to error-RT shape at high coherences because
       errors are rare, so log-mean-RT residuals fill that gap.

    3. Global CAF term — pools across all non-zero coherences, bins by RT quantile,
       penalizes accuracy-over-time mismatch. Constrains leak vs urgency.
    """

    def __init__(
        self,
        nbins: int = 5,
        p_min: float = 1e-6,
        caf_weight: float = 1.0,
        caf_bins: int = 5,
        rt_var_weight: float = 0.0,
        stability_weight: float = 0.0,
        chrono_weight: float = 2.0,
    ):
        self.nbins            = nbins
        self.p_min            = p_min
        self.caf_weight       = caf_weight
        self.caf_bins         = caf_bins
        self.rt_var_weight    = rt_var_weight
        self.stability_weight = stability_weight
        self.chrono_weight    = chrono_weight
        self.eps              = 1e-12
        self._q_grid          = np.linspace(1 / (nbins + 1), nbins / (nbins + 1), nbins)

    # ----------------------------
    # Helpers
    # ----------------------------

    def _valid_mask(self, rt, choice):
        return np.isfinite(rt) & np.isfinite(choice)

    def _joint_counts(self, rt, choice, boundaries, choice_vals):
        n_rt_bins  = self.nbins + 1
        rt_bins    = np.searchsorted(boundaries, rt)
        choice_map = {c: i for i, c in enumerate(choice_vals)}
        counts     = np.zeros((len(choice_vals), n_rt_bins), dtype=float)

        for c, b in zip(choice, rt_bins):
            if not np.isfinite(c):
                continue
            if c not in choice_map:
                continue
            counts[choice_map[c], b] += 1

        return counts.reshape(-1)

    def _chronometric_nll(self, rt_d, ch_d, rt_p, ch_p) -> float:
        """
        Log-ratio mean RT mismatch per boundary choice.

        Weighting by choice proportion ensures that rare choices (errors at high
        coherence) receive proportional — rather than zero — gradient signal.
        Without this, the KL at high-coherence conditions is dominated by the
        majority choice and error-RT mismatch is effectively invisible.
        """
        total = 0.0
        n_d   = len(ch_d)

        for choice_val in [0, 1]:
            mask_d = ch_d == choice_val
            mask_p = ch_p == choice_val

            if mask_d.sum() < 3 or mask_p.sum() < 3:
                continue

            mrt_d = np.mean(rt_d[mask_d])
            mrt_p = np.mean(rt_p[mask_p])

            # weight by choice proportion: equal contribution per trial, not per choice
            p_choice = mask_d.sum() / n_d
            total   += p_choice * (np.log(mrt_d + self.eps) - np.log(mrt_p + self.eps)) ** 2

        return total

    def _caf_nll(self, rt_data, ch_data, coh_data, rt_pred, ch_pred, coh_pred):
        """
        MSE between data and predicted CAF, scaled by n_obs × caf_weight.

        RT quantile boundaries are derived from the data so bins are comparable.
        Coherence=0 trials excluded (no ground-truth correct answer).
        """
        def _filter(rt, ch, coh):
            mask = (np.round(coh, 2) != 0) & np.isfinite(rt) & np.isfinite(ch)
            return rt[mask], ch[mask], coh[mask]

        rt_d, ch_d, coh_d = _filter(rt_data, ch_data, coh_data)
        rt_p, ch_p, coh_p = _filter(rt_pred,  ch_pred,  coh_pred)

        if len(rt_d) < self.caf_bins or len(rt_p) < self.caf_bins:
            return 0.0

        correct_d = ((coh_d > 0) & (ch_d == 1)) | ((coh_d < 0) & (ch_d == 0))
        correct_p = ((coh_p > 0) & (ch_p == 1)) | ((coh_p < 0) & (ch_p == 0))

        boundaries      = np.quantile(rt_d, np.linspace(0, 1, self.caf_bins + 1))
        boundaries[-1] += 1e-9

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

    def _variance_nll(self, rt_data, coh_data, rt_pred, coh_pred):
        """
        Log-ratio RT variance mismatch per coherence, scale-invariant.
        """
        coh_data = np.round(coh_data.astype(float), 2)
        coh_pred = np.round(coh_pred.astype(float), 2)

        total  = 0.0
        n_bins = 0

        for coh in np.unique(coh_data):
            rd = rt_data[coh_data == coh]
            rp = rt_pred[coh_pred == coh]

            if len(rd) < 5 or len(rp) < 5:
                continue

            var_d = np.var(rd)
            var_p = np.var(rp)

            total  += (np.log(var_p + 1e-6) - np.log(var_d + 1e-6)) ** 2
            n_bins += 1

        if n_bins == 0:
            return 0.0

        return self.rt_var_weight * len(rt_data) * (total / n_bins)

    def _distribution_stability(self, rt_d, ch_d, rt_p, ch_p) -> float:
        eps = 1e-12

        def entropy(x):
            hist, _ = np.histogram(x, bins=10)
            p = hist / (hist.sum() + eps)
            p = p[p > 0]
            return -np.sum(p * np.log(p + eps))

        def choice_entropy(ch):
            if len(ch) == 0:
                return 0.0
            vals, counts = np.unique(ch, return_counts=True)
            p = counts / counts.sum()
            return -np.sum(p * np.log(p + eps))

        entropy_term   = (entropy(rt_d) - entropy(rt_p)) ** 2
        coverage_d     = (np.max(rt_d) - np.min(rt_d)) if len(rt_d) > 1 else 0.0
        coverage_p     = (np.max(rt_p) - np.min(rt_p)) if len(rt_p) > 1 else 0.0
        coverage_term  = (coverage_d - coverage_p) ** 2
        choice_term    = (choice_entropy(ch_d) - choice_entropy(ch_p)) ** 2

        return self.stability_weight * (entropy_term + 0.5 * coverage_term + 0.5 * choice_term)

    # ----------------------------
    # Main likelihood
    # ----------------------------

    def compute_nll(
        self,
        rt_pred, choice_pred,
        rt_data, choice_data,
        coh_pred, coh_data,
    ):
        try:
            rt_pred    = np.asarray(rt_pred)
            rt_data    = np.asarray(rt_data)
            choice_pred = np.asarray(choice_pred)
            choice_data = np.asarray(choice_data)
            coh_pred   = np.round(np.asarray(coh_pred).astype(float), 2)
            coh_data   = np.round(np.asarray(coh_data).astype(float), 2)

            if rt_pred.size == 0 or rt_data.size == 0:
                return 1e6

            vp = self._valid_mask(rt_pred, choice_pred)
            vd = self._valid_mask(rt_data, choice_data)

            if not (vp.any() and vd.any()):
                return 1e6

            rt_pred,    choice_pred,  coh_pred  = rt_pred[vp],    choice_pred[vp],  coh_pred[vp]
            rt_data,    choice_data,  coh_data  = rt_data[vd],    choice_data[vd],  coh_data[vd]

            total = 0.0

            for coh in np.unique(coh_data):
                dm = coh_data == coh
                pm = coh_pred == coh

                rt_d, ch_d = rt_data[dm], choice_data[dm]
                rt_p, ch_p = rt_pred[pm], choice_pred[pm]

                if len(rt_d) < 5:
                    logger.warning(f"Coherence {coh}: too few data trials ({len(rt_d)}), skipping.")
                    return 1e6

                if len(rt_p) < 5:
                    # Model predicts almost no crossings — clearly wrong parameters
                    # (boundary too wide, drift too weak, or time window too short).
                    # Penalize proportional to the data size so the signal is strong,
                    # but stay finite so gradient from other coherences is preserved.
                    logger.warning(
                        f"Coherence {coh}: only {len(rt_p)} predicted crossings "
                        f"vs {len(rt_d)} data trials — adding crossing penalty."
                    )
                    total += 10.0 * len(rt_d)
                    continue

                if np.all(rt_d == rt_d[0]):
                    logger.warning(f"Coherence {coh}: degenerate RT distribution in data, skipping.")
                    return 1e6

                try:
                    boundaries = np.quantile(rt_d, self._q_grid)
                except Exception:
                    return 1e6

                choice_vals  = np.unique(ch_d)
                obs_counts   = self._joint_counts(rt_d, ch_d, boundaries, choice_vals)
                pred_counts  = self._joint_counts(rt_p, ch_p, boundaries, choice_vals)

                if obs_counts.sum() == 0 or pred_counts.sum() == 0:
                    return 1e6

                K      = len(obs_counts)
                obs_p  = (obs_counts  + self.eps) / (obs_counts.sum()  + self.eps * K)
                pred_p = (pred_counts + self.eps) / (pred_counts.sum() + self.eps * K)
                pred_p = np.clip(pred_p, self.p_min, 1.0)

                kl = np.sum(obs_counts * (np.log(obs_p) - np.log(pred_p)))

                if not np.isfinite(kl):
                    return 1e6

                total += kl

                # Choice-conditional chronometric matching.
                # Added directly inside the coherence loop so it scales with
                # per-coherence trial count — consistent with the KL above.
                if self.chrono_weight > 0:
                    total += self.chrono_weight * len(rt_d) * self._chronometric_nll(
                        rt_d, ch_d, rt_p, ch_p
                    )

            if self.caf_weight > 0:
                total += self._caf_nll(
                    rt_data, choice_data, coh_data,
                    rt_pred, choice_pred, coh_pred,
                )

            if self.rt_var_weight > 0:
                total += self._variance_nll(rt_data, coh_data, rt_pred, coh_pred)

            if self.stability_weight > 0:
                total += self._distribution_stability(
                    rt_data, choice_data, rt_pred, choice_pred
                )

            return float(total) if np.isfinite(total) else 1e6

        except Exception:
            return 1e6
