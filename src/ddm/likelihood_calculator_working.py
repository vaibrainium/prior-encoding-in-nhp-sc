
from __future__ import annotations
import logging
import warnings
from dataclasses import dataclass  # still used by ParamSpec
from typing import Tuple
import numpy as np
import torch
from numba import jit, prange
from scipy.optimize import differential_evolution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENTROPY_THRESHOLD = 0.2  # empirically determined threshold for low RT entropy


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
        rt_var_weight: float = 0.0,
        stability_weight: float = 0.1,
    ):
        self.nbins      = nbins
        self.p_min      = p_min
        self.caf_weight = caf_weight
        self.caf_bins   = caf_bins
        self.stability_weight = stability_weight
        self.rt_var_weight = rt_var_weight
        self.eps        = 1e-12
        self._q_grid    = np.linspace(1/(nbins+1), nbins/(nbins+1), nbins)

        if rt_var_weight is not None or kwargs:
            warnings.warn(
                "Deprecated likelihood parameters (rt_weight, sparse_threshold, etc.) "
                "are ignored in JointLikelihoodCalculator.",
                RuntimeWarning
            )

    # ----------------------------
    # Helpers
    # ----------------------------

    def _distribution_stability(self, rt_d, ch_d, rt_p, ch_p) -> float:
        eps = 1e-12

        # -------------------------
        # 1. RT entropy mismatch
        # -------------------------
        def entropy(x):
            hist, _ = np.histogram(x, bins=10)
            p = hist / (hist.sum() + eps)
            p = p[p > 0]
            return -np.sum(p * np.log(p + eps))

        ent_d = entropy(rt_d)
        ent_p = entropy(rt_p)

        entropy_term = (ent_d - ent_p) ** 2

        # -------------------------
        # 2. Coverage mismatch
        # (how much of RT space is actually used)
        # -------------------------
        coverage_d = (np.max(rt_d) - np.min(rt_d)) if len(rt_d) > 1 else 0.0
        coverage_p = (np.max(rt_p) - np.min(rt_p)) if len(rt_p) > 1 else 0.0

        coverage_term = (coverage_d - coverage_p) ** 2

        # -------------------------
        # 3. Choice entropy mismatch
        # -------------------------
        def choice_entropy(ch):
            if len(ch) == 0:
                return 0.0
            vals, counts = np.unique(ch, return_counts=True)
            p = counts / counts.sum()
            return -np.sum(p * np.log(p + eps))

        choice_term = (choice_entropy(ch_d) - choice_entropy(ch_p)) ** 2

        return (
            self.stability_weight *
            (entropy_term + 0.5 * coverage_term + 0.5 * choice_term)
        )

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

    def _rt_entropy(self, rt: np.ndarray) -> float:
        hist, _ = np.histogram(rt, bins=10)
        p = hist / (hist.sum() + 1e-12)
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    # ----------------------------
    # Main likelihood
    # ----------------------------
    def compute_nll(
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
                    return 1e6

                rt_d, ch_d = rt_data[dm], choice_data[dm]
                rt_p, ch_p = rt_pred[pm], choice_pred[pm]

                # guard against tiny samples
                if len(rt_d) < 5 or len(rt_p) < 5:
                    logger.warning(f"Coherence {coh}: too few valid trials (data={len(rt_d)}, pred={len(rt_p)}), skipping.")
                    return 1e6

                # skip degenerate RT distributions
                if np.all(rt_d == rt_d[0]):
                    logger.warning(f"Coherence {coh}: degenerate RT distribution in data, skipping.")
                    return 1e6

                # compute quantile bins safely
                try:
                    boundaries = np.quantile(rt_d, self._q_grid)
                except Exception:
                    logger.warning(f"Coherence {coh}: failed to compute quantile boundaries, skipping.")
                    return 1e6

                # FIXED: use data-defined choice set
                choice_vals = np.unique(ch_d)

                obs_counts = self._joint_counts(rt_d, ch_d, boundaries, choice_vals)
                pred_counts = self._joint_counts(rt_p, ch_p, boundaries, choice_vals)

                if obs_counts.sum() == 0 or pred_counts.sum() == 0:
                    logger.warning(f"Coherence {coh}: zero counts in observed or predicted data, skipping.")
                    return 1e6

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

                entropy_d = self._rt_entropy(rt_d)
                if entropy_d < ENTROPY_THRESHOLD:
                    logger.warning(f"Coherence {coh}: very low RT entropy in data ({entropy_d:.4f}), skipping.")
                    total += 500

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


            if self.stability_weight > 0:
                total += self._distribution_stability(rt_data, choice_data, rt_pred, choice_pred)

            return float(total) if np.isfinite(total) else 1e6



        except Exception:
            return 1e6
