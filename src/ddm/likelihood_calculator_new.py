from __future__ import annotations

import numpy as np


class LikelihoodCalculator:
    """
    Unified geometry-aware likelihood.

    Matches:
    - psychometric curve
    - chronometric curve
    - RT variance structure
    - RT skewness structure
    - conditional accuracy geometry

    without requiring:
    - KL divergence
    - CAF penalties
    - entropy penalties
    - quantile binning

    This is substantially more stable for fitting
    leak vs urgency simultaneously.
    """

    def __init__(
        self,
        moment_weight: float = 1.0,
        eps: float = 1e-12,
        entropy_floor: float = 0.2,
    ):
        self.moment_weight = moment_weight
        self.eps = eps
        self.entropy_floor = entropy_floor

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _safe_var(self, x):
        if len(x) < 2:
            return 0.0
        return float(np.var(x))

    def _safe_mean(self, x):
        if len(x) == 0:
            return np.nan
        return float(np.mean(x))

    def _safe_skew(self, x):

        if len(x) < 3:
            return 0.0

        mean = np.mean(x)
        std = np.std(x)

        if std < self.eps:
            return 0.0

        z = (x - mean) / std
        return float(np.mean(z ** 3))

    def _rt_entropy(self, rt):

        hist, _ = np.histogram(rt, bins=20)

        p = hist / (hist.sum() + self.eps)
        p = p[p > 0]

        return -np.sum(p * np.log(p + self.eps))

    def _caf_curve(self, rt, choice, coh, bins=5):

        valid = (
            np.isfinite(rt)
            & np.isfinite(choice)
            & (coh != 0)
        )

        rt = rt[valid]
        choice = choice[valid]
        coh = coh[valid]

        if len(rt) < bins:
            return None

        correct = (
            ((coh > 0) & (choice == 1))
            | ((coh < 0) & (choice == 0))
        )

        q = np.quantile(rt, np.linspace(0, 1, bins + 1))
        q[-1] += 1e-9

        curve = []

        for lo, hi in zip(q[:-1], q[1:]):

            mask = (rt >= lo) & (rt < hi)

            if mask.sum() == 0:
                continue

            curve.append(correct[mask].mean())

        if len(curve) == 0:
            return None

        return np.asarray(curve)

    # ---------------------------------------------------------
    # Main likelihood
    # ---------------------------------------------------------

    def calculate_likelihood(
        self,
        rt_pred,
        choice_pred,
        rt_data,
        choice_data,
        coh_pred,
        coh_data,
    ):

        try:

            rt_pred = np.asarray(rt_pred)
            rt_data = np.asarray(rt_data)

            choice_pred = np.asarray(choice_pred)
            choice_data = np.asarray(choice_data)

            coh_pred = np.round(np.asarray(coh_pred).astype(float), 3)
            coh_data = np.round(np.asarray(coh_data).astype(float), 3)

            valid_pred = np.isfinite(rt_pred) & np.isfinite(choice_pred)
            valid_data = np.isfinite(rt_data) & np.isfinite(choice_data)

            if valid_pred.sum() < 10:
                return 1e6

            if valid_data.sum() < 10:
                return 1e6

            rt_pred = rt_pred[valid_pred]
            choice_pred = choice_pred[valid_pred]
            coh_pred = coh_pred[valid_pred]

            rt_data = rt_data[valid_data]
            choice_data = choice_data[valid_data]
            coh_data = coh_data[valid_data]

            total = 0.0

            coherences = np.unique(coh_data)

            # -------------------------------------------------
            # Per-coherence geometry matching
            # -------------------------------------------------

            for coh in coherences:

                dmask = coh_data == coh
                pmask = coh_pred == coh

                if dmask.sum() < 5 or pmask.sum() < 5:
                    continue

                rd = rt_data[dmask]
                rp = rt_pred[pmask]

                cd = choice_data[dmask]
                cp = choice_pred[pmask]

                # -----------------------------------------
                # Psychometric
                # -----------------------------------------

                p_d = np.mean(cd == 1)
                p_p = np.mean(cp == 1)

                psychometric_loss = (p_d - p_p) ** 2

                # -----------------------------------------
                # Chronometric
                # -----------------------------------------

                mrt_d = np.mean(rd)
                mrt_p = np.mean(rp)

                chronometric_loss = (
                    np.log(mrt_d + self.eps)
                    - np.log(mrt_p + self.eps)
                ) ** 2

                # -----------------------------------------
                # RT variance geometry
                # -----------------------------------------

                var_d = self._safe_var(rd)
                var_p = self._safe_var(rp)

                variance_loss = (
                    np.log(var_d + self.eps)
                    - np.log(var_p + self.eps)
                ) ** 2

                # -----------------------------------------
                # RT skewness geometry
                # -----------------------------------------

                skew_d = self._safe_skew(rd)
                skew_p = self._safe_skew(rp)

                skew_loss = (skew_d - skew_p) ** 2

                total += (
                    3.0 * psychometric_loss
                    + 2.0 * chronometric_loss
                    + 1.5 * variance_loss
                    + 1.0 * skew_loss
                )

            # -------------------------------------------------
            # Global CAF geometry
            # -------------------------------------------------

            caf_d = self._caf_curve(
                rt_data,
                choice_data,
                coh_data,
            )

            caf_p = self._caf_curve(
                rt_pred,
                choice_pred,
                coh_pred,
            )

            if caf_d is not None and caf_p is not None:

                m = min(len(caf_d), len(caf_p))

                caf_loss = np.mean(
                    (caf_d[:m] - caf_p[:m]) ** 2
                )

                total += 4.0 * caf_loss

            # -------------------------------------------------
            # Entropy regularization
            # -------------------------------------------------

            entropy_d = self._rt_entropy(rt_data)
            entropy_p = self._rt_entropy(rt_pred)

            entropy_loss = (
                entropy_d - entropy_p
            ) ** 2

            total += 0.25 * entropy_loss

            # -------------------------------------------------
            # Scale by trial count
            # -------------------------------------------------

            total *= len(rt_data)

            if not np.isfinite(total):
                return 1e6

            return float(total)

        except Exception:
            return 1e6
