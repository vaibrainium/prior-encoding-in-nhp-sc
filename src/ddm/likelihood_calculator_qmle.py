import numpy as np

class LikelihoodCalculator:
    def __init__(self, nbins=9, rt_nllh_weight=1.0):
        """
        Class to compute Ratcliff-style QMLE likelihoods for diffusion model fits.

        Args:
            nbins (int): Number of quantile bins for RT likelihood estimation (default=9).
            rt_nllh_weight (float): Scalar applied to the RT NLL before summing with
                the choice NLL.  The two terms have different natural scales: choice NLL
                is O(n_trials * log(2)) per coherence (~n*0.7) while RT NLL is
                O(n_trials * nbins * log(nbins)) and grows with nbins.  The default of
                1.0 sums them directly; increase to up-weight RT shape, decrease to
                up-weight choice accuracy.
        """
        self.nbins = nbins
        self.eps = 1e-24  # small number to avoid log(0)
        self.rt_nllh_weight = rt_nllh_weight

    def calculate_llh_QMLE(self, rt_model, rt_data):
        """
        Quasi-Maximum Likelihood (Ratcliff quantile likelihood) for RT distributions.

        Args:
            rt_model (array-like): simulated RTs (ms)
            rt_data (array-like): empirical RTs (ms)

        Returns:
            nllh (float): negative log likelihood for RT data given model

        """
        rt_model = np.asarray(rt_model)
        rt_data = np.sort(np.asarray(rt_data))
        n = len(rt_data)
        if self.nbins > n:
            # Too few data trials to form stable bins — skip this cell.
            return 0.0
        else:
            nbins_used = self.nbins

        quantiles = np.linspace(0, 1, nbins_used + 1)
        bin_edges = np.quantile(rt_data, quantiles)

        # Expand edges to fully cover model RTs
        bin_edges[0] = min(bin_edges[0], np.min(rt_model))
        bin_edges[-1] = max(bin_edges[-1], np.max(rt_model))

        counts_per_bin = np.histogram(rt_data, bins=bin_edges)[0]
        probs_per_bin = np.histogram(rt_model, bins=bin_edges)[0] / len(rt_model)
        nllh = -np.sum(counts_per_bin * np.log(probs_per_bin + self.eps))
        return nllh

    def calculate_choice_likelihood(self, choice_model, choice_data):
        """
        Negative log likelihood for choice proportions.

        Args:
            choice_model (array-like): simulated choices
            choice_data (array-like): empirical choices

        Returns:
            nllh (float): negative log likelihood for choice data given model

        """
        choice_model = np.asarray(choice_model)
        choice_data = np.asarray(choice_data)

        categories = np.unique(choice_data)
        nllh = 0.0
        for cat in categories:
            n_cat_data = np.sum(choice_data == cat)
            p_cat_model = np.mean(choice_model == cat)
            nllh -= n_cat_data * np.log(p_cat_model + self.eps)
        return nllh

    def compute_nll(self, rt_pred_or_prediction, choice_pred_or_data=None,
                    rt_data=None, choice_data=None, coh_pred=None, coh_data=None):
        """
        Computes Ratcliff-style QMLE negative log likelihood summed over coherence × choice subsets.

        Accepts two call signatures for backward compatibility:

          New: compute_nll(prediction_dict, data_dict)
            prediction / data: dicts with keys 'signed_coherence', 'choice', 'rt'

          Old: compute_nll(rt_pred, choice_pred, rt_data, choice_data, coh_pred, coh_data)
            Six positional arrays matching the interface of LikelihoodCalculator.

        Returns:
            total_nllh (float): total negative log likelihood

        """
        if isinstance(rt_pred_or_prediction, dict):
            # New dict signature: compute_nll(prediction_dict, data_dict)
            prediction = rt_pred_or_prediction
            data = choice_pred_or_data
        else:
            # Old 6-array signature: compute_nll(rt_pred, choice_pred, rt_data, choice_data, coh_pred, coh_data)
            prediction = {
                "rt": np.asarray(rt_pred_or_prediction),
                "choice": np.asarray(choice_pred_or_data),
                "signed_coherence": np.asarray(coh_pred),
            }
            data = {
                "rt": np.asarray(rt_data),
                "choice": np.asarray(choice_data),
                "signed_coherence": np.asarray(coh_data),
            }

        # remove trials with NaN choices or RTs
        mask_data = (~np.isnan(data["choice"])) & (~np.isnan(data["rt"]))
        data = {k: v[mask_data] for k, v in data.items()}
        mask_model = (~np.isnan(prediction["choice"])) & (~np.isnan(prediction["rt"]))
        prediction = {k: v[mask_model] for k, v in prediction.items()}

        total_nllh = 0.0
        unique_cohs = np.unique(data["signed_coherence"])
        unique_choices = np.unique(data["choice"])

        for coh in unique_cohs:
            mask_data_coh = data["signed_coherence"] == coh
            mask_model_coh = prediction["signed_coherence"] == coh

            if np.sum(mask_data_coh) == 0 or np.sum(mask_model_coh) == 0:
                continue

            choice_data_coh = data["choice"][mask_data_coh]
            choice_model_coh = prediction["choice"][mask_model_coh]

            if len(choice_data_coh) == 0 or len(choice_model_coh) == 0:
                continue

            # Choice likelihood per coherence
            total_nllh += self.calculate_choice_likelihood(choice_model_coh, choice_data_coh)

            # RT likelihood per choice
            for choice_val in unique_choices:
                mask_data = mask_data_coh & (data["choice"] == choice_val)
                mask_model = mask_model_coh & (prediction["choice"] == choice_val)

                rt_data_sub = data["rt"][mask_data]
                rt_model_sub = prediction["rt"][mask_model]

                if len(rt_data_sub) == 0 or len(rt_model_sub) == 0:
                    continue

                nllh_rt = self.calculate_llh_QMLE(rt_model_sub, rt_data_sub)
                total_nllh += nllh_rt * self.rt_nllh_weight

        return total_nllh
