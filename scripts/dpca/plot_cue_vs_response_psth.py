#!/usr/bin/env python
"""Demonstrate that cue-aligned and response-aligned PSTHs differ in the overlap window.

The cross-projection Z = sum_i D_i * PSTH_i uses the SAME weights D_visual in every column,
so Z(cue) vs Z(response) can differ only if the PSTHs differ. This script shows, for the
top choice-loading neurons, the cue-aligned and response-aligned PSTHs together with the
number of contributing (non-NaN) trials vs time — revealing that:
  (a) the cue-aligned PSTH is truncated / rolls off as trials drop out (each trial is
      NaN'd ~50 ms before its own saccade), so the peri-saccadic burst is absent; and
  (b) the response-aligned PSTH at t=0 is built from exactly that burst, across all trials.
Hence the PSTHs are NOT similar in the overlap, which is what makes the projection flip.

Usage:
    conda run -n prior-sc python scripts/dpca/plot_cue_vs_response_psth.py
"""

import copy
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import dir_config, ephys_config
from src.utils import dpca_utils

GROUP   = "awayRF"
MARG    = "c"          # choice axis
TOP_N   = 6
PC      = 0
CONDITION_DICT = {
    "state_values": ["biased", "unbiased"],
    "coherences": [0, 0.06, 0.2, 0.5],
    "choices": ["awayRF", "toRF"],
}
SESSION_TO_EXCLUDE = ["210210_GP_JP", "241209_GP_TZ"]
ALIGNMENTS = list(ephys_config["alignment_settings_GP"].keys())


def _log(m):
    import datetime
    print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)


def main():
    processed_dir = Path(dir_config.data.processed)
    compiled_dir  = Path(dir_config.data.compiled)
    out_dir = PROJECT_ROOT / "dissemination" / "dpca" / f"dpca_{GROUP}_session_all_neuron_norm_zscore"

    session_metadata = pd.read_csv(processed_dir / "sessions_metadata.csv")
    session_metadata = session_metadata[~session_metadata.session_id.isin(SESSION_TO_EXCLUDE)].reset_index(drop=True)
    neuron_metadata = pd.read_csv(processed_dir / "neuron_metadata.csv")
    neuron_metadata = neuron_metadata[~neuron_metadata.session_id.isin(SESSION_TO_EXCLUDE)].reset_index(drop=True)
    with open(out_dir / "loadings.pkl", "rb") as f:
        loadings = pickle.load(f)

    sessions = session_metadata.session_id[session_metadata.prior_direction == GROUP]
    nids = dpca_utils.get_neuron_ids(neuron_metadata, sessions)

    w = loadings["visual"]["D"][MARG][:, PC]
    top_idx = np.argsort(np.abs(w))[::-1][:TOP_N]
    top_nids = nids[top_idx]
    top_w = w[top_idx]
    _log(f"top {TOP_N} visual choice-loading neurons: {list(zip(top_nids, np.round(top_w, 2)))}")

    _log("loading glm_hmm + ephys ...")
    with open(processed_dir / "glm_hmm_models" / "glm_hmm_masked_final.pkl", "rb") as f:
        glm_hmm = pickle.load(f)
    glm_hmm_original = copy.deepcopy(glm_hmm)
    with open(processed_dir / "ephys_neuron_wise.pkl", "rb") as f:
        ephys = pickle.load(f)
    biased_ti, unbiased_ti, _ = dpca_utils.extract_hmm_state_trial_info(
        session_metadata, glm_hmm_original, glm_hmm["data"], compiled_dir=compiled_dir,
    )
    state_trial_info = {"biased": biased_ti, "unbiased": unbiased_ti}

    # median reaction time across toRF trials (ms), for the cue→saccade reference line
    rts = []
    for sid in sessions:
        d = glm_hmm["data"][sid]
        if "reaction_time" in d.columns:
            rts.append(np.asarray(d["reaction_time"], dtype=float))
    rts = np.concatenate(rts) if rts else np.array([np.nan])
    rts = rts[np.isfinite(rts)]
    median_rt = float(np.median(rts)) if rts.size else np.nan
    # reaction_time may be in seconds; convert to ms if so
    if median_rt < 5:
        median_rt *= 1000.0
    _log(f"median RT ≈ {median_rt:.0f} ms (n={rts.size})")

    avg, tw = dpca_utils.create_dpca_matrix(
        sessions, CONDITION_DICT, nids, state_trial_info,
        neuron_metadata, ephys, ephys_config, condition_type="states",
    )
    settings = ephys_config["alignment_settings_GP"]

    def taxis(a):
        return np.arange(settings[a]["start_time_ms"], settings[a]["end_time_ms"] + 1)

    cols = ["cue", "response"]
    fig, axes = plt.subplots(TOP_N, len(cols), figsize=(6 * len(cols), 2.4 * TOP_N), squeeze=False)
    for col, a in enumerate(cols):
        axes[0, col].set_title(f"{a}-aligned", fontsize=11)
    for row, (ni, nid, wt) in enumerate(zip(top_idx, top_nids, top_w)):
        for col, a in enumerate(cols):
            ax = axes[row, col]
            t = taxis(a)
            # PSTH: mean over conditions (state,coh,choice) and trials, raw Hz
            psth = np.nanmean(avg[a][ni], axis=(0, 1, 2))            # (n_time_full,)
            # contributing observations per timepoint across trials × conditions
            cell = tw[a][:, ni]                                      # (300, n_s, n_c, n_ch, n_time)
            count = np.sum(~np.isnan(cell), axis=(0, 1, 2, 3))       # (n_time_full,)
            n = min(len(t), len(psth), len(count))
            ax.plot(t[:n], psth[:n], color="k", lw=1.3, label="PSTH (Hz)")
            ax.set_ylabel("Hz", fontsize=8)
            ax2 = ax.twinx()
            ax2.fill_between(t[:n], 0, count[:n], color="tab:blue", alpha=0.18)
            ax2.plot(t[:n], count[:n], color="tab:blue", lw=0.8)
            ax2.set_ylabel("# trials×cond", color="tab:blue", fontsize=8)
            ax2.tick_params(axis="y", labelcolor="tab:blue", labelsize=7)
            ax.axvline(0, color="r", lw=0.6, ls=":")
            if a == "cue" and np.isfinite(median_rt):
                ax.axvline(median_rt, color="g", lw=0.9, ls="--")
                ax.text(median_rt, ax.get_ylim()[1], " med RT\n (saccade)", color="g",
                        fontsize=6, va="top")
            ax.tick_params(labelsize=7)
            wcol = "tab:red" if wt > 0 else "tab:purple"
            ax.text(0.02, 0.95, f"#{nid}  w={wt:+.2f}", transform=ax.transAxes,
                    fontsize=8, va="top", color=wcol,
                    bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.6))
            if row == TOP_N - 1:
                ax.set_xlabel("time (ms)", fontsize=8)
    fig.suptitle(
        f"{GROUP}: cue- vs response-aligned PSTH (top {TOP_N} visual choice-loading neurons)\n"
        f"black=PSTH, blue=# contributing trials×conditions; green dashed=median saccade time on cue axis",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    out_png = out_dir / f"cue_vs_response_psth_top{TOP_N}_{MARG}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"saved {out_png}")


if __name__ == "__main__":
    main()
