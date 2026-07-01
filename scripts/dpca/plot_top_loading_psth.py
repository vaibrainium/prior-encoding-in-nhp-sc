#!/usr/bin/env python
"""Plot PSTHs (all epochs) of the top-N decoder-loading neurons for one marginalization/epoch.

Defaults: toRF, bias axis 'b', decoder D, PC0, fit epoch = visual, N = 15.
Loadings are read from the saved analysis output (loadings.pkl); the neuron order in D
matches get_neuron_ids(...) used by the analysis (clean_dpca_data drops only timepoints/
trials, never neurons), so row i of D ↔ nids[i].

PSTHs are raw firing rate (Hz), averaged over all conditions (state×coh×choice).

Usage:
    conda run -n prior-sc python scripts/dpca/plot_top_loading_psth.py
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

# ── Config ──────────────────────────────────────────────────────────────────────
GROUP      = "toRF"
FIT_EPOCH  = "visual"
MARG       = "b"          # bias axis
WEIGHT     = "D"          # "D" decoder (contribution to component) | "P" encoder
PC         = 0
TOP_N      = 15

ALIGNMENTS = list(ephys_config["alignment_settings_GP"].keys())
CONDITION_DICT = {
    "state_values": ["biased", "unbiased"],
    "coherences": [0, 0.06, 0.2, 0.5],
    "choices": ["awayRF", "toRF"],
}
SESSION_TO_EXCLUDE = ["210210_GP_JP", "241209_GP_TZ"]


def _log(m):
    import datetime
    print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)


def main():
    processed_dir = Path(dir_config.data.processed)
    compiled_dir  = Path(dir_config.data.compiled)
    out_dir = PROJECT_ROOT / "dissemination" / "dpca" / f"dpca_{GROUP}_session_all_neuron_norm_zscore"
    load_path = out_dir / "loadings.pkl"

    _log(f"loading metadata + loadings ({load_path.name}) ...")
    session_metadata = pd.read_csv(processed_dir / "sessions_metadata.csv")
    session_metadata = session_metadata[~session_metadata.session_id.isin(SESSION_TO_EXCLUDE)].reset_index(drop=True)
    neuron_metadata = pd.read_csv(processed_dir / "neuron_metadata.csv")
    neuron_metadata = neuron_metadata[~neuron_metadata.session_id.isin(SESSION_TO_EXCLUDE)].reset_index(drop=True)
    with open(load_path, "rb") as f:
        loadings = pickle.load(f)

    sessions = session_metadata.session_id[session_metadata.prior_direction == GROUP]

    # Reconstruct the exact neuron order used at fit time → maps D rows to neuron_ids.
    nids = dpca_utils.get_neuron_ids(neuron_metadata, sessions)
    Wmat = loadings[FIT_EPOCH][WEIGHT][MARG]            # (n_neurons, n_components)
    _log(f"{len(nids)} neurons (matches D row count: {Wmat.shape[0]})")
    assert len(nids) == Wmat.shape[0], "neuron order mismatch!"

    w = Wmat[:, PC]
    top_idx = np.argsort(np.abs(w))[::-1][:TOP_N]
    top_nids = nids[top_idx]
    top_w = w[top_idx]

    # cell-type lookup
    nm = neuron_metadata.set_index("neuron_id")
    _log(f"top {TOP_N} |{WEIGHT}['{MARG}'] PC{PC}| in {FIT_EPOCH}:")
    for rank, (nid, wt) in enumerate(zip(top_nids, top_w), 1):
        ct = nm.loc[nid, "classification"] if nid in nm.index else "?"
        _log(f"  {rank:2d}. neuron {nid:>6}  weight={wt:+.4f}  [{ct}]")

    # Build PSTHs: raw averaged dPCA matrix, then nanmean over conditions (state,coh,choice).
    _log("loading glm_hmm + ephys, building PSTH matrix ...")
    with open(processed_dir / "glm_hmm_models" / "glm_hmm_masked_final.pkl", "rb") as f:
        glm_hmm = pickle.load(f)
    glm_hmm_original = copy.deepcopy(glm_hmm)
    with open(processed_dir / "ephys_neuron_wise.pkl", "rb") as f:
        ephys = pickle.load(f)
    biased_ti, unbiased_ti, _ = dpca_utils.extract_hmm_state_trial_info(
        session_metadata, glm_hmm_original, glm_hmm["data"], compiled_dir=compiled_dir,
    )
    state_trial_info = {"biased": biased_ti, "unbiased": unbiased_ti}

    avg, _ = dpca_utils.create_dpca_matrix(
        sessions, CONDITION_DICT, nids, state_trial_info,
        neuron_metadata, ephys, ephys_config, condition_type="states",
    )
    # per-neuron PSTH per alignment: nanmean over state(1), coh(2), choice(3)
    psth = {a: np.nanmean(avg[a], axis=(1, 2, 3)) for a in ALIGNMENTS}  # (n_neurons, n_time)
    settings = ephys_config["alignment_settings_GP"]
    taxis = {a: np.arange(settings[a]["start_time_ms"], settings[a]["end_time_ms"] + 1)
             for a in ALIGNMENTS}

    _log("plotting ...")
    n_cols = len(ALIGNMENTS)
    fig, axes = plt.subplots(TOP_N, n_cols, figsize=(3 * n_cols, 1.5 * TOP_N),
                             squeeze=False, sharex="col")
    for col, a in enumerate(ALIGNMENTS):
        axes[0, col].set_title(a)
    for row, (ni, nid, wt) in enumerate(zip(top_idx, top_nids, top_w)):
        ct = nm.loc[nid, "classification"] if nid in nm.index else "?"
        for col, a in enumerate(ALIGNMENTS):
            ax = axes[row, col]
            t = taxis[a]
            y = psth[a][ni]
            n = min(len(t), len(y))
            ax.plot(t[:n], y[:n], lw=1.0, color="k")
            ax.axvline(0, color="r", lw=0.6, ls=":")
            if col == 0:
                ax.set_ylabel(f"#{nid}\nw={wt:+.2f}\n{ct}", fontsize=7, rotation=0,
                              ha="right", va="center", labelpad=22)
            ax.tick_params(labelsize=6)
        if row == TOP_N - 1:
            for col, a in enumerate(ALIGNMENTS):
                axes[row, col].set_xlabel("time (ms)", fontsize=7)
    fig.suptitle(f"Top {TOP_N} {WEIGHT}['{MARG}'] PC{PC} neurons in {FIT_EPOCH} — "
                 f"PSTH (Hz, all conditions) — {GROUP}", y=1.002)
    fig.tight_layout()
    out_png = out_dir / f"top{TOP_N}_{WEIGHT}_{MARG}_{FIT_EPOCH}_psth.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"saved {out_png}")


if __name__ == "__main__":
    main()
