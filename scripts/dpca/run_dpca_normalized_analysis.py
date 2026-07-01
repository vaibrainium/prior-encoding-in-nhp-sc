#!/usr/bin/env python
"""Normalized dPCA analysis — toRF & awayRF, all neurons, self + cross projection + sig masks.

Same pipeline as scripts/dpca/run_dpca_analysis.py (all-neuron path), but each neuron is
per-neuron z-score normalized (soft: divide by std + 5 Hz) before fitting. A single scale
per neuron is shared across all alignment periods, which keeps cross-period projection valid.
Z-score only divides by a per-neuron scalar, so it does not change the NaN / trial structure;
the significance trial-splitting behaves exactly as in the raw run.

Usage:
    conda run -n prior-sc python scripts/dpca/run_dpca_normalized_analysis.py

Outputs (under dissemination/dpca/):
    dpca_{group}_session_all_neuron_norm_zscore/
        self_projection.png
        cross_projection_baseline.png
        cross_projection_response.png
        cross_projection_visual.png

Significance masks cached as .pkl in processed_dir/dpca/ (suffix _norm_zscore) and reused.
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
from src.utils import dpca_plot_utils, dpca_utils

# ── Constants (mirror run_dpca_analysis.py) ─────────────────────────────────────

ALIGNMENTS = list(ephys_config["alignment_settings_GP"].keys())
MARGINALIZATION_KEYS = ["b", "s", "c", "t"]
COH_LABELS = ["0%", "6%", "20%", "50%"]
CONDITION_DICT = {
    "state_values": ["biased", "unbiased"],
    "coherences": [0, 0.06, 0.2, 0.5],
    "choices": ["awayRF", "toRF"],
}
SESSION_TO_EXCLUDE = ["210210_GP_JP", "241209_GP_TZ"]

NORM_METHOD = "zscore"        # per-neuron (std + SOFT_NORM_CONST)
SOFT_NORM_CONST = 5.0

SIG_KWARGS_SELF = dict(
    n_shuffles=100, n_splits=50, n_consecutive=1,
    keys=["b", "s", "c"],
    key_groups={"s": [[0, 1], [2, 3]]},
    smooth_sigma=10,
)
SIG_KWARGS_CROSS = dict(
    n_shuffles=100, n_splits=15, n_consecutive=1,
    keys=["b", "c"],
    smooth_sigma=10,
    cross_decode_keys={"b": ["b", "c"], "c": ["b", "c"]},
    seed=0,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _log(msg, indent=0):
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}]{'  ' * indent}{msg}", flush=True)


def _load_or_compute(path, compute_fn, label):
    if path.exists():
        with open(path, "rb") as f:
            masks = pickle.load(f)
        _log(f"[cache] {label}", indent=1)
    else:
        _log(f"[computing] {label} ...", indent=1)
        masks = compute_fn()
        with open(path, "wb") as f:
            pickle.dump(masks, f)
        _log(f"[saved] {label} → {path.name}", indent=1)
    return masks


def _compute_self_sig(results, fa, ftw):
    masks = {}
    for alignment in ALIGNMENTS:
        _log(f"self sig: {alignment} ...", indent=2)
        masks[alignment], _, _ = dpca_utils.dpca_significance_analysis(
            copy.deepcopy(results[alignment]["model"]),
            fa[alignment], ftw[alignment],
            **SIG_KWARGS_SELF,
        )
        _log(f"self sig: {alignment} done", indent=2)
    return masks


def _compute_cross_sig(results, fa, ftw, fit_aligns, proj_aligns):
    proj_Xs  = {pa: fa[pa]  for pa in proj_aligns}
    proj_tws = {pa: ftw[pa] for pa in proj_aligns}
    masks = {}
    for fit_align in fit_aligns:
        model = results[fit_align]["model"]
        _log(f"cross sig: fit={fit_align} → proj {proj_aligns} ...", indent=2)
        masks[fit_align], _, _ = dpca_utils.dpca_cross_significance_analysis(
            copy.deepcopy(model),
            fa[fit_align], ftw[fit_align],
            proj_Xs, proj_tws,
            fit_align=fit_align,
            **SIG_KWARGS_CROSS,
        )
        _log(f"cross sig: fit={fit_align} → done", indent=2)
    return masks


# ── Normalization ───────────────────────────────────────────────────────────────

def compute_neuron_scale(fa, alignments, method="zscore", soft_const=5.0):
    """One scale per neuron, pooled across all alignment periods. Shape (n_neurons,)."""
    n_neurons = fa[alignments[0]].shape[0]
    pooled = np.concatenate(
        [fa[a].reshape(n_neurons, -1) for a in alignments], axis=1
    )
    if method == "zscore":
        scale = np.nanstd(pooled, axis=1) + soft_const
    elif method == "softnorm_range":
        scale = (np.nanmax(pooled, axis=1) - np.nanmin(pooled, axis=1)) + soft_const
    else:
        raise ValueError(f"unknown method {method!r}")
    return np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)


def apply_neuron_scale(fa, ftw, scale, alignments):
    """Divide each neuron by its scale: averaged (neuron axis 0), trial-wise (neuron axis 1)."""
    fa_n, ftw_n = {}, {}
    for a in alignments:
        avg_shape = [1] * fa[a].ndim
        avg_shape[0] = scale.shape[0]
        fa_n[a] = fa[a] / scale.reshape(avg_shape)
        tw_shape = [1] * ftw[a].ndim
        tw_shape[1] = scale.shape[0]
        ftw_n[a] = ftw[a] / scale.reshape(tw_shape)
    return fa_n, ftw_n


def _prepare_data(sessions, neuron_metadata, state_trial_info, ephys):
    _log("getting neuron IDs ...", indent=1)
    nids = dpca_utils.get_neuron_ids(neuron_metadata, sessions)
    _log(f"{len(nids)} neurons selected", indent=1)

    _log("building dPCA matrix ...", indent=1)
    avg, tw = dpca_utils.create_dpca_matrix(
        sessions, CONDITION_DICT, nids,
        state_trial_info, neuron_metadata, ephys, ephys_config,
        condition_type="states",
    )
    _log("cleaning data ...", indent=1)
    fa, ftw, _, _ = dpca_utils.clean_dpca_data(avg, tw, ALIGNMENTS)

    _log(f"normalizing ({NORM_METHOD}) ...", indent=1)
    scale = compute_neuron_scale(fa, ALIGNMENTS, NORM_METHOD, SOFT_NORM_CONST)
    _log(f"scale: median={np.median(scale):.3f}, min={scale.min():.3f}, max={scale.max():.3f}", indent=1)
    fa, ftw = apply_neuron_scale(fa, ftw, scale, ALIGNMENTS)

    _log("fitting dPCA (all alignments) ...", indent=1)
    results = dpca_utils.fit_dpca_all_alignments(
        fa, ftw, ALIGNMENTS, marginalization_keys=MARGINALIZATION_KEYS,
    )
    _log("computing cross-period projections ...", indent=1)
    proj = dpca_utils.cross_period_projection(results, fa, ALIGNMENTS)
    ta = dpca_utils.build_time_axes(fa, ephys_config)
    return results, proj, ta, fa, ftw


# ── Analysis runner ─────────────────────────────────────────────────────────────

def run_all_neuron(sessions, group_tag, neuron_metadata, state_trial_info,
                   ephys, dpca_dir, out_base):
    _log(f"=== {group_tag} — all neurons (norm={NORM_METHOD}) ===")
    out_dir = out_base / f"dpca_{group_tag}_session_all_neuron_norm_{NORM_METHOD}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results, proj, ta, fa, ftw = _prepare_data(
        sessions, neuron_metadata, state_trial_info, ephys,
    )

    # Save per-alignment loadings: D (decoder — axes used for projection) and
    # P (encoder — per-neuron contribution, for inspecting which neurons drive each axis).
    loadings = {a: {"D": results[a]["model"].D, "P": results[a]["model"].P} for a in ALIGNMENTS}
    with open(out_dir / "loadings.pkl", "wb") as f:
        pickle.dump(loadings, f)
    _log("saved loadings.pkl", indent=1)

    sig = _load_or_compute(
        dpca_dir / f"{group_tag}_session_all_neuron_norm_{NORM_METHOD}_significance_masks.pkl",
        lambda: _compute_self_sig(results, fa, ftw),
        f"{group_tag} all-neuron self sig",
    )

    _log("plotting self-projection ...", indent=1)
    fig = dpca_plot_utils.plot_self_projection(
        proj, ta, ALIGNMENTS, ["s", "c", "b"],
        significance_masks=sig, PC=0, coh_labels=COH_LABELS,
        title=f"Self-projection — {group_tag} all neurons (norm {NORM_METHOD})",
    )
    fig.savefig(out_dir / "self_projection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log("saved self_projection.png", indent=1)

    # Combined cross sig: baseline, response, and visual fit epochs in one call/cache.
    # All project to ALIGNMENTS; the visual figure below uses a subset of those columns.
    cross_sig = _load_or_compute(
        dpca_dir / f"{group_tag}_session_all_neuron_norm_{NORM_METHOD}_cross_sig_masks_refit.pkl",
        lambda: _compute_cross_sig(results, fa, ftw, ["baseline", "response", "visual"], ALIGNMENTS),
        f"{group_tag} all-neuron cross sig",
    )
    # Diagonal-self cell (proj==fit, class==PC var) is skipped in cross; fill from self mask.
    for fit_align in ["baseline", "response", "visual"]:
        for k in ["b", "c"]:
            cross_sig[fit_align][fit_align][k][k] = sig[fit_align][k]

    for fit_align in ["baseline", "response"]:
        _log(f"plotting cross-projection ({fit_align}) ...", indent=1)
        fig = dpca_plot_utils.plot_cross_projection(
            proj, ta, ALIGNMENTS, ["b", "c"], fit_align,
            coh_labels=COH_LABELS,
            significance_masks=cross_sig[fit_align],
        )
        fig.savefig(out_dir / f"cross_projection_{fit_align}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        _log(f"saved cross_projection_{fit_align}.png", indent=1)

    # Visual fit plotted over a subset of projection epochs; masks were computed for all
    # alignments in the combined call above, so the subset column lookup is valid.
    visual_proj = ["visual", "cue", "response"]
    _log("plotting cross-projection (visual) ...", indent=1)
    fig = dpca_plot_utils.plot_cross_projection(
        proj, ta, visual_proj, ["b", "c"], "visual",
        coh_labels=COH_LABELS,
        significance_masks=cross_sig["visual"],
    )
    fig.savefig(out_dir / "cross_projection_visual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log("saved cross_projection_visual.png", indent=1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _log("Starting normalized dPCA analysis (zscore, toRF + awayRF, all neurons)")

    processed_dir = Path(dir_config.data.processed)
    compiled_dir  = Path(dir_config.data.compiled)
    dpca_dir = processed_dir / "dpca"
    dpca_dir.mkdir(parents=True, exist_ok=True)
    out_base = PROJECT_ROOT / "dissemination" / "dpca"

    _log("loading session and neuron metadata ...")
    session_metadata = pd.read_csv(processed_dir / "sessions_metadata.csv")
    session_metadata = session_metadata[
        ~session_metadata.session_id.isin(SESSION_TO_EXCLUDE)
    ].reset_index(drop=True)
    neuron_metadata = pd.read_csv(processed_dir / "neuron_metadata.csv")
    neuron_metadata = neuron_metadata[
        ~neuron_metadata.session_id.isin(SESSION_TO_EXCLUDE)
    ].reset_index(drop=True)

    _log("loading GLM-HMM model ...")
    with open(processed_dir / "glm_hmm_models" / "glm_hmm_masked_final.pkl", "rb") as f:
        glm_hmm = pickle.load(f)
    glm_hmm_original = copy.deepcopy(glm_hmm)

    _log("loading ephys data ...")
    with open(processed_dir / "ephys_neuron_wise.pkl", "rb") as f:
        ephys = pickle.load(f)

    _log("extracting HMM state trial info ...")
    biased_ti, unbiased_ti, _ = dpca_utils.extract_hmm_state_trial_info(
        session_metadata, glm_hmm_original, glm_hmm["data"], compiled_dir=compiled_dir,
    )
    state_trial_info = {"biased": biased_ti, "unbiased": unbiased_ti}

    toRF_sessions   = session_metadata.session_id[session_metadata.prior_direction == "toRF"]
    awayRF_sessions = session_metadata.session_id[session_metadata.prior_direction == "awayRF"]
    _log(f"toRF sessions: {len(toRF_sessions)},  awayRF sessions: {len(awayRF_sessions)}")

    for group_tag, sessions in [("toRF", toRF_sessions), ("awayRF", awayRF_sessions)]:
        run_all_neuron(
            sessions, group_tag, neuron_metadata, state_trial_info,
            ephys, dpca_dir, out_base,
        )

    _log("All analyses complete.")


if __name__ == "__main__":
    main()
