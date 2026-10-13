#!/usr/bin/env python
"""Run all dPCA analyses — toRF and awayRF sessions, all neurons + cell-type leave-out.

Usage:
    conda run -n prior-sc python scripts/dpca/run_dpca_analysis.py

Outputs (under dissemination/dpca/):
    dpca_{group}_session_all_neuron/
        variance_explained.png
        self_projection.png                  (plain style: all condition traces)
        cross_projection_ci_{baseline,response,visual}.png
                                             (CI style: 4 state x choice means + SEM bands)
    dpca_{group}_cell_type_leaveout/{tag}/
        self_projection_ci_{s,c,b}.png       (CI style: one figure per marginalization)

Significance masks are cached as .pkl in processed_dir/dpca/ and reused if present.
Figures are NOT cached — every run rewrites them. Use --only / --group to skip analyses
whose figures are already current; each analysis costs a dPCA refit.

    --only all-neuron | cell-type     default: both
    --group toRF | awayRF             default: both
"""

import argparse
import copy
import pickle
import shutil
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

# ── Constants ─────────────────────────────────────────────────────────────────

ALIGNMENTS = list(ephys_config["alignment_settings_GP"].keys())
MARGINALIZATION_KEYS = ["b", "s", "c", "t"]
COH_LABELS = ["0%", "6%", "20%", "50%"]
CONDITION_DICT = {
    "state_values": ["biased", "unbiased"],
    "coherences": [0, 0.06, 0.2, 0.5],
    "choices": ["awayRF", "toRF"],
}
SESSION_TO_EXCLUDE = ["210210_GP_JP", "241209_GP_TZ"]

CELL_TYPE_CONFIGS = [
    ("no_undefined",        ["trash", "undefined"]),
    ("no_visuomotor",       ["trash", "visuomotor"]),
    ("no_motor",            ["trash", "motor"]),
    ("no_visual_phasic",    ["trash", "visual_phasic"]),
    ("no_visual_tonic",     ["trash", "visual_tonic"]),
    ("no_visuomotor_motor", ["trash", "visuomotor", "motor"]),
]

SIG_KWARGS_SELF = dict(
    n_shuffles=100, n_splits=50, n_consecutive=1,
    keys=["b", "s", "c"],
    key_groups={"s": [[0, 1], [2, 3]]},
    smooth_sigma=10,
)
# Rigorous shared-trial cross-period significance: axes refit on 80% of FIT-epoch
# trials each split, held-out 20% decoded in the PROJ epoch (no trial leakage).
# refit already provides CV, so n_splits is much smaller than the self-sig path.
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
        # One call per fit epoch: axes refit once per split, all proj epochs decoded
        # through the shared axes. Returns masks[proj_align][proj_key][class_key].
        masks[fit_align], _, _ = dpca_utils.dpca_cross_significance_analysis(
            copy.deepcopy(model),
            fa[fit_align], ftw[fit_align],          # FIT epoch  → refit axes on 80%
            proj_Xs, proj_tws,                      # PROJ epochs decoded together
            fit_align=fit_align,                    # skip diagonal-self (== self-proj mask)
            **SIG_KWARGS_CROSS,
        )
        _log(f"cross sig: fit={fit_align} → done", indent=2)
    return masks


def _prepare_data(sessions, neuron_metadata, state_trial_info, ephys,
                  compiled_dir, exclude_cell_types=None):
    _log("getting neuron IDs ...", indent=1)
    nids = dpca_utils.get_neuron_ids(
        neuron_metadata, sessions,
        **({"exclude_cell_types": exclude_cell_types} if exclude_cell_types else {}),
    )
    _log(f"{len(nids)} neurons selected", indent=1)

    _log("building dPCA matrix ...", indent=1)
    avg, tw = dpca_utils.create_dpca_matrix(
        sessions, CONDITION_DICT, nids,
        state_trial_info, neuron_metadata, ephys, ephys_config,
        condition_type="states",
    )
    _log("cleaning data ...", indent=1)
    fa, ftw, _, _ = dpca_utils.clean_dpca_data(avg, tw, ALIGNMENTS)

    _log("fitting dPCA (all alignments) ...", indent=1)
    results = dpca_utils.fit_dpca_all_alignments(
        fa, ftw, ALIGNMENTS, marginalization_keys=MARGINALIZATION_KEYS,
    )
    _log("computing cross-period projections ...", indent=1)
    proj = dpca_utils.cross_period_projection(results, fa, ALIGNMENTS)
    ta = dpca_utils.build_time_axes(fa, ephys_config)
    _log("data preparation done", indent=1)
    return results, proj, ta, fa, ftw

# ── Analysis runners ──────────────────────────────────────────────────────────

def run_all_neuron(sessions, group_tag, neuron_metadata, state_trial_info,
                   ephys, compiled_dir, dpca_dir, out_base):
    _log(f"=== {group_tag} — all neurons ===")
    out_dir = out_base / f"dpca_{group_tag}_session_all_neuron"
    out_dir.mkdir(parents=True, exist_ok=True)

    results, proj, ta, fa, ftw = _prepare_data(
        sessions, neuron_metadata, state_trial_info, ephys, compiled_dir,
    )

    sig = _load_or_compute(
        dpca_dir / f"{group_tag}_session_all_neuron_significance_masks.pkl",
        lambda: _compute_self_sig(results, fa, ftw),
        f"{group_tag} all-neuron self sig",
    )

    _log("plotting self-projection ...", indent=1)
    fig = dpca_plot_utils.plot_self_projection(
        proj, ta, ALIGNMENTS, ["s", "c", "b"],
        significance_masks=sig, PC=0, coh_labels=COH_LABELS,
        title=f"Self-projection — {group_tag} all neurons",
    )
    fig.savefig(out_dir / "self_projection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"saved self_projection.png", indent=1)

    cross_sig = _load_or_compute(
        dpca_dir / f"{group_tag}_session_all_neuron_cross_sig_masks_refit.pkl",
        lambda: _compute_cross_sig(results, fa, ftw, ["baseline", "response"], ALIGNMENTS),
        f"{group_tag} all-neuron cross sig",
    )

    # The diagonal-self cell (proj==fit, class var == PC var) is skipped in the cross
    # computation because it equals the self-projection mask — fill it from `sig` for plotting.
    for fit_align in ["baseline", "response"]:
        for k in ["b", "c"]:
            cross_sig[fit_align][fit_align][k][k] = sig[fit_align][k]

    for fit_align in ["baseline", "response"]:
        _log(f"plotting cross-projection ({fit_align}) ...", indent=1)
        # Only the 4 state x choice means (coherence collapsed). One sign flip taken from
        # the fit epoch and applied to every projected column — see plot_cross_projection_ci.
        # show_band=False → bare mean traces; flip to True for the z*SEM bands.
        fig = dpca_plot_utils.plot_cross_projection_ci(
            proj, ta, ALIGNMENTS, ["b", "c"], fit_align,
            ftw, results,
            significance_masks=cross_sig[fit_align], z=1.96, show_band=False,
        )
        fig.savefig(out_dir / f"cross_projection_ci_{fit_align}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        _log(f"saved cross_projection_ci_{fit_align}.png", indent=1)

    # Visual fit → project on cue & response (separate cache so it doesn't recompute the
    # baseline/response cross sig above). The visual self column reuses the self-proj mask.
    visual_proj = ["visual", "cue", "response"]
    cross_sig_visual = _load_or_compute(
        dpca_dir / f"{group_tag}_session_all_neuron_cross_sig_visual_refit.pkl",
        lambda: _compute_cross_sig(results, fa, ftw, ["visual"], visual_proj),
        f"{group_tag} all-neuron cross sig (visual fit)",
    )
    for k in ["b", "c"]:
        cross_sig_visual["visual"]["visual"][k][k] = sig["visual"][k]

    _log("plotting cross-projection (visual) ...", indent=1)
    fig = dpca_plot_utils.plot_cross_projection_ci(
        proj, ta, visual_proj, ["b", "c"], "visual",
        ftw, results,
        significance_masks=cross_sig_visual["visual"], z=1.96, show_band=False,
    )
    fig.savefig(out_dir / "cross_projection_ci_visual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log("saved cross_projection_ci_visual.png", indent=1)


def run_cell_type_leaveout(sessions, group_tag, neuron_metadata, state_trial_info,
                           ephys, compiled_dir, dpca_dir, out_base):
    _log(f"=== {group_tag} — cell-type leave-out ===")
    out_root = out_base / f"dpca_{group_tag}_cell_type_leaveout"
    out_root.mkdir(parents=True, exist_ok=True)

    for i, (tag, exclude_types) in enumerate(CELL_TYPE_CONFIGS, 1):
        _log(f"config {i}/{len(CELL_TYPE_CONFIGS)}: {tag}", indent=1)
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        results, proj, ta, fa, ftw = _prepare_data(
            sessions, neuron_metadata, state_trial_info, ephys, compiled_dir,
            exclude_cell_types=exclude_types,
        )

        sig = _load_or_compute(
            dpca_dir / f"{group_tag}_cell_type_{tag}_significance_masks.pkl",
            lambda r=results, f=fa, ft=ftw: _compute_self_sig(r, f, ft),
            f"{group_tag} {tag} self sig",
        )

        # CI style: condition means only (show_thin=False) with a 95% SEM band, one figure
        # per marginalization. The thin per-condition traces are dropped here because the
        # overlapping bands are unreadable underneath them.
        _log("plotting self-projection ...", indent=2)
        figs = dpca_plot_utils.plot_self_projection_ci(
            proj, ta, ALIGNMENTS, ["s", "c", "b"],
            ftw, results,
            significance_masks=sig, PC=0, coh_labels=COH_LABELS,
            title=f"Self-projection ± SEM — {group_tag} exclude {exclude_types}",
            z=1.96, show_thin=False,
        )
        for marg, fig in figs.items():
            fig.savefig(out_dir / f"self_projection_ci_{marg}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        _log(f"saved self_projection_ci_{{{','.join(figs)}}}.png", indent=2)

# ── Main ──────────────────────────────────────────────────────────────────────

CACHE_DIR = Path.home() / ".cache" / "sc-priors"


def _cached_path(src, use_cache=True):
    """Return a local copy of `src`, copying it from the network share if needed.

    `processed_dir` lives on a network share that reads at roughly 1 MB/s, so the ~3.3 GB
    ephys pickle costs ~45 min per run. The copy is refreshed whenever the source's mtime
    or size differs, so an updated source is picked up automatically rather than silently
    serving a stale cache.
    """
    if not use_cache:
        return src
    src = Path(src)
    try:
        st = src.stat()
    except OSError as e:                       # share unmounted/unreachable
        dst = CACHE_DIR / src.name
        if dst.exists():
            _log(f"[cache] source unreachable ({e.strerror}); using local {dst}", indent=1)
            return dst
        raise

    dst = CACHE_DIR / src.name
    if dst.exists():
        d = dst.stat()
        # mtime can drift by a second across filesystems; size is the stronger check
        if d.st_size == st.st_size and abs(d.st_mtime - st.st_mtime) < 2:
            _log(f"[cache] hit  {dst}  ({st.st_size / 1e9:.2f} GB)", indent=1)
            return dst
        _log(f"[cache] stale (source changed) — refreshing {dst.name}", indent=1)
    else:
        _log(f"[cache] miss — copying {src.name} ({st.st_size / 1e9:.2f} GB) to {CACHE_DIR}",
             indent=1)
        _log("        one-time cost; later runs read it locally", indent=1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".partial")
    try:
        shutil.copy2(src, tmp)                 # copy2 preserves mtime, so the check above works
        tmp.replace(dst)                       # atomic: a killed copy never leaves a valid-looking cache
    finally:
        tmp.unlink(missing_ok=True)
    _log(f"[cache] stored {dst}", indent=1)
    return dst


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="dPCA analysis: all-neuron and cell-type leave-out figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Figures are always rewritten (no existence check) — only the significance\n"
            "masks are cached. Use --only/--group to avoid redoing analyses whose figures\n"
            "are already current, since each analysis costs a dPCA refit.\n\n"
            "Examples:\n"
            "  --only all-neuron              2 refits; self- and cross-projection figures\n"
            "  --only cell-type --group toRF  6 refits; toRF cell-type figures only\n"
        ),
    )
    p.add_argument(
        "--only", choices=["all-neuron", "cell-type"], default=None,
        help="run just one analysis (default: both)",
    )
    p.add_argument(
        "--group", choices=["toRF", "awayRF"], default=None,
        help="run just one session group (default: both)",
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help=f"read the ephys pickle straight from the share instead of {CACHE_DIR}",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    _log("Starting dPCA analysis script")
    _log(f"only  : {args.only or 'all-neuron + cell-type'}")
    _log(f"group : {args.group or 'toRF + awayRF'}")

    processed_dir = Path(dir_config.data.processed)
    compiled_dir  = Path(dir_config.data.compiled)
    dpca_dir = processed_dir / "dpca"
    dpca_dir.mkdir(parents=True, exist_ok=True)
    out_base = PROJECT_ROOT / "dissemination" / "dpca"
    _log(f"processed_dir : {processed_dir}")
    _log(f"output base   : {out_base}")

    _log("loading session and neuron metadata ...")
    session_metadata = pd.read_csv(processed_dir / "sessions_metadata.csv")
    session_metadata = session_metadata[
        ~session_metadata.session_id.isin(SESSION_TO_EXCLUDE)
    ].reset_index(drop=True)
    _log(f"{len(session_metadata)} sessions loaded")

    neuron_metadata = pd.read_csv(processed_dir / "neuron_metadata.csv")
    neuron_metadata = neuron_metadata[
        ~neuron_metadata.session_id.isin(SESSION_TO_EXCLUDE)
    ].reset_index(drop=True)
    _log(f"{len(neuron_metadata)} neurons loaded")

    _log("loading GLM-HMM model ...")
    with open(processed_dir / "glm_hmm_models" / "glm_hmm_masked_final.pkl", "rb") as f:
        glm_hmm = pickle.load(f)
    glm_hmm_original = copy.deepcopy(glm_hmm)

    _log("loading ephys data ...")
    ephys_path = _cached_path(processed_dir / "ephys_neuron_wise.pkl",
                              use_cache=not args.no_cache)
    with open(ephys_path, "rb") as f:
        ephys = pickle.load(f)

    _log("extracting HMM state trial info ...")
    data = glm_hmm["data"]
    biased_ti, unbiased_ti, _ = dpca_utils.extract_hmm_state_trial_info(
        session_metadata, glm_hmm_original, data, compiled_dir=compiled_dir,
    )
    state_trial_info = {"biased": biased_ti, "unbiased": unbiased_ti}
    _log("HMM state trial info ready")

    toRF_sessions   = session_metadata.session_id[session_metadata.prior_direction == "toRF"]
    awayRF_sessions = session_metadata.session_id[session_metadata.prior_direction == "awayRF"]
    _log(f"toRF sessions: {len(toRF_sessions)},  awayRF sessions: {len(awayRF_sessions)}")

    groups = [("toRF", toRF_sessions), ("awayRF", awayRF_sessions)]
    if args.group:
        groups = [g for g in groups if g[0] == args.group]

    for group_tag, sessions in groups:
        if args.only in (None, "all-neuron"):
            run_all_neuron(
                sessions, group_tag, neuron_metadata, state_trial_info,
                ephys, compiled_dir, dpca_dir, out_base,
            )
        if args.only in (None, "cell-type"):
            run_cell_type_leaveout(
                sessions, group_tag, neuron_metadata, state_trial_info,
                ephys, compiled_dir, dpca_dir, out_base,
            )

    _log("All analyses complete.")


if __name__ == "__main__":
    main()
