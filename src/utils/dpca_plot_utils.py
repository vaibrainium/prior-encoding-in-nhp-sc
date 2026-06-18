"""
Plotting utilities for dPCA results.

Public functions
----------------
plot_variance_explained   – cumulative variance explained per marginalization and alignment
plot_dpca_traces          – plot all condition traces for one PC onto an axes
bar_significance          – draw a significance bar at the bottom of an axes
plot_self_projection      – margs × alignments self-projection grid
plot_cross_projection     – cross-period projection with highlighted self-projection panel
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Color palettes ─────────────────────────────────────────────────────────────
# 4-coherence palettes; slice [:3] for 3-coherence fits
BIASED_COLORS   = ["#96D6EC", "#6FC3EB", "#5289C6", "#4469B1"]
UNBIASED_COLORS = ["#F2A448", "#EF8D41", "#EC6A50", "#AC3626"]

MARG_LABELS  = {"b": "GLM State", "s": "Stimulus", "c": "Choice", "t": "Time"}
ALIGN_LABELS = {"baseline": "Baseline", "visual": "Visual",
                "cue": "Cue", "response": "Response"}

# Thick mean-trace grey palette for stimulus (light→dark per coherence)
_STIM_GREYS = ["#d0d0d0", "#a0a0a0", "#707070", "#404040"]


def plot_variance_explained(dpca_results, alignments, margs_to_plot, n_components=3):
    """Cumulative variance explained per marginalization and alignment."""
    fig, axes = plt.subplots(len(alignments), len(margs_to_plot),
                             figsize=(4 * len(margs_to_plot), 3 * len(alignments)), sharey=True)
    for row, alignment in enumerate(alignments):
        evr = dpca_results[alignment]["model"].explained_variance_ratio_
        for col, marg in enumerate(margs_to_plot):
            ax = axes[row, col]
            cumvar = np.cumsum(evr[marg]) / np.sum(evr[marg])
            ax.plot(np.arange(1, n_components + 1), cumvar, marker="o")
            ax.axhline(0.95, color="r", linestyle="--", linewidth=1)
            ax.set_ylim(0, 1.1)
            ax.set_xticks(np.arange(1, n_components + 1))
            if row == 0:
                ax.set_title(MARG_LABELS.get(marg, marg))
            if col == 0:
                ax.set_ylabel(ALIGN_LABELS.get(alignment, alignment))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle("Cumulative explained variance (red = 95%)", y=1.01)
    plt.tight_layout()
    return fig


def plot_dpca_traces(ax, time, data, n_states=2, n_coh=4,
                     biased_colors=None, unbiased_colors=None, alpha=1.0):
    """Plot all (state × coh × choice) traces for one PC onto ax.

    data : (n_states, n_coh, n_choices, n_time) as returned by dpca_transform.
    c=0 → awayRF choice (dashed), c=1 → toRF choice (solid).
    """
    if biased_colors is None:
        biased_colors = BIASED_COLORS[:n_coh]
    if unbiased_colors is None:
        unbiased_colors = UNBIASED_COLORS[:n_coh]
    ls_map = ["--", "-"]  # c=0 awayRF=dashed, c=1 toRF=solid
    for b in range(n_states):
        colors = biased_colors if b == 0 else unbiased_colors
        for s in range(n_coh):
            for c in range(2):
                ax.plot(time, data[b, s, c, :].reshape(-1),
                        color=colors[s], linestyle=ls_map[c], linewidth=1.5, alpha=alpha)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def overlay_thick_means(ax, time, data_pc, n_coh=4,
                        biased_colors=None, unbiased_colors=None, linewidth=2.5):
    """Overlay 4 thick mean traces — state (biased/unbiased) × choice (toRF/awayRF) — each
    collapsed across coherence. State = color (dark biased/unbiased shade), choice = linestyle
    (toRF solid, awayRF dashed).  data_pc : (n_states, n_coh, n_choices, n_time)."""
    if biased_colors is None:
        biased_colors = BIASED_COLORS[:n_coh]
    if unbiased_colors is None:
        unbiased_colors = UNBIASED_COLORS[:n_coh]
    mean_data = np.nanmean(data_pc, axis=1)   # (n_states, n_choices, n_time): collapse coherence
    for b in range(mean_data.shape[0]):
        color = biased_colors[-1] if b == 0 else unbiased_colors[-1]
        for c, ls in enumerate(["--", "-"]):  # c=0 awayRF dashed, c=1 toRF solid
            ax.plot(time, mean_data[b, c], color=color, linestyle=ls, linewidth=linewidth)


SIG_COLORS = {"b": "#2CA02C", "c": "#E6177A"}  # state=green, choice=magenta-pink
                                               # (avoid blue/orange-red trace palettes)


def bar_significance(ax, time, mask, y_offset=0.0, height=0.04, color="black"):
    """Draw a thin significance bar at the bottom of ax at significant timepoints."""
    ax.fill_between(
        time, y_offset, y_offset + height,
        where=mask,
        transform=ax.get_xaxis_transform(),
        color=color, alpha=1, zorder=5,
    )


def plot_self_projection(projections, time_axes, alignments, margs_to_plot,
                         significance_masks=None, PC=0,
                         n_states=2, n_coh=4,
                         biased_colors=None, unbiased_colors=None,
                         coh_labels=None, title=None, trace_alpha=0.5):
    """Self-projection figure: margs_to_plot × alignments grid, PC traces.

    Individual condition traces are plotted at trace_alpha (default 0.5).
    Thick mean traces (mean over coherences) are overlaid at full opacity.
    Row order is determined by margs_to_plot (suggest ['s', 'c', 'b']).
    """
    if biased_colors is None:
        biased_colors = BIASED_COLORS[:n_coh]
    if unbiased_colors is None:
        unbiased_colors = UNBIASED_COLORS[:n_coh]
    if coh_labels is None:
        coh_labels = [f"coh{i}" for i in range(n_coh)]

    n_rows, n_cols = len(margs_to_plot), len(alignments)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    for col, alignment in enumerate(alignments):
        Z    = projections[alignment][alignment]
        time = time_axes[alignment]
        for row, marg in enumerate(margs_to_plot):
            if n_rows > 1 and n_cols > 1:
                ax = axes[row, col]
            elif n_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row]
            plot_dpca_traces(ax, time, Z[marg][PC], n_states, n_coh,
                             biased_colors, unbiased_colors, alpha=trace_alpha)
            # Thick mean traces — collapse across non-primary variables per marg
            data_pc = Z[marg][PC]  # (n_states, n_coh, n_choices, n_time)
            if marg == 's':
                mean_s = np.nanmean(data_pc, axis=(0, 2))  # (n_coh, n_time)
                for s in range(n_coh):
                    ax.plot(time, mean_s[s, :], color=_STIM_GREYS[s], linewidth=2.5)
            elif marg == 'c':
                mean_c = np.nanmean(data_pc, axis=(0, 1))  # (n_choices, n_time)
                for c, ls in enumerate(["--", "-"]):  # awayRF=dashed, toRF=solid
                    ax.plot(time, mean_c[c, :], color="black", linestyle=ls, linewidth=2.5)
            elif marg == 'b':
                mean_b = np.nanmean(data_pc, axis=(1, 2))  # (n_states, n_time)
                state_thick = [biased_colors[-1], unbiased_colors[-1]]  # 50% coh shade per state
                for b in range(mean_b.shape[0]):
                    ax.plot(time, mean_b[b, :], color=state_thick[b], linewidth=2.5)
            else:
                mean_data = np.nanmean(data_pc, axis=1)  # (n_states, n_choices, n_time)
                for b in range(n_states):
                    mc = biased_colors[-1] if b == 0 else unbiased_colors[-1]
                    for c in range(2):
                        ls = "--" if c == 0 else "-"
                        ax.plot(time, mean_data[b, c, :], color=mc, linestyle=ls, linewidth=2.5)
            if significance_masks is not None and marg in significance_masks.get(alignment, {}):
                bar_significance(ax, time, significance_masks[alignment][marg][PC])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(ALIGN_LABELS.get(alignment, alignment), fontsize=13)
            if col == 0:
                ax.set_ylabel(f"{MARG_LABELS.get(marg, marg)} PC{PC + 1}", fontsize=12)
            if row == n_rows - 1:
                ax.set_xlabel("Time aligned to event (ms)", fontsize=11)

    if n_states == 1:
        color_handles = [plt.Line2D([], [], color=biased_colors[i], label=f"{coh_labels[i]} coh") for i in range(n_coh)]
    else:
        color_handles = (
            [plt.Line2D([], [], color=biased_colors[i],   label=f"{coh_labels[i]} coh biased state")   for i in range(n_coh)] +
            [plt.Line2D([], [], color=unbiased_colors[i], label=f"{coh_labels[i]} coh unbiased state") for i in range(n_coh)]
        )
    style_handles = [
        plt.Line2D([], [], color="k", linestyle="-",  label="toRF choice"),
        plt.Line2D([], [], color="k", linestyle="--", label="awayRF choice"),
        plt.Rectangle((0, 0), 1, 0.05, fc="black", label="significant"),
    ]
    fig.legend(handles=color_handles + style_handles,
               bbox_to_anchor=(1.01, 0.9), loc="upper left", fontsize=10)
    if title:
        fig.suptitle(title, y=1.01)
    plt.tight_layout()
    return fig


def plot_cross_projection(projections, time_axes, alignments, margs_to_plot, fit_align,
                          PC=0, n_coh=4, coh_labels=None,
                          biased_colors=None, unbiased_colors=None,
                          significance_masks=None):
    """Cross-period projection: each row is a marginalization, each column an alignment period.

    The self-projection panel (proj_align == fit_align) is highlighted with an orange border.
    significance_masks : dict[proj_align][marg] → (n_components, n_time) bool, optional
    """
    if biased_colors is None:
        biased_colors = BIASED_COLORS[:n_coh]
    if unbiased_colors is None:
        unbiased_colors = UNBIASED_COLORS[:n_coh]
    if coh_labels is None:
        coh_labels = [f"coh{i}" for i in range(n_coh)]

    n_rows, n_cols = len(margs_to_plot), len(alignments)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    for row, marg in enumerate(margs_to_plot):
        for col, proj_align in enumerate(alignments):
            ax = axes[row, col] if n_rows > 1 else axes[col]
            plot_dpca_traces(ax, time_axes[proj_align],
                             projections[fit_align][proj_align][marg][PC],
                             n_coh=n_coh,
                             biased_colors=biased_colors,
                             unbiased_colors=unbiased_colors, alpha=0.5)
            # 4 thick mean traces (state × choice, collapsed across coherence)
            overlay_thick_means(ax, time_axes[proj_align],
                                projections[fit_align][proj_align][marg][PC],
                                n_coh=n_coh, biased_colors=biased_colors,
                                unbiased_colors=unbiased_colors)
            if significance_masks is not None:
                panel_masks = significance_masks.get(proj_align, {}).get(marg, {})
                if isinstance(panel_masks, dict):
                    # cross_decode format: masks[proj_key][class_key][PC]
                    # Bars hug the bottom edge (tight, low offsets) so they don't cover traces.
                    for y_off, ck in zip([0.025, 0.0], ["b", "c"]):
                        if ck in panel_masks:
                            bar_significance(ax, time_axes[proj_align], panel_masks[ck][PC],
                                             y_offset=y_off, height=0.025,
                                             color=SIG_COLORS.get(ck, "black"))
                elif panel_masks is not None and hasattr(panel_masks, "__len__"):
                    bar_significance(ax, time_axes[proj_align], panel_masks[PC])
            # open empty space below the traces for the sig bars (drawn at axes-fraction
            # 0–0.05) so they sit under the data instead of overlapping it
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin - 0.12 * (ymax - ymin), ymax)
            if row == 0:
                ax.set_title(ALIGN_LABELS.get(proj_align, proj_align), fontsize=12)
            if col == 0:
                ax.set_ylabel(f"{MARG_LABELS.get(marg, marg)} PC{PC + 1}", fontsize=12)
            if row == n_rows - 1:
                ax.set_xlabel("Time aligned to event (ms)", fontsize=11)
            if proj_align == fit_align:
                for spine in ax.spines.values():
                    spine.set_edgecolor("#e05c00")
                    spine.set_linewidth(2)

    handles = (
        [plt.Line2D([], [], color=biased_colors[i],   label=f"{coh_labels[i]} coh biased state")   for i in range(n_coh)] +
        [plt.Line2D([], [], color=unbiased_colors[i], label=f"{coh_labels[i]} coh unbiased state") for i in range(n_coh)] +
        [plt.Line2D([], [], color="k", linestyle="-",  label="toRF choice"),
         plt.Line2D([], [], color="k", linestyle="--", label="awayRF choice"),
         plt.Rectangle((0, 0), 1, 0.04, fc=SIG_COLORS["b"], label="GLM state sig"),
         plt.Rectangle((0, 0), 1, 0.04, fc=SIG_COLORS["c"], label="choice sig")]
    )
    fig.legend(handles=handles, bbox_to_anchor=(1.01, 0.9), loc="upper left", fontsize=10)
    fig.suptitle(
        f"Cross-period projection — fit on {ALIGN_LABELS.get(fit_align, fit_align)}"
        "  (orange border = self-projection)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    return fig
