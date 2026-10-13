"""
Plotting utilities for dPCA results.

Public functions
----------------
plot_variance_explained   – cumulative variance explained per marginalization and alignment
plot_dpca_traces          – plot all condition traces for one PC onto an axes
bar_significance          – draw a significance bar at the bottom of an axes
plot_self_projection      – margs × alignments self-projection grid
plot_self_projection_ci   – one figure per marg; condition means with SEM bands
plot_cross_projection     – cross-period projection with highlighted self-projection panel
plot_cross_projection_ci  – cross-period projection, 4 state × choice means with SEM bands

Helpers shared with the notebook-local self-projection-CI figure
----------------------------------------------------------------
thick_line_sem            – analytic SEM of a condition-balanced marginal projection
projection_sign_flip      – ±1 so a panel satisfies the requested condition ordering
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


# ── Analytic SEM of a thick condition mean ────────────────────────────────────
# cell axes of a trial-wise matrix after trial-averaging: 0=neuron 1=state(b) 2=coh(s)
# 3=choice(c) 4=time.  For each marginalization, collapse the NON-key conditions with
# equal weight per condition (matches dpca_utils.dpca_significance_analysis):
MARG_COLLAPSE = {'s': (1, 3), 'c': (1, 2), 'b': (2, 3)}

# cross-projection groups by state × choice, so only coherence is collapsed:
CROSS_COLLAPSE = (2,)


def thick_line_sem(trialX, w, collapse_axes):
    """Analytic SEM of the condition-balanced marginal projection (the thick mean line).

        thick(v,t) = sum_n w_n * Xbar_n(v,t),   w_n = D[marg][:, PC]
        var(thick) = sum_n w_n^2 * (1/K^2) * sum_cells( s2_cell / n_cell )

    trialX        : (n_trial, n_neuron, n_state, n_coh, n_choice, n_time)  = trial-wise matrix
    w             : (n_neuron,) axis loading = dpca_results[align]['model'].D[marg][:, PC]
    collapse_axes : non-key condition axes to average over (cell space, see MARG_COLLAPSE)
    returns sem   : (*target_shape, n_time)   aligned with the thick-line order
    """
    cell_mean = np.nanmean(trialX, axis=0)                 # (neuron, b, s, c, time)
    cell_var  = np.nanvar(trialX,  axis=0, ddof=1)         # per-cell trial variance
    cell_cnt  = np.sum(~np.isnan(trialX), axis=0)          # per-cell trial count
    with np.errstate(invalid='ignore', divide='ignore'):
        cell_sem2 = cell_var / cell_cnt                    # SEM^2 of each cell mean
    K = np.sum(~np.isnan(cell_mean), axis=collapse_axes, keepdims=True)  # #valid cells
    with np.errstate(invalid='ignore', divide='ignore'):
        var_Xbar = np.nansum(cell_sem2, axis=collapse_axes, keepdims=True) / K**2
    var_Xbar = np.squeeze(var_Xbar, axis=collapse_axes)    # (neuron, *target, time)
    var_thick = np.tensordot(w**2, var_Xbar, axes=([0], [0]))  # (*target, time)
    return np.sqrt(var_thick)


def projection_sign_flip(projections, alignment, marg, PC):
    """Sign (±1) for this (alignment, marg, PC) so the epoch's OWN panel satisfies
    toRF > awayRF (c), biased > unbiased (b), high coh > low coh (s). dPCA fits each
    alignment's axes independently, so the sign is independently arbitrary per epoch."""
    data_pc = projections[alignment][alignment][marg][PC]  # (n_states, n_coh, n_choices, n_time)
    if marg == 's':
        m = np.nanmean(data_pc, axis=(0, 2))          # (n_coh, time), 0=lowest..-1=highest coh
        diff = np.nanmean(m[-1] - m[0])
    elif marg == 'c':
        m = np.nanmean(data_pc, axis=(0, 1))          # (n_choices, time), 0=awayRF,1=toRF
        diff = np.nanmean(m[1] - m[0])
    else:  # 'b'
        m = np.nanmean(data_pc, axis=(1, 2))          # (n_states, time), 0=biased,1=unbiased
        diff = np.nanmean(m[0] - m[1])
    return -1.0 if diff < 0 else 1.0


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


# Sequential purple ramp (light→dark = low→high coherence) for the CI self-projection,
# instead of the greyscale _STIM_GREYS — greys were nearly indistinguishable once the
# overlapping semi-transparent CI bands blended together. Purple stays visually distinct from
# the blue/orange state palettes. 0% coh's shade must stay dark enough to be visible against
# the white figure background — a near-white ramp start reads as a blank "white trace".
STIM_CI_COLORS = ["#c4c2df", "#9e9ac8", "#6a51a3", "#3f007d"]


def _plot_self_projection_ci_marg(ax_row, marg, alignments, projections, time_axes,
                                  fit_tw, dpca_results, significance_masks, PC,
                                  n_states, n_coh, coh_labels, biased_colors, unbiased_colors,
                                  trace_alpha, z, show_thin):
    """Draw one marg's row of subplots (one alignment per column) onto ax_row."""
    n_cols = len(alignments)
    for col, alignment in enumerate(alignments):
        Z, time = projections[alignment][alignment], time_axes[alignment]
        D_model, trialX = dpca_results[alignment]["model"].D, fit_tw[alignment]
        ax = ax_row[col]
        flip = projection_sign_flip(projections, alignment, marg, PC)
        data_pc = flip * Z[marg][PC]                       # (n_states, n_coh, n_choices, n_time)
        if show_thin:
            plot_dpca_traces(ax, time, data_pc, n_states, n_coh,
                             biased_colors, unbiased_colors, alpha=trace_alpha)
        sem = thick_line_sem(trialX, D_model[marg][:, PC], MARG_COLLAPSE[marg])  # (target,time)

        if marg == 's':
            mean = np.nanmean(data_pc, axis=(0, 2))        # (n_coh, time)
            colors, styles = [STIM_CI_COLORS[s] for s in range(n_coh)], ["-"] * n_coh
            labels = [f"{coh_labels[s]} coh" for s in range(n_coh)]
        elif marg == 'c':
            mean = np.nanmean(data_pc, axis=(0, 1))        # (n_choices, time)  0=awayRF,1=toRF
            colors, styles = ["black", "black"], ["--", "-"]
            labels = ["awayRF choice", "toRF choice"]
        else:  # 'b'
            mean = np.nanmean(data_pc, axis=(1, 2))        # (n_states, time)
            colors, styles = [biased_colors[-1], unbiased_colors[-1]], ["-"] * n_states
            labels = ["biased state", "unbiased state"]
        for k in range(mean.shape[0]):
            ax.plot(time, mean[k], color=colors[k], linestyle=styles[k], linewidth=2.5,
                    label=labels[k])
            ax.fill_between(time, mean[k] - z * sem[k], mean[k] + z * sem[k],
                            color=colors[k], alpha=0.22, linewidth=0)
        if col == n_cols - 1:                          # one legend per row, outside the figure
            ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

        if significance_masks is not None and marg in significance_masks.get(alignment, {}):
            bar_significance(ax, time, significance_masks[alignment][marg][PC])
        if not show_thin:
            ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
        ax.set_yticks([])
        ax.set_title(ALIGN_LABELS.get(alignment, alignment), fontsize=13)
        if col == 0:
            ax.set_ylabel(f"{MARG_LABELS.get(marg, marg)} PC{PC + 1}", fontsize=12)
        ax.set_xlabel("Time aligned to event (ms)", fontsize=11)


def plot_self_projection_ci(projections, time_axes, alignments, margs_to_plot,
                            fit_tw, dpca_results, significance_masks=None, PC=0,
                            n_states=2, n_coh=4, coh_labels=None, title=None,
                            trace_alpha=0.15, z=1.96, show_thin=True, show=False):
    """plot_self_projection + a z*SEM shaded band on each thick condition mean, split into
    ONE FIGURE PER MARG (returns {marg: fig}) instead of one combined figure — each figure
    has alignments as columns.

    The band is the analytic SEM of the condition-balanced marginal projection (the thick
    line), propagated from per-cell trial statistics in fit_tw — the same trials the
    significance test uses. Bands pulling apart = the condition means separate = encoding
    onset for that variable.

    Each (alignment, marg, PC) panel gets its OWN sign flip (see projection_sign_flip) so
    toRF>awayRF, biased>unbiased, high coh>low coh in every panel — dPCA fits each alignment
    independently, so a single sign aggregated across epochs can leave an individual epoch's
    ordering backwards even when most epochs are correct.

    show_thin : also draw the faint per-condition traces underneath the means.
    show      : call plt.show() per figure (notebook use); off by default for scripts.
    """
    biased_colors, unbiased_colors = BIASED_COLORS[:n_coh], UNBIASED_COLORS[:n_coh]
    if coh_labels is None:
        coh_labels = [f"coh{i}" for i in range(n_coh)]
    n_cols = len(alignments)

    figs = {}
    for marg in margs_to_plot:
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), squeeze=False)
        _plot_self_projection_ci_marg(
            axes[0], marg, alignments, projections, time_axes, fit_tw, dpca_results,
            significance_masks, PC, n_states, n_coh, coh_labels,
            biased_colors, unbiased_colors, trace_alpha, z, show_thin,
        )
        marg_title = (f"{title} — {MARG_LABELS.get(marg, marg)}" if title
                      else MARG_LABELS.get(marg, marg))
        fig.suptitle(marg_title, y=1.05)
        fig.tight_layout(rect=[0, 0, 0.88, 1])  # leave room on the right for the row legend
        if show:
            plt.show()
        figs[marg] = fig
    return figs


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


def plot_cross_projection_ci(projections, time_axes, alignments, margs_to_plot, fit_align,
                             fit_tw, dpca_results, significance_masks=None, PC=0,
                             n_coh=4, z=1.96, title=None, show_band=True):
    """Cross-period projection showing ONLY the 4 state × choice means, each with a z*SEM band.

    Differs from plot_cross_projection (which scripts/dpca/* still use) in three ways:
      * no thin per-condition traces — just the 2 states × 2 choices, collapsed over coherence
      * a z*SEM shaded band on each mean, from thick_line_sem on the PROJ epoch's trials with
        the FIT epoch's loading w = D[marg][:, PC]. That product is exactly what the cross
        projection computes, so the band is the analytic SEM of the line it wraps.
      * ONE sign flip per (fit_align, marg, PC), decided on the fit epoch's own panel and
        applied to every projected column. Flipping per column (as the self-projection figure
        does) would hide the very thing a cross-projection asks: whether the other epochs
        preserve the fit epoch's ordering.

    State = color (biased/unbiased), choice = linestyle (toRF solid, awayRF dashed).
    show_band         : draw the z*SEM bands. Set False for the 4 bare mean traces — the
                        SEM is then not computed at all, so fit_tw may be omitted (None).
    fit_tw            : dict[align] → trial-wise matrix, one entry per projected alignment
    significance_masks: dict[proj_align][marg][class_key] → (n_components, n_time) bool,
                        the cross_decode format (see dpca_utils.dpca_cross_significance_analysis)
    """
    biased_colors, unbiased_colors = BIASED_COLORS[:n_coh], UNBIASED_COLORS[:n_coh]
    state_colors  = [biased_colors[-1], unbiased_colors[-1]]   # 0=biased, 1=unbiased
    state_labels  = ["biased", "unbiased"]
    choice_styles = ["--", "-"]                                # c=0 awayRF, c=1 toRF
    choice_labels = ["awayRF", "toRF"]

    n_rows, n_cols = len(margs_to_plot), len(alignments)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for row, marg in enumerate(margs_to_plot):
        flip = projection_sign_flip(projections, fit_align, marg, PC)  # fit epoch decides once
        w = dpca_results[fit_align]["model"].D[marg][:, PC]
        for col, proj_align in enumerate(alignments):
            ax, time = axes[row, col], time_axes[proj_align]
            data_pc = flip * projections[fit_align][proj_align][marg][PC]  # (b, s, c, t)
            mean = np.nanmean(data_pc, axis=1)                             # (b, c, t)
            sem = thick_line_sem(fit_tw[proj_align], w, CROSS_COLLAPSE) if show_band else None
            for b in range(mean.shape[0]):
                for c in range(mean.shape[1]):
                    ax.plot(time, mean[b, c], color=state_colors[b],
                            linestyle=choice_styles[c], linewidth=2.5)
                    if show_band:
                        ax.fill_between(time,
                                        mean[b, c] - z * sem[b, c], mean[b, c] + z * sem[b, c],
                                        color=state_colors[b], alpha=0.2, linewidth=0)

            if significance_masks is not None:
                panel_masks = significance_masks.get(proj_align, {}).get(marg, {})
                # bars hug the bottom edge (tight, low offsets) so they don't cover the bands
                for y_off, ck in zip([0.025, 0.0], ["b", "c"]):
                    if ck in panel_masks:
                        bar_significance(ax, time, panel_masks[ck][PC],
                                         y_offset=y_off, height=0.025,
                                         color=SIG_COLORS.get(ck, "black"))
            # open empty space below the traces for the sig bars (axes-fraction 0–0.05)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin - 0.12 * (ymax - ymin), ymax)
            ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_yticks([])
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

    handles = [
        plt.Line2D([], [], color=state_colors[b], linestyle=choice_styles[c], linewidth=2.5,
                   label=f"{state_labels[b]} / {choice_labels[c]}")
        for b in range(2) for c in range(2)
    ] + [
        plt.Rectangle((0, 0), 1, 0.04, fc=SIG_COLORS["b"], label="GLM state sig"),
        plt.Rectangle((0, 0), 1, 0.04, fc=SIG_COLORS["c"], label="choice sig"),
    ]
    fig.legend(handles=handles, bbox_to_anchor=(1.01, 0.9), loc="upper left", fontsize=10)
    default_title = "Cross-period projection ± SEM" if show_band else "Cross-period projection"
    fig.suptitle(
        f"{title or default_title}"
        f" — fit on {ALIGN_LABELS.get(fit_align, fit_align)}"
        "  (orange border = self-projection)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    return fig
