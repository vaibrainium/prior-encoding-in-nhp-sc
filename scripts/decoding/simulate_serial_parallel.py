"""Idealized predicted decoding time courses under serial vs parallel processing.

This is a SCHEMATIC, not a simulation of neural activity. Accuracy curves are generated
directly from a stated generative form — no spikes, no decoder, no noise. Its job is to
fix what each hypothesis predicts so the real 8.10/8.11 results can be read against it.
It cannot tell you whether the real pipeline has the power to distinguish them; that
needs synthetic spikes pushed through the actual decoder.

Generative form
---------------
Each variable v has a gain profile g_v(t) in [0, 1] — how strongly it is encoded at time t:

    acc_v(t) = 0.5 + A * g_v(t)

The hypotheses, stated as gain profiles:
  serial   — the stimulus representation is transient: it rises after stimulus onset,
             then decays within the cue epoch. Choice only rises as it fades. The
             sensory code is handed off to a decision code.
  parallel — the stimulus representation is sustained: it rises and is still strongly
             encoded at the response. Choice is ALREADY above chance before the stimulus
             arrives (the prior biases the upcoming choice) and is driven further up once
             sensory evidence appears, so the two are co-active throughout.

Under both hypotheses the stimulus is at chance until stimulus onset plus a sensory
latency — the targets carry no coherence information.

Epoch geometry comes from config/ephys-config.yaml. The alignment events matter: "visual"
is TARGET onset and carries no coherence information, so stimulus decoding is at chance
there under BOTH hypotheses — the decision stimulus does not appear until stimulus_onset,
which is t = 0 inside the "cue" window.

Run:  python scripts/decoding/simulate_serial_parallel.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "dissemination" / "decoding"

# ── Palette ───────────────────────────────────────────────────────────────────
# Categorical slots 1 and 4 of the validated reference palette. Checked with
# validate_palette.js: CVD dE 31.5 (protan), normal-vision dE 37.5, chroma and lightness
# in band. Yellow warns on contrast vs a light surface, which is why both series are
# direct-labelled as well as legended — identity is never colour-alone.
C_STIM = "#2a78d6"    # blue
C_CHOICE = "#eda100"  # yellow
INK = "#252525"
MUTED = "#8a8a85"

A = 0.42              # peak accuracy above chance -> 0.92 at full gain
CHANCE = 0.5

# Real epoch geometry from config/ephys-config.yaml (alignment_settings_GP), baseline
# dropped as in 8.20.
EPOCHS = [("Visual", "target onset", 0, 250),
          ("Cue", "stimulus onset", -100, 800),
          ("Response", "response onset", -300, 50)]
BIN_MS = 25

HYPOTHESES = {
    "serial": ("Serial",
               "sensory code is handed off — stimulus decays before choice rises"),
    "parallel": ("Parallel",
                 "stimulus is maintained through the response; choice is already biased by "
                 "the prior before the stimulus arrives"),
}


def _logistic(x, x0, k):
    return 1.0 / (1.0 + np.exp(-(x - x0) / k))


def build_bins():
    """Concatenated bins across epochs, with each bin's within-epoch time in ms.

    Bin counts follow real epoch durations, so the 900 ms cue epoch occupies its true
    share of the axis rather than being squashed to match the 250 ms visual epoch.
    """
    epoch_of, t_local, edges = [], [], [0]
    for name, _event, t0, t1 in EPOCHS:
        centres = np.arange(t0 + BIN_MS / 2, t1, BIN_MS)
        epoch_of += [name] * len(centres)
        t_local.append(centres)
        edges.append(edges[-1] + len(centres))
    return np.array(epoch_of), np.concatenate(t_local), edges


def gain_profiles(hypothesis, epoch_of, t_local):
    """(stimulus, choice) gain profiles in [0,1], defined per epoch in that epoch's own time.

    Each epoch is time-locked to a different event, so a gain profile is only meaningful
    within an epoch; the concatenated axis is pseudo-time, exactly as in the 8.2x figures.
    Times below are ms relative to each epoch's own alignment event.
    """
    stim, choice = np.zeros(len(t_local)), np.zeros(len(t_local))
    vis, cue, resp = (epoch_of == "Visual"), (epoch_of == "Cue"), (epoch_of == "Response")
    t_cue, t_resp = t_local[cue], t_local[resp]

    # Stimulus is at chance before stimulus onset under BOTH hypotheses — the targets carry
    # no coherence information — and rises only after a sensory latency (STIM_LATENCY_MS).
    STIM_LATENCY_MS = 120

    if hypothesis == "serial":
        # rises after latency, peaks, then decays within the cue epoch: handed off
        stim[cue] = _logistic(t_cue, STIM_LATENCY_MS, 28) * (1 - _logistic(t_cue, 450, 60))
        stim[resp] = 0.04                                    # essentially gone by response
        choice[cue] = _logistic(t_cue, 520, 70) * 0.75       # rises only as stimulus fades
        choice[resp] = 0.75 + 0.25 * _logistic(t_resp, -140, 45)
    elif hypothesis == "parallel":
        # stimulus is MAINTAINED: still strongly encoded at the response, no late decay
        stim[cue] = _logistic(t_cue, STIM_LATENCY_MS, 28)
        stim[resp] = 0.90
        # choice is already above chance BEFORE the stimulus — the prior biases the
        # upcoming choice, so it is partially decodable from target onset onward — and is
        # then driven further up once sensory evidence arrives.
        PRIOR_GAIN = 0.30
        choice[vis] = PRIOR_GAIN
        choice[cue] = PRIOR_GAIN + (0.90 - PRIOR_GAIN) * _logistic(t_cue, 240, 90)
        choice[resp] = 0.90 + 0.10 * _logistic(t_resp, -140, 45)
    else:
        raise ValueError(hypothesis)

    return np.clip(stim, 0, 1), np.clip(choice, 0, 1)


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)


def make_figure(hypothesis, epoch_of, t_local, edges):
    G = len(t_local)
    x = np.arange(G)
    stim, choice = gain_profiles(hypothesis, epoch_of, t_local)
    acc_s, acc_c = CHANCE + A * stim, CHANCE + A * choice

    fig, ax = plt.subplots(figsize=(9.6, 5.0))

    for b in edges[1:-1]:                                    # epoch boundaries
        ax.axvline(b - 0.5, color=MUTED, lw=0.9, ls="--", zorder=1)
    ax.axhline(CHANCE, color=MUTED, lw=1.0, ls=":", zorder=1)

    # stimulus onset sits inside the cue epoch (its t = 0), not at the epoch boundary —
    # mark it, since it is the moment coherence information becomes available at all
    cue_start = edges[1]
    onset_x = cue_start + np.argmin(np.abs(t_local[epoch_of == "Cue"])) - 0.5
    ax.axvline(onset_x, color=INK, lw=1.2, zorder=2)
    ax.annotate("stimulus onset", (onset_x, 1.005), xycoords=("data", "axes fraction"),
                ha="center", va="bottom", fontsize=8.5, color=INK)

    ax.plot(x, acc_s, color=C_STIM, lw=2.4, zorder=3)
    ax.plot(x, acc_c, color=C_CHOICE, lw=2.4, zorder=3)

    # Direct labels ride each trace's peak, but a peak in the last bins would push the text
    # off the axes — clamp the anchor inside and let it sit above the curve there.
    lo, hi = int(0.04 * G), int(0.88 * G)
    for acc, colour, name in [(acc_s, C_STIM, "Stimulus"), (acc_c, C_CHOICE, "Choice")]:
        k = min(max(int(np.argmax(acc)), lo), hi)
        ax.annotate(name, (x[k], acc[k]), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=10, color=colour, weight="bold")

    ax.set_xlim(0, G - 1)
    ax.set_ylim(0.46, 1.0)
    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(EPOCHS))]
    ax.set_xticks(centres)
    ax.set_xticklabels([f"{n}\n({ev})" for n, ev, _, _ in EPOCHS])
    ax.set_ylabel("Decoding accuracy", fontsize=10, color=INK)
    _style(ax)
    ax.text(0.004, 0.035, "chance", transform=ax.transAxes, fontsize=8, color=MUTED)

    label, blurb = HYPOTHESES[hypothesis]
    ax.set_title(f"{label} processing — predicted decoding time course",
                 fontsize=13, color=INK, loc="left", pad=34)
    ax.text(0, 1.075, blurb, transform=ax.transAxes, fontsize=9.5, color=MUTED, va="bottom")

    handles = [Line2D([], [], color=C_STIM, lw=2.5, label="Stimulus decoder"),
               Line2D([], [], color=C_CHOICE, lw=2.5, label="Choice decoder")]
    # upper left is empty in both hypotheses (nothing is decodable before stimulus onset)
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9.5,
              bbox_to_anchor=(0.005, 0.99))

    fig.text(0.012, 0.012,
             "Idealized schematic — generated from stated gain profiles, not simulated activity. "
             "Epoch widths are proportional to real durations;\neach epoch is locked to its own "
             "event, so the concatenated axis is pseudo-time.",
             fontsize=7.8, color=MUTED, va="bottom", linespacing=1.5)

    fig.tight_layout(rect=[0, 0.075, 1, 1])
    out = OUT_DIR / f"prediction_{hypothesis}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out, stim, choice


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    epoch_of, t_local, edges = build_bins()
    print(f"bins: {len(t_local)} total  "
          + "  ".join(f"{n}={edges[i+1]-edges[i]}" for i, (n, *_) in enumerate(EPOCHS)))

    for h in HYPOTHESES:
        out, stim, choice = make_figure(h, epoch_of, t_local, edges)
        above = lambda g: np.flatnonzero(g > 0.5)            # noqa: E731
        s_on = above(stim)[0] if len(above(stim)) else None
        c_on = above(choice)[0] if len(above(choice)) else None
        overlap = float(np.mean((stim > 0.5) & (choice > 0.5)))
        lag_ms = (c_on - s_on) * BIN_MS if (s_on is not None and c_on is not None) else None
        print(f"  {h:9s} -> {out.name}   onset lag = {lag_ms} ms   co-active bins = {overlap:.0%}")


if __name__ == "__main__":
    main()
