#!/usr/bin/env python
"""
Prepare per-neuron trial parquet files for Poisson GLM fitting.

Reads compiled spike times and behavioral data, extracts trial-aligned spike
trains, and saves one parquet file per neuron per (prior_cond, outcome_filter)
combination to:

    processed/poisson_glm/data/prior_cond_{cond}_outcome_{filter}/{neuron_id}.parquet

Each row is one trial with columns:
    duration, spike_train, n_bins,
    target_onset, stimulus_onset, stimulus_offset, response_onset,
    coherence (signed, normalized: -0.5 ... 0.5),
    choice, state, reaction_time, trial_idx

Usage:
    python scripts/poisson_glm/prepare_neuronal_data.py
    python scripts/poisson_glm/prepare_neuronal_data.py --prior_cond equal_only --outcome_filter correct_only
    python scripts/poisson_glm/prepare_neuronal_data.py --neuron_id 42
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import dir_config

SESSIONS_TO_EXCLUDE = ["210210_GP_JP", "241209_GP_TZ"]
SAMPLE_RATE_HZ = 30_000   # spike times are in samples at 30 kHz
PRE_TARGET_MS  = 50        # trial starts 50 ms before target onset
BIN_SIZE_MS    = 1.0


# ---------------------------------------------------------------------------
# Spike-data loader
# ---------------------------------------------------------------------------

def _load_spike_data(data_path: Path):
    """Load spike_times and spike_clusters; supports both .npy and .mat."""
    if (data_path / "spike_times.npy").exists():
        return (
            np.load(data_path / "spike_times.npy"),
            np.load(data_path / "spike_clusters.npy"),
        )
    return (
        loadmat(data_path / "spike_times.mat")["spike_times"][0],
        loadmat(data_path / "spike_clusters.mat")["spike_clusters"][0],
    )


def _samples_to_ms(samples):
    """Convert 30 kHz samples to milliseconds (rounded to nearest ms)."""
    return (np.asarray(samples, dtype=float) / (SAMPLE_RATE_HZ / 1000)).round().astype(int)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_neuron_trials(
    neuron_id: int,
    neuron_metadata: pd.DataFrame,
    prior_cond: str,
    outcome_filter: str,
    compiled_dir: Path,
    bin_size: float = BIN_SIZE_MS,
) -> pd.DataFrame:
    """
    Return a DataFrame of trial-aligned spike trains for one neuron.

    Parameters
    ----------
    neuron_id       : int
    neuron_metadata : DataFrame with columns neuron_id, session_id, cluster
    prior_cond      : 'equal_only' | 'unequal_only'
    outcome_filter  : 'correct_only' | 'incorrect_only' | 'all'
    compiled_dir    : Path to compiled session folders
    bin_size        : bin width in ms (default 1.0)

    Returns
    -------
    pd.DataFrame -- one row per valid trial
    """
    meta_row   = neuron_metadata.loc[neuron_metadata["neuron_id"] == neuron_id].iloc[0]
    session    = meta_row["session_id"]
    cluster_id = meta_row["cluster"]
    data_path  = compiled_dir / session

    # Spike times for this neuron (ms)
    spike_times_raw, spike_clusters = _load_spike_data(data_path)
    neuron_spike_ms = _samples_to_ms(spike_times_raw[spike_clusters == cluster_id])

    # Timestamps and trial info
    timestamps_ms = _samples_to_ms(
        pd.read_csv(data_path / f"{session}_timestamps.csv", index_col=None)
    )
    trial_info = pd.read_csv(data_path / f"{session}_trial.csv", index_col=None)

    # GP task trials only
    trials = trial_info[trial_info.task_type == 1].copy()
    trials["signed_coherence"] = trials["coherence"] * (2 * trials["target"] - 1)
    trials = trials[trials.reaction_time.notna()].reset_index(drop=True)

    # Prior condition filter
    if prior_cond == "equal_only":
        trials = trials[trials.prob_toRF == 50].copy()
    elif prior_cond == "unequal_only":
        trials = trials[trials.prob_toRF != 50].copy()
    else:
        raise ValueError(f"Unknown prior_cond: {prior_cond!r}. Use 'equal_only' or 'unequal_only'.")
    trials["state"] = 0

    # Outcome filter
    if outcome_filter == "correct_only":
        trials = trials[trials.outcome == 1].reset_index(drop=True)
    elif outcome_filter == "incorrect_only":
        trials = trials[trials.outcome == 0].reset_index(drop=True)
    elif outcome_filter == "all":
        trials = trials.reset_index(drop=True)
    else:
        raise ValueError(
            f"Unknown outcome_filter: {outcome_filter!r}. Use 'correct_only', 'incorrect_only', or 'all'."
        )

    # Build trial rows
    records = []
    for _, row in trials.iterrows():
        trial_idx = int(row.trial_number) - 1  # 0-based index into timestamps

        target_onset_ms   = timestamps_ms.loc[trial_idx, "target_onset"]
        stimulus_onset_ms = timestamps_ms.loc[trial_idx, "stimulus_onset"]
        response_onset_ms = timestamps_ms.loc[trial_idx, "response_onset"]

        trial_start = target_onset_ms - PRE_TARGET_MS
        duration    = response_onset_ms - trial_start

        if pd.isna(duration) or duration <= 0:
            continue

        # Binned spike train (trial-relative)
        mask = (neuron_spike_ms >= trial_start) & (neuron_spike_ms <= response_onset_ms)
        trial_spikes = neuron_spike_ms[mask] - trial_start

        n_bins      = int(np.ceil(duration / bin_size))
        spike_train = np.zeros(n_bins)
        for t in trial_spikes:
            b = int(np.floor(t / bin_size))
            if 0 <= b < n_bins:
                spike_train[b] += 1

        records.append({
            "duration":        duration,
            "spike_train":     spike_train,
            "n_bins":          n_bins,
            # Event timings relative to trial start
            "target_onset":    PRE_TARGET_MS,
            "stimulus_onset":  stimulus_onset_ms - trial_start,
            "stimulus_offset": response_onset_ms - trial_start,
            "response_onset":  response_onset_ms - trial_start,
            # Experimental variables
            "coherence":       row.signed_coherence / 100,  # signed, normalised: -0.5...0.5
            "choice":          row.choice,
            "state":           row.state,
            "reaction_time":   row.reaction_time,
            "trial_idx":       trial_idx,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare per-neuron parquet files for Poisson GLM."
    )
    parser.add_argument(
        "--prior_cond", type=str, default=None,
        help="One of: equal_only, unequal_only. Default: run all.",
    )
    parser.add_argument(
        "--outcome_filter", type=str, default=None,
        help="One of: correct_only, incorrect_only, all. Default: run all.",
    )
    parser.add_argument(
        "--neuron_id", type=int, default=None,
        help="Process a single neuron ID. Default: all neurons.",
    )
    args = parser.parse_args()

    compiled_dir  = Path(dir_config.data.compiled)
    processed_dir = Path(dir_config.data.processed)

    neuron_metadata = pd.read_csv(processed_dir / "neuron_metadata.csv")
    neuron_metadata = neuron_metadata[
        ~np.isin(neuron_metadata["session_id"], SESSIONS_TO_EXCLUDE)
    ].reset_index(drop=True)

    prior_conds     = [args.prior_cond]     if args.prior_cond     else ["equal_only", "unequal_only"]
    outcome_filters = [args.outcome_filter] if args.outcome_filter else ["correct_only", "all"]
    neuron_ids      = [args.neuron_id]      if args.neuron_id      else neuron_metadata["neuron_id"].tolist()

    for prior_cond in prior_conds:
        for outcome_filter in outcome_filters:
            output_dir = (
                processed_dir / "poisson_glm" / "data"
                / f"prior_cond_{prior_cond}_outcome_{outcome_filter}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[{prior_cond} / {outcome_filter}]  -> {output_dir}")

            for nid in neuron_ids:
                out_path = output_dir / f"{nid}.parquet"
                if out_path.exists():
                    print(f"  neuron {nid:>4}  skipped (already exists)")
                    continue
                try:
                    df = extract_neuron_trials(
                        nid, neuron_metadata, prior_cond, outcome_filter, compiled_dir
                    )
                    df.to_parquet(out_path, index=False)
                    print(f"  neuron {nid:>4}  {len(df):>4} trials -> {out_path.name}")
                except Exception as e:
                    print(f"  neuron {nid:>4}  ERROR: {e}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
