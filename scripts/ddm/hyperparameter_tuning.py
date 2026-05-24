"""
Hyperparameter tuning for LikelihoodCalculator weights.

Strategy: behavioral RMSE on held-out simulation.
  1. Fit the DDM under each (chrono_weight, caf_weight, rt_var_weight) combo.
  2. Score by RMSE between data and post-fit simulation on psychometric and
     chronometric (upper + lower) curves — independent of whichever likelihood
     was used to fit, so the comparison is fair across configs.

Usage (notebook):
    from scripts.ddm.hyperparameter_tuning import run_tuning, plot_tuning_results

    # on real data
    results_df = run_tuning(data, stimulus, save_path=Path("tuning_cache.pkl"))

    # on synthetic data (parameter recovery)
    results_df = run_tuning(save_path=Path("tuning_cache.pkl"))
"""
from __future__ import annotations

import pickle
import time
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.ddm.ddm import FreeParam, FixedParam, DecisionModel

# -----------------------------------------------------------------------
# Defaults shared across helpers
# -----------------------------------------------------------------------

_FIXED_PARAMS = {
    "dt":       FixedParam(0.001),
    "variance": FixedParam(1.0),
}

_FREE_PARAMS = {
    "ndt":           FreeParam(0.1,  1.0),
    "a":             FreeParam(0.8,  6.0),
    "z":             FreeParam(0.1,  0.9),
    "drift_gain":    FreeParam(1.0, 10.0),
    "drift_offset":  FreeParam(-5.0, 5.0),
    "leak_rate":     FreeParam(0.0,  0.3),
    "time_constant": FreeParam(-2.0, 2.0),
}

# "true" parameters used when no real data is provided
_DEFAULT_TRUE_PARAMS = {
    "ndt":           0.25,
    "a":             2.0,
    "z":             0.5,
    "drift_gain":    7.0,
    "drift_offset":  0.0,
    "variance":      1.0,
    "dt":            0.001,
    "leak_rate":     0.1,
    "time_constant": 0.3,
}

_COHERENCES     = np.array([-0.50, -0.18, -0.09, 0.0, 0.09, 0.18, 0.50]).round(2)
_TRIALS_PER_COH = 150
_MAX_DURATION   = 3.0
_DT             = 0.001


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

@dataclass
class TuningConfig:
    chrono_weight: float
    caf_weight:    float
    rt_var_weight: float
    caf_bins:      int = 5
    nbins:         int = 5

    def label(self) -> str:
        return (
            f"ch={self.chrono_weight:.1f}"
            f"_caf={self.caf_weight:.1f}"
            f"_var={self.rt_var_weight:.1f}"
        )


def build_param_grid(
    chrono_weights: list[float] = (0.5, 1.0, 2.0, 4.0),
    caf_weights:    list[float] = (0.5, 1.0, 2.0),
    rt_var_weights: list[float] = (0.0, 0.5, 1.0),
    caf_bins:       int = 5,
    nbins:          int = 5,
) -> list[TuningConfig]:
    return [
        TuningConfig(cw, aw, vw, caf_bins=caf_bins, nbins=nbins)
        for cw, aw, vw in product(chrono_weights, caf_weights, rt_var_weights)
    ]


# -----------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------

def make_stimulus(
    coherences=_COHERENCES,
    trials_per_coh=_TRIALS_PER_COH,
    max_duration=_MAX_DURATION,
    dt=_DT,
) -> np.ndarray:
    n_tp = int(max_duration / dt)
    coh  = np.repeat(coherences, trials_per_coh)
    return np.tile(coh.reshape(-1, 1), (1, n_tp)).astype(np.float32)


def _make_ddm_model(config: TuningConfig) -> DecisionModel:
    """Instantiate DDMModel with the given likelihood config."""
    from scripts.ddm.ddm_fitting import DDMModel
    return DDMModel(
        fixed_params=_FIXED_PARAMS,
        free_params=_FREE_PARAMS,
        likelihood_params={
            "chrono_weight": config.chrono_weight,
            "caf_weight":    config.caf_weight,
            "caf_bins":      config.caf_bins,
            "rt_var_weight": config.rt_var_weight,
            "nbins":         config.nbins,
        },
    )


def generate_synthetic_data(
    true_params: dict = _DEFAULT_TRUE_PARAMS,
    coherences=_COHERENCES,
    trials_per_coh=_TRIALS_PER_COH,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Simulate data from known parameters for parameter recovery tests."""
    stimulus  = make_stimulus(coherences=coherences, trials_per_coh=trials_per_coh)
    tmp_model = _make_ddm_model(TuningConfig(chrono_weight=2.0, caf_weight=1.0, rt_var_weight=0.0))
    data      = tmp_model.simulate(stimulus=stimulus, params=true_params, n_reps=3, seed=seed)
    return pd.DataFrame(data), stimulus


# -----------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------

def behavioral_summary(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-coherence psychometric and choice-conditional mean RT."""
    df   = df.copy()
    df["signed_coherence"] = df["signed_coherence"].round(2)
    cohs = sorted(df["signed_coherence"].unique())

    p_upper, mrt_upper, mrt_lower = [], [], []
    for c in cohs:
        sub = df[df["signed_coherence"] == c].dropna(subset=["rt", "choice"])
        p_upper.append((sub["choice"] == 1).mean() if len(sub) else np.nan)
        u = sub[sub["choice"] == 1]["rt"]
        l = sub[sub["choice"] == 0]["rt"]
        mrt_upper.append(u.mean() if len(u) else np.nan)
        mrt_lower.append(l.mean() if len(l) else np.nan)

    return {
        "cohs":      np.array(cohs),
        "p_upper":   np.array(p_upper),
        "mrt_upper": np.array(mrt_upper),
        "mrt_lower": np.array(mrt_lower),
    }


def behavioral_rmse(data_sum: dict, sim_sum: dict) -> dict[str, float]:
    def _rmse(a, b):
        d    = a - b
        mask = np.isfinite(d)
        return float(np.sqrt(np.nanmean(d[mask] ** 2))) if mask.any() else np.nan

    return {
        "psychometric":       _rmse(data_sum["p_upper"],   sim_sum["p_upper"]),
        "chrono_upper":       _rmse(data_sum["mrt_upper"], sim_sum["mrt_upper"]),
        "chrono_lower":       _rmse(data_sum["mrt_lower"], sim_sum["mrt_lower"]),
    }


# -----------------------------------------------------------------------
# Fit + score one config
# -----------------------------------------------------------------------

def fit_and_score(
    data:        pd.DataFrame,
    stimulus:    np.ndarray,
    config:      TuningConfig,
    n_reps:      int = 5,
    max_iter:    int = 300,
    eval_reps:   int = 10,
    seed:        int = 0,
) -> dict:
    """
    Fit the model under `config`, simulate with the recovered params,
    and return behavioral RMSE vs data.

    n_reps / max_iter can be reduced for speed during grid search.
    eval_reps controls the simulation used for scoring (more = stable).
    """
    model  = _make_ddm_model(config)
    t0     = time.perf_counter()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(
            data={
                "rt":               data["rt"],
                "choice":           data["choice"],
                "signed_coherence": data["signed_coherence"],
            },
            stimulus=stimulus,
            n_reps=n_reps,
            max_iterations=max_iter,
            l1_weight=0.0,
            verbose=False,
        )

    elapsed = time.perf_counter() - t0

    sim      = model.simulate(stimulus=stimulus, params=result["parameters"], n_reps=eval_reps, seed=seed)
    sim_df   = pd.DataFrame(sim)

    data_sum = behavioral_summary(data)
    sim_sum  = behavioral_summary(sim_df)
    rmse     = behavioral_rmse(data_sum, sim_sum)

    composite = float(np.nanmean(list(rmse.values())))

    return {
        "config":     config,
        "label":      config.label(),
        **rmse,
        "composite":  composite,
        "nll":        result["likelihood"],
        "parameters": result["parameters"],
        "elapsed_s":  elapsed,
    }


# -----------------------------------------------------------------------
# Grid search
# -----------------------------------------------------------------------

def grid_search(
    data:       pd.DataFrame,
    stimulus:   np.ndarray,
    configs:    list[TuningConfig],
    n_reps:     int  = 5,
    max_iter:   int  = 300,
    save_path:  Optional[Path] = None,
    resume:     bool = True,
) -> pd.DataFrame:
    """
    Run all configs sequentially and return results sorted by composite RMSE.

    Checkpoints after every config if save_path is given, so the search can
    be interrupted and resumed without losing progress.
    """
    raw_results: list[dict] = []

    completed = set()
    if resume and save_path and save_path.exists():
        with open(save_path, "rb") as f:
            raw_results = pickle.load(f)
        completed = {r["label"] for r in raw_results}
        print(f"Resuming: {len(completed)}/{len(configs)} configs already done.")

    for i, config in enumerate(configs):
        if config.label() in completed:
            continue

        print(f"[{i+1:3d}/{len(configs)}] {config.label()} ...", end=" ", flush=True)

        try:
            row = fit_and_score(data, stimulus, config, n_reps=n_reps, max_iter=max_iter)
            print(
                f"composite={row['composite']:.4f}  "
                f"(psychometric={row['psychometric']:.4f}, "
                f"chrono_upper={row['chrono_upper']:.4f}, "
                f"chrono_lower={row['chrono_lower']:.4f})  "
                f"nll={row['nll']:.1f}  "
                f"{row['elapsed_s']:.0f}s"
            )
        except Exception as e:
            print(f"FAILED: {e}")
            row = {
                "config":     config,
                "label":      config.label(),
                "psychometric": np.nan,
                "chrono_upper": np.nan,
                "chrono_lower": np.nan,
                "composite":  np.nan,
                "nll":        np.nan,
                "parameters": None,
                "elapsed_s":  np.nan,
            }

        raw_results.append(row)

        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(raw_results, f)

    rows = []
    for r in raw_results:
        rows.append({
            "label":          r["label"],
            "chrono_weight":  r["config"].chrono_weight,
            "caf_weight":     r["config"].caf_weight,
            "rt_var_weight":  r["config"].rt_var_weight,
            "composite":      r["composite"],
            "psychometric":   r["psychometric"],
            "chrono_upper":   r["chrono_upper"],
            "chrono_lower":   r["chrono_lower"],
            "nll":            r["nll"],
            "elapsed_s":      r["elapsed_s"],
        })

    return pd.DataFrame(rows).sort_values("composite").reset_index(drop=True)


# -----------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------

def plot_tuning_results(df: pd.DataFrame) -> None:
    """
    One row per metric, one column per hyperparameter pair.
    Each cell is a heatmap averaged over the third hyperparameter.
    Lower (darker) is better.
    """
    metrics = ["composite", "psychometric", "chrono_upper", "chrono_lower"]
    pairs = [
        ("chrono_weight", "caf_weight",   "rt_var_weight"),
        ("chrono_weight", "rt_var_weight","caf_weight"),
        ("caf_weight",    "rt_var_weight","chrono_weight"),
    ]

    valid_metrics = [m for m in metrics if m in df.columns and df[m].notna().any()]
    fig, axes = plt.subplots(
        len(valid_metrics), len(pairs),
        figsize=(5 * len(pairs), 4 * len(valid_metrics)),
        squeeze=False,
    )

    for row_i, metric in enumerate(valid_metrics):
        for col_i, (x, y, _) in enumerate(pairs):
            ax = axes[row_i, col_i]
            pivot = (
                df.groupby([x, y])[metric]
                .mean()
                .reset_index()
                .pivot(index=y, columns=x, values=metric)
            )
            im = ax.imshow(pivot.values, aspect="auto", cmap="viridis_r")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns], fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{v:.1f}" for v in pivot.index], fontsize=8)
            ax.set_xlabel(x, fontsize=9)
            ax.set_ylabel(y, fontsize=9)
            ax.set_title(metric.replace("_", " "), fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Behavioral RMSE by hyperparameter combo (lower = better)", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_best_fit(
    data:     pd.DataFrame,
    stimulus: np.ndarray,
    best_row: pd.Series,
    n_reps:   int = 15,
) -> None:
    """Simulate with the best config's recovered params and plot vs data."""
    config = TuningConfig(
        chrono_weight=best_row["chrono_weight"],
        caf_weight=best_row["caf_weight"],
        rt_var_weight=best_row["rt_var_weight"],
    )
    model = _make_ddm_model(config)

    # re-fit to get parameters (or load from raw_results if cached)
    result = model.fit(
        data={"rt": data["rt"], "choice": data["choice"], "signed_coherence": data["signed_coherence"]},
        stimulus=stimulus,
        n_reps=n_reps,
        max_iterations=500,
        l1_weight=0.0,
        verbose=True,
    )

    sim    = pd.DataFrame(model.simulate(stimulus=stimulus, params=result["parameters"], n_reps=n_reps))
    d_sum  = behavioral_summary(data)
    s_sum  = behavioral_summary(sim)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Best config: {best_row['label']}")

    ax1.plot(s_sum["cohs"], s_sum["p_upper"], color="steelblue", label="model")
    ax1.scatter(d_sum["cohs"], d_sum["p_upper"], color="steelblue", zorder=5, label="data")
    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax1.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set(xlabel="Coherence", ylabel="P(upper boundary)", title="Psychometric", ylim=(0, 1))
    ax1.legend()

    ax2.plot(s_sum["cohs"], s_sum["mrt_upper"], color="steelblue", label="model upper")
    ax2.plot(s_sum["cohs"], s_sum["mrt_lower"], color="tomato",    label="model lower")
    ax2.scatter(d_sum["cohs"], d_sum["mrt_upper"], color="steelblue", zorder=5, label="data upper")
    ax2.scatter(d_sum["cohs"], d_sum["mrt_lower"], color="tomato",    zorder=5, label="data lower")
    ax2.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set(xlabel="Coherence", ylabel="Mean RT (s)", title="Chronometric")
    ax2.legend()

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def run_tuning(
    data:           Optional[pd.DataFrame] = None,
    stimulus:       Optional[np.ndarray]   = None,
    true_params:    Optional[dict]         = None,
    chrono_weights: list[float] = (0.5, 1.0, 2.0, 4.0),
    caf_weights:    list[float] = (0.5, 1.0, 2.0),
    rt_var_weights: list[float] = (0.0, 0.5, 1.0),
    n_reps:         int  = 5,
    max_iter:       int  = 300,
    save_path:      Optional[Path] = None,
    resume:         bool = True,
) -> pd.DataFrame:
    """
    Main entry point.

    If data/stimulus are None, generates synthetic data from true_params
    (useful for parameter recovery tests). Otherwise uses the provided data.

    Parameters
    ----------
    n_reps   : simulation reps per objective call — 5 is enough for tuning,
               use 15 for the final fit after picking the best config.
    max_iter : DE iterations per config — 300 distinguishes good from bad
               configs without a full convergence run.
    save_path: path to a .pkl file for checkpointing (allows resume after crash).
    """
    if data is None or stimulus is None:
        print("Generating synthetic data from true_params for parameter recovery...")
        tp       = true_params or _DEFAULT_TRUE_PARAMS
        data, stimulus = generate_synthetic_data(true_params=tp)
        print(f"Synthetic data: {len(data)} trials, "
              f"{data['signed_coherence'].nunique()} coherences\n")

    configs = build_param_grid(
        chrono_weights=list(chrono_weights),
        caf_weights=list(caf_weights),
        rt_var_weights=list(rt_var_weights),
    )

    secs_per_config = max_iter * 0.3          # rough estimate
    total_min       = len(configs) * secs_per_config / 60
    print(
        f"Grid: {len(configs)} configs × {max_iter} DE iters "
        f"(~{total_min:.0f}–{total_min*2:.0f} min total)\n"
    )

    results_df = grid_search(
        data, stimulus, configs,
        n_reps=n_reps,
        max_iter=max_iter,
        save_path=save_path,
        resume=resume,
    )

    print("\n=== Top 5 configs ===")
    cols = ["label", "composite", "psychometric", "chrono_upper", "chrono_lower", "nll"]
    print(results_df[cols].head().to_string(index=False))

    plot_tuning_results(results_df)

    return results_df
