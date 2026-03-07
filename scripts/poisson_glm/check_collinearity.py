#!/usr/bin/env python
"""
Compute collinearity diagnostics (condition number + per-block VIF) for any
Poisson GLM model variant.

Usage:
    python scripts/poisson_glm/check_collinearity.py --model_file 1stim_1coh_2choice_1500ms
    python scripts/poisson_glm/check_collinearity.py --model_file 7stim_7coh_2choice_1500ms --prior_cond equal_only --outcome_filter correct_only
    python scripts/poisson_glm/check_collinearity.py --model_file 1stim_1coh_2choice_1500ms --n_neurons 10 --plot

Available model files (scripts/poisson_glm/models/):
    0stim_2choice_1500ms
    1stim_1coh_0choice
    1stim_1coh_2choice_1500ms
    1stim_7coh_0choice
    1stim_7coh_2choice_1500ms
    7stim_7coh_0choice
    7stim_7coh_2choice_1500ms
"""

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import variation

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import dir_config

# ---------------------------------------------------------------------------
# Helpers shared with fit_neuron_cv.py
# ---------------------------------------------------------------------------

def _load_model_module(model_name: str):
    model_file = Path(__file__).parent / "models" / f"{model_name}.py"
    if not model_file.exists():
        print(f"Model file not found: {model_file}", file=sys.stderr)
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("model_module", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Collinearity diagnostics
# ---------------------------------------------------------------------------

def condition_number(C: np.ndarray) -> float:
    """Condition number of a pre-computed correlation matrix."""
    return float(np.linalg.cond(C))


def _corr_matrix(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the correlation matrix for X (constant columns stripped,
    rows subsampled to max 50k for speed).

    Returns (C, surviving_col_indices) where surviving_col_indices maps
    positions in C back to the original column indices of X.
    """
    if X.shape[0] > 50_000:
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(X.shape[0], size=50_000, replace=False)
        Xsub = X[idx]
    else:
        Xsub = X
    std = Xsub.std(axis=0)
    surviving = np.where(std > 1e-10)[0]
    Xs = Xsub[:, surviving]
    std_s = std[surviving]
    Xs = (Xs - Xs.mean(axis=0)) / std_s
    C = (Xs.T @ Xs) / Xs.shape[0]
    return C, surviving


def _standardize_X(X: np.ndarray, max_rows: int = 50_000) -> np.ndarray:
    """
    Drop constant columns, standardize, and optionally subsample rows.
    Subsampling speeds up the Gram matrix computation on large matrices while
    preserving the correlation structure (correlations are scale-free).
    """
    if X.shape[0] > max_rows:
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(X.shape[0], size=max_rows, replace=False)
        X = X[idx]
    std = X.std(axis=0)
    # drop constant columns (intercept or all-zero blocks)
    X = X[:, std > 1e-10]
    std = std[std > 1e-10]
    return (X - X.mean(axis=0)) / std


def block_vif(C: np.ndarray, surviving: np.ndarray, feature_idx: dict) -> dict:
    """
    Compute mean VIF for each feature block.
    VIF_j = (C⁻¹)_jj  for standardized predictors.

    Parameters
    ----------
    C         : correlation matrix built from surviving columns only
    surviving : original column indices that survive (from _corr_matrix)
    """
    ridge = 1e-6 * np.eye(C.shape[0])
    try:
        C_inv_diag = np.diag(np.linalg.inv(C + ridge))
    except np.linalg.LinAlgError:
        C_inv_diag = np.full(C.shape[0], np.inf)

    # Build a mapping: original_col_idx → position in C_inv_diag
    col_to_pos = {orig: pos for pos, orig in enumerate(surviving)}

    blocks = {
        'target':   (feature_idx['target_start'],   feature_idx['target_end']),
        'stimulus': (feature_idx['stim_start'],      feature_idx['stim_end']),
        'saccade':  (feature_idx['saccade_start'],   feature_idx['saccade_end']),
        'history':  (feature_idx['history_start'],   feature_idx['history_end']),
    }

    vif_results = {}
    for name, (start, end) in blocks.items():
        if start == end:
            continue
        positions = [col_to_pos[c] for c in range(start, end) if c in col_to_pos]
        if not positions:
            continue
        vif_results[name] = float(np.mean(C_inv_diag[positions]))

    return vif_results


def cross_block_correlation(X: np.ndarray, feature_idx: dict) -> float:
    """
    Mean absolute correlation between every stimulus column and every saccade
    column.  Operates on the raw (unstandardized) X so we can slice columns;
    internally standardizes before computing correlations.
    """
    s_start, s_end = feature_idx['stim_start'],    feature_idx['stim_end']
    c_start, c_end = feature_idx['saccade_start'], feature_idx['saccade_end']
    if s_start == s_end or c_start == c_end:
        return np.nan

    # subsample rows for speed
    if X.shape[0] > 50_000:
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(X.shape[0], size=50_000, replace=False)
        X = X[idx]

    def _zscore(M):
        std = M.std(axis=0)
        std[std < 1e-10] = 1.0
        return (M - M.mean(axis=0)) / std

    S = _zscore(X[:, s_start:s_end])
    C = _zscore(X[:, c_start:c_end])
    corr_matrix = (S.T @ C) / S.shape[0]
    return float(np.mean(np.abs(corr_matrix)))


def rt_stats(df: pd.DataFrame) -> dict:
    """Return RT distribution statistics for a neuron's trial data."""
    rt = df['response_onset'] - df['stimulus_onset']
    return {
        'mean_ms':   float(rt.mean()),
        'std_ms':    float(rt.std()),
        'cv':        float(rt.std() / rt.mean()),   # coefficient of variation
        'p5_ms':     float(rt.quantile(0.05)),
        'p95_ms':    float(rt.quantile(0.95)),
    }


def build_constant_rt_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df where every trial's response_onset is replaced by
    stimulus_onset + median(RT).  This collapses all RT variance and gives
    the theoretical worst-case collinearity (purely structural).
    """
    median_rt = int(round((df['response_onset'] - df['stimulus_onset']).median()))
    df_const = df.copy()
    df_const['response_onset'] = df_const['stimulus_onset'] + median_rt
    # ensure response_onset doesn't exceed trial duration
    df_const['response_onset'] = df_const[['response_onset', 'duration']].min(axis=1)
    return df_const


def interpret_vif(v: float) -> str:
    if v < 5:    return "low"
    if v < 10:   return "moderate"
    if v < 100:  return "high"
    return "severe"


def interpret_cond(c: float) -> str:
    if c < 100:    return "fine"
    if c < 1000:   return "mild"
    if c < 10000:  return "moderate"
    return "severe"


def print_report(model_name: str, neuron_id: int, cond: float, vifs: dict,
                 cross_corr: float = None, rt: dict = None,
                 cond_const: float = None, vifs_const: dict = None):
    print(f"\n{'─'*60}")
    print(f"  Model : {model_name}   Neuron : {neuron_id}")
    print(f"{'─'*60}")

    if rt is not None:
        print(f"  RT distribution:  mean={rt['mean_ms']:.0f}ms  std={rt['std_ms']:.0f}ms  "
              f"CV={rt['cv']:.2f}  [p5={rt['p5_ms']:.0f}  p95={rt['p95_ms']:.0f}]")
        print(f"    CV interpretation: {'good separation leverage' if rt['cv'] > 0.3 else 'narrow RT — high structural collinearity risk'}")

    print(f"\n  ── ACTUAL (real RT variability) ──")
    print(f"  Condition number : {cond:.2e}  [{interpret_cond(cond)}]")
    if cross_corr is not None and not np.isnan(cross_corr):
        print(f"  Stim↔Saccade cross-corr : {cross_corr:.3f}  "
              f"[{'low' if cross_corr < 0.3 else 'moderate' if cross_corr < 0.6 else 'HIGH — collinear'}]")
    print(f"  Block VIFs:")
    for block, v in vifs.items():
        print(f"    {block:<12} {v:>8.1f}  [{interpret_vif(v)}]")

    if cond_const is not None:
        print(f"\n  ── STRUCTURAL ONLY (constant RT = median) ──")
        print(f"  Condition number : {cond_const:.2e}  [{interpret_cond(cond_const)}]")
        print(f"  Block VIFs:")
        for block, v in vifs_const.items():
            print(f"    {block:<12} {v:>8.1f}  [{interpret_vif(v)}]")
        if 'stimulus' in vifs and 'stimulus' in vifs_const:
            ratio = vifs_const['stimulus'] / max(vifs['stimulus'], 1.0)
            print(f"\n  RT-variance benefit (VIF ratio const/actual): {ratio:.1f}×")
            print(f"    → RT variability reduces stim collinearity by {(1 - 1/ratio)*100:.0f}%" if ratio > 1 else "")
    print()


# ---------------------------------------------------------------------------
# Optional summary plot
# ---------------------------------------------------------------------------

def plot_vif_summary(records: list, model_name: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    blocks = [r for r in records[0]['vif'].keys()]
    x = np.arange(len(records))
    has_const = 'vif_const' in records[0] and records[0]['vif_const'] is not None

    ncols = 3 if has_const else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    fig.suptitle(f"Collinearity diagnostics — {model_name}", fontsize=12)

    import matplotlib.cm as cm
    colors = cm.tab10(np.linspace(0, 1, len(blocks)))

    # Condition number
    conds = [r['cond'] for r in records]
    axes[0].semilogy(x, conds, 'k-o', markersize=4, label='actual')
    if has_const:
        conds_c = [r['cond_const'] for r in records]
        axes[0].semilogy(x, conds_c, 'k--s', markersize=4, alpha=0.5, label='constant RT')
    for thresh, label, color in [(100, 'mild', 'gold'), (1000, 'moderate', 'orange'), (10000, 'severe', 'red')]:
        axes[0].axhline(thresh, linestyle='--', color=color, alpha=0.6, label=label)
    axes[0].set_xlabel('Neuron (sorted by ID)')
    axes[0].set_ylabel('Condition number (log scale)')
    axes[0].set_title('Gram matrix condition number')
    axes[0].legend(fontsize=8)

    # VIF per block — actual
    for i, block in enumerate(blocks):
        vifs_vals = [r['vif'].get(block, np.nan) for r in records]
        axes[1].plot(x, vifs_vals, '-o', markersize=4, label=block, color=colors[i])
    for thresh, label, color in [(5, 'moderate', 'gold'), (10, 'high', 'orange'), (100, 'severe', 'red')]:
        axes[1].axhline(thresh, linestyle='--', color=color, alpha=0.5)
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Neuron (sorted by ID)')
    axes[1].set_ylabel('Mean block VIF (log scale)')
    axes[1].set_title('Per-block VIF — actual RT')
    axes[1].legend(fontsize=8)

    # VIF comparison actual vs constant RT (stim & saccade only)
    if has_const:
        for i, block in enumerate(['stimulus', 'saccade']):
            if block not in records[0]['vif']:
                continue
            actual_v = [r['vif'].get(block, np.nan) for r in records]
            const_v  = [r['vif_const'].get(block, np.nan) for r in records]
            axes[2].plot(x, actual_v, '-o',  markersize=4, label=f'{block} actual',   color=colors[i])
            axes[2].plot(x, const_v,  '--s', markersize=4, label=f'{block} const RT', color=colors[i], alpha=0.5)
        for thresh, label, color in [(5, 'moderate', 'gold'), (10, 'high', 'orange'), (100, 'severe', 'red')]:
            axes[2].axhline(thresh, linestyle='--', color=color, alpha=0.5)
        axes[2].set_yscale('log')
        axes[2].set_xlabel('Neuron (sorted by ID)')
        axes[2].set_ylabel('Mean block VIF (log scale)')
        axes[2].set_title('Stim vs Saccade VIF: actual vs structural')
        axes[2].legend(fontsize=8)

    plt.tight_layout()
    out_path = PROJECT_ROOT / f'scripts/poisson_glm/collinearity_{model_name}.png'
    plt.savefig(out_path, dpi=120)
    print(f"Plot saved to {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check design-matrix collinearity for a Poisson GLM model.')
    parser.add_argument('--model_file',      type=str, required=True, help='Model name (stem of file in models/)')
    parser.add_argument('--prior_cond',      type=str, default='equal_only',    help='Prior condition (default: equal_only)')
    parser.add_argument('--outcome_filter',  type=str, default='correct_only',  help='Outcome filter (default: correct_only)')
    parser.add_argument('--n_neurons',       type=int, default=5,               help='Number of neurons to sample (default: 5)')
    parser.add_argument('--neuron_ids',      type=int, nargs='+',               help='Specific neuron IDs to check (overrides --n_neurons)')
    parser.add_argument('--plot',            action='store_true',               help='Save and show a summary plot')
    parser.add_argument('--no_const_rt',     action='store_true',               help='Skip the constant-RT simulation (faster)')
    args = parser.parse_args()

    # --- load model module ---
    module = _load_model_module(args.model_file)
    config = module.StateBasedPoissonGLMConfig()
    feature_idx = module.get_feature_idx(config)
    build_design_matrix = module.build_design_matrix

    # --- locate data ---
    processed_dir = Path(dir_config.data.processed)
    data_dir = processed_dir / 'poisson_glm' / 'data' / f'prior_cond_{args.prior_cond}_outcome_{args.outcome_filter}'
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        print(f"Available:\n  " + "\n  ".join(str(p.name) for p in (processed_dir / 'poisson_glm' / 'data').iterdir()))
        sys.exit(1)

    all_parquets = sorted(data_dir.glob('*.parquet'), key=lambda p: int(p.stem))

    if args.neuron_ids:
        selected = [data_dir / f'{nid}.parquet' for nid in args.neuron_ids]
        missing = [p for p in selected if not p.exists()]
        if missing:
            print(f"Missing data files: {missing}", file=sys.stderr)
            sys.exit(1)
    else:
        rng = np.random.default_rng(seed=42)
        selected = rng.choice(all_parquets, size=min(args.n_neurons, len(all_parquets)), replace=False).tolist()
        selected = sorted(selected, key=lambda p: int(p.stem))

    print(f"\nModel       : {args.model_file}")
    print(f"Data dir    : {data_dir.name}")
    print(f"Checking {len(selected)} neuron(s): {[int(p.stem) for p in selected]}")

    run_const_rt = not args.no_const_rt

    records = []
    for fpath in selected:
        nid = int(fpath.stem)
        df = pd.read_parquet(fpath)
        coh_levels = np.sort(df['coherence'].unique() / 100)

        # --- actual RT ---
        X, _ = build_design_matrix(df, coh_levels, feature_idx)
        X_dense = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
        C, surviving  = _corr_matrix(X_dense)
        cond  = condition_number(C)
        vifs  = block_vif(C, surviving, feature_idx)
        xcorr = cross_block_correlation(X_dense, feature_idx)
        rts   = rt_stats(df)

        # --- constant RT (structural collinearity only) ---
        cond_const, vifs_const = None, None
        if run_const_rt:
            df_const = build_constant_rt_df(df)
            X_const, _ = build_design_matrix(df_const, coh_levels, feature_idx)
            X_const_dense = X_const.toarray() if hasattr(X_const, 'toarray') else np.asarray(X_const)
            C_const, surviving_const = _corr_matrix(X_const_dense)
            cond_const = condition_number(C_const)
            vifs_const = block_vif(C_const, surviving_const, feature_idx)

        print_report(args.model_file, nid, cond, vifs,
                     cross_corr=xcorr, rt=rts,
                     cond_const=cond_const, vifs_const=vifs_const)
        records.append({'neuron_id': nid, 'cond': cond, 'vif': vifs,
                        'cross_corr': xcorr, 'rt': rts,
                        'cond_const': cond_const, 'vif_const': vifs_const})

    # --- aggregate summary ---
    print(f"{'═'*60}")
    print(f"  SUMMARY across {len(records)} neurons")
    print(f"{'═'*60}")

    rt_cvs = [r['rt']['cv'] for r in records]
    print(f"  RT CV  median={np.median(rt_cvs):.2f}  "
          f"({'good' if np.median(rt_cvs) > 0.3 else 'narrow — collinearity risk'})")

    cross_corrs = [r['cross_corr'] for r in records if not np.isnan(r['cross_corr'])]
    if cross_corrs:
        print(f"  Stim↔Saccade cross-corr  median={np.median(cross_corrs):.3f}")

    print(f"\n  ── Actual RT ──")
    conds = [r['cond'] for r in records]
    print(f"  Condition number  median={np.median(conds):.2e}  [{interpret_cond(np.median(conds))}]")
    all_blocks = list(records[0]['vif'].keys())
    for block in all_blocks:
        vals = [r['vif'][block] for r in records if block in r['vif']]
        med = float(np.median(vals))
        print(f"  VIF {block:<12} median={med:>8.1f}  [{interpret_vif(med)}]")

    if run_const_rt:
        print(f"\n  ── Structural only (constant RT) ──")
        conds_c = [r['cond_const'] for r in records]
        print(f"  Condition number  median={np.median(conds_c):.2e}  [{interpret_cond(np.median(conds_c))}]")
        for block in all_blocks:
            vals = [r['vif_const'][block] for r in records if r['vif_const'] and block in r['vif_const']]
            if not vals:
                continue
            med = float(np.median(vals))
            print(f"  VIF {block:<12} median={med:>8.1f}  [{interpret_vif(med)}]")

        # RT benefit
        for block in ['stimulus', 'saccade']:
            act = [r['vif'][block] for r in records if block in r['vif']]
            con = [r['vif_const'][block] for r in records if r['vif_const'] and block in r['vif_const']]
            if act and con:
                ratio = np.median(con) / max(np.median(act), 1.0)
                print(f"\n  RT-variance benefit on '{block}': {ratio:.1f}× VIF reduction from RT spread")
    print()

    if args.plot:
        plot_vif_summary(records, args.model_file)
