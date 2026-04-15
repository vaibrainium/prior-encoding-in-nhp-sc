"""
validate_refactor.py

Checks that the refactored ddm_model.py / ddm_base.py produce equivalent
results to the original full_ddm_vary_all_params.py /
full_ddm_vary_three_params.py, and benchmarks both.

Tests
-----
1. LikelihoodCalculator       -- exact match on deterministic inputs
2. CUDA/Torch simulator       -- exact match with same seed (sequential time loop,
                                 all-trial parallelism is deterministic given same seed)
3. CPU (Numba) simulator      -- statistical equivalence only (prange reorders RNG draws)
4. AllParamsModel objective   -- close match with same seed
5. ThreeParamsModel objective -- close match with same seed
6. Speed                      -- wall-clock comparison for simulation and objective

Usage
-----
    python validate_refactor.py
    python validate_refactor.py --device cpu   # skip CUDA tests
    python validate_refactor.py --n_trials 500 --n_reps 10
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def report(label: str, passed: bool, detail: str = "") -> None:
    status = PASS if passed else FAIL
    line = f"  [{status}] {label}"
    if detail:
        line += f"  ({detail})"
    print(line)


def allclose(a: np.ndarray, b: np.ndarray, atol: float = 1e-5) -> bool:
    return bool(np.allclose(a, b, atol=atol, equal_nan=True))


def make_stimulus(n_trials: int, coherences: np.ndarray, dt: float = 0.001,
                  max_duration: float = 3.0) -> np.ndarray:
    """Constant-coherence stimulus: each trial's coherence tiled over time."""
    n_timepoints = int(max_duration / dt)
    coh_per_trial = np.tile(coherences, int(np.ceil(n_trials / len(coherences))))[:n_trials]
    return np.tile(coh_per_trial.reshape(-1, 1), (1, n_timepoints)).astype(np.float32)


def make_synthetic_data(
    n_trials: int,
    coherences: np.ndarray,
    dual_prior: bool,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    coh = np.tile(coherences, int(np.ceil(n_trials / len(coherences))))[:n_trials]
    data: dict[str, np.ndarray] = {
        "signed_coherence": coh,
        "choice":           rng.integers(0, 2, n_trials).astype(float),
        "rt":               rng.uniform(0.2, 1.5, n_trials),
    }
    if dual_prior:
        blocks = np.array(["equal"] * (n_trials // 2) + ["unequal"] * (n_trials - n_trials // 2))
        data["prior_block"] = blocks
    return data


# ---------------------------------------------------------------------------
# Individual test functions
# ---------------------------------------------------------------------------

def test_likelihood_exact(OldLC, NewLC) -> None:
    """LikelihoodCalculator should give bit-for-bit identical output."""
    print("\n── LikelihoodCalculator (exact) ──")
    rng = np.random.default_rng(0)
    n = 400
    cohs = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
    coh_arr = np.tile(cohs, n // len(cohs) + 1)[:n]

    rt_pred    = rng.uniform(0.2, 1.5, n).astype(float)
    ch_pred    = rng.integers(0, 2, n).astype(float)
    rt_data    = rng.uniform(0.2, 1.5, n).astype(float)
    ch_data    = rng.integers(0, 2, n).astype(float)

    for nbins, rt_w in [(5, 1.0), (3, 2.0), (8, 0.5)]:
        old_lc = OldLC(nbins=nbins, rt_weight=rt_w)
        new_lc = NewLC(nbins=nbins, rt_weight=rt_w)

        old_val = old_lc.calculate_likelihood(rt_pred, ch_pred, rt_data, ch_data, coh_arr, coh_arr)
        new_val = new_lc.calculate_likelihood(rt_pred, ch_pred, rt_data, ch_data, coh_arr, coh_arr)

        passed = abs(old_val - new_val) < 1e-9
        report(f"nbins={nbins}, rt_weight={rt_w}", passed,
               f"old={old_val:.6f}  new={new_val:.6f}  diff={abs(old_val-new_val):.2e}")


def test_torch_simulator_exact(OldCUDA, NewCUDA, device: str, n_trials: int) -> None:
    """Torch simulator: same seed → identical rt and choice arrays."""
    print(f"\n── Torch simulator, device={device} (exact) ──")
    cohs = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
    stim = make_stimulus(n_trials, cohs)

    param_sets = [
        dict(drift_gain=7.0, drift_offset=0.0, a=2.0, z=0.5,
             ndt=0.1, variance=1.0, leak_rate=0.01, time_constant=0.5),
        dict(drift_gain=5.0, drift_offset=0.3, a=1.5, z=0.4,
             ndt=0.15, variance=0.8, leak_rate=0.05, time_constant=1.0),
    ]

    for i, params in enumerate(param_sets):
        old_sim = OldCUDA(leak=True, time_dependence=True, device=device)
        new_sim = NewCUDA(leak=True, time_dependence=True, device=device)

        for sim in (old_sim, new_sim):
            for k, v in params.items():
                setattr(sim, k, torch.tensor(v, device=sim.device, dtype=torch.float32))

        torch.manual_seed(42)
        rt_old, ch_old, _ = old_sim.simulate_trials(stim)

        torch.manual_seed(42)
        rt_new, ch_new, _ = new_sim.simulate_trials(stim)

        rt_match  = allclose(rt_old,  rt_new)
        ch_match  = allclose(ch_old,  ch_new)
        passed = rt_match and ch_match
        report(f"param_set {i+1}: rt match={rt_match}, choice match={ch_match}", passed)


def test_cpu_simulator_stats(OldSim, NewSim, n_trials: int) -> None:
    """
    CPU (Numba) simulator: prange reorders RNG draws so exact match is not
    expected. Check that mean RT and choice proportion are close.
    """
    print("\n── CPU (Numba) simulator (statistical) ──")
    cohs = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
    stim = make_stimulus(n_trials, cohs)

    old_sim = OldSim(leak=True, time_dependence=True)
    new_sim = NewSim(leak=True, time_dependence=True)

    for sim in (old_sim, new_sim):
        sim.drift_gain    = 7.0
        sim.drift_offset  = 0.0
        sim.a             = 2.0
        sim.z             = 0.5
        sim.ndt           = 0.1
        sim.variance      = 1.0
        sim.leak_rate     = 0.01
        sim.time_constant = 0.5

    # Run multiple reps to get stable statistics
    N_REPS = 5
    old_rts, new_rts, old_chs, new_chs = [], [], [], []
    for seed in range(N_REPS):
        np.random.seed(seed)
        rt, ch, _ = old_sim.simulate_trials(stim)
        old_rts.append(rt[~np.isnan(rt)])
        old_chs.append(ch[~np.isnan(ch)])

        np.random.seed(seed)
        rt, ch, _ = new_sim.simulate_trials(stim)
        new_rts.append(rt[~np.isnan(rt)])
        new_chs.append(ch[~np.isnan(ch)])

    old_mean_rt  = np.mean(np.concatenate(old_rts))
    new_mean_rt  = np.mean(np.concatenate(new_rts))
    old_prop_ch  = np.mean(np.concatenate(old_chs))
    new_prop_ch  = np.mean(np.concatenate(new_chs))

    rt_close  = abs(old_mean_rt - new_mean_rt) < 0.05
    ch_close  = abs(old_prop_ch - new_prop_ch) < 0.05
    report(f"mean RT   old={old_mean_rt:.4f}  new={new_mean_rt:.4f}", rt_close)
    report(f"choice %  old={old_prop_ch:.4f}  new={new_prop_ch:.4f}", ch_close)


def _simulate_data_from_params(
    simulator_cls,
    param_defaults: dict[str, float],
    stim: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Generate synthetic 'empirical' data by running the simulator once at
    known parameters.  Returns exactly len(stim) rows (parallel to stimulus)
    so that boolean masks can be applied to both data and stimulus together,
    as required by the ThreeParamsModel objective function.

    Trials that did not hit a boundary will have NaN rt/choice; the
    LikelihoodCalculator filters those out automatically.
    """
    np.random.seed(seed)
    sim = simulator_cls(leak=True, time_dependence=True)
    for name, val in param_defaults.items():
        setattr(sim, name, val)
    rt, ch, _ = sim.simulate_trials(stim)
    return {
        "signed_coherence": stim[:, 0].copy(),
        "choice":           ch,
        "rt":               rt,
    }


def _eval_nll_across_seeds(model, params, data, stim, seeds, n_reps) -> list[float]:
    """Return NLL for each seed over independent stochastic draws."""
    return [
        model._objective_function(params, data, stim, n_reps=n_reps, seed=s, l1_weight=0.01)
        for s in seeds
    ]


def test_objective_allparams(OldModel, NewModel, OldCPU, n_trials: int, device: str | None) -> None:
    """
    AllParams objective function equivalence check.

    Strategy: generate synthetic data from the simulator itself (not random),
    so the model is a near-perfect fit and NLL values are in a stable range.
    Then compare median NLL across N_SEEDS independent seeds, each with
    N_REPS simulation reps.  Numba prange reorders RNG draws so exact
    agreement per seed is not expected; we check medians agree within 5%.
    """
    print("\n── AllParamsModel objective function ──")
    N_REPS  = 30    # reps per evaluation (↑ reduces per-seed variance)
    SEEDS   = list(range(20))
    TOL     = 0.05

    cohs = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
    stim = make_stimulus(n_trials, cohs)

    old_model = OldModel(enable_leak=True, enable_time_dependence=True, device=device)
    new_model = NewModel(enable_leak=True, enable_time_dependence=True, device=device)

    old_keys = list(old_model._param_bounds.keys())
    new_keys = list(new_model._param_bounds.keys())
    assert old_keys == new_keys, f"Key mismatch:\n  old={old_keys}\n  new={new_keys}"

    # Default parameter values (same for both)
    param_defaults = {k: old_model._param_bounds[k][0] for k in old_keys}
    # Drop the time-indexed suffix params that the CPU sim doesn't have
    sim_defaults = {k: v for k, v in param_defaults.items()
                    if not (k[-1].isdigit())}
    params = np.array([param_defaults[k] for k in old_keys])

    # Generate model-consistent data (not purely random) for stable NLL
    data = _simulate_data_from_params(OldCPU, sim_defaults, stim, seed=99)

    old_nlls = _eval_nll_across_seeds(old_model, params, data, stim, SEEDS, N_REPS)
    new_nlls = _eval_nll_across_seeds(new_model, params, data, stim, SEEDS, N_REPS)

    old_med = float(np.median(old_nlls))
    new_med = float(np.median(new_nlls))
    rel_diff = abs(old_med - new_med) / (abs(old_med) + 1e-8)
    passed = rel_diff < TOL
    report(
        f"median NLL  old={old_med:.4f}  new={new_med:.4f}  rel_diff={rel_diff:.3f}",
        passed,
        f"(n_seeds={len(SEEDS)}, n_reps={N_REPS})",
    )


def test_objective_threeparams(OldModel, NewModel, OldCPU, n_trials: int, device: str | None) -> None:
    """
    ThreeParams objective function equivalence check.
    Same strategy as AllParams: model-generated data + median across seeds.
    """
    print("\n── ThreeParamsModel objective function ──")
    N_REPS  = 30
    SEEDS   = list(range(20))
    TOL     = 0.05

    cohs = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
    stim = make_stimulus(n_trials, cohs)

    old_model = OldModel(enable_leak=True, enable_time_dependence=True, device=device)
    new_model = NewModel(enable_leak=True, enable_time_dependence=True, device=device)

    old_keys = list(old_model._param_bounds.keys())
    new_keys = list(new_model._param_bounds.keys())
    assert old_keys == new_keys, f"Key mismatch:\n  old={old_keys}\n  new={new_keys}"

    param_defaults = {k: old_model._param_bounds[k][0] for k in old_keys}
    # Use the _1-condition defaults (equal prior) for data generation
    sim_defaults = {}
    for k, v in param_defaults.items():
        if k.endswith("_1"):
            sim_defaults[k[:-2]] = v   # strip suffix
        elif not k[-1].isdigit():
            sim_defaults[k] = v
    params = np.array([param_defaults[k] for k in old_keys])

    # Generate model-consistent data with dual-prior labels.
    # prior_block must have the same length as stim (200) so that
    # stimulus[mask] indexing works inside the objective function.
    base_data = _simulate_data_from_params(OldCPU, sim_defaults, stim, seed=99)
    n = len(base_data["rt"])   # == n_trials (one entry per stimulus row)
    blocks = np.array(["equal"] * (n // 2) + ["unequal"] * (n - n // 2))
    data = {**base_data, "prior_block": blocks}

    old_nlls = _eval_nll_across_seeds(old_model, params, data, stim, SEEDS, N_REPS)
    new_nlls = _eval_nll_across_seeds(new_model, params, data, stim, SEEDS, N_REPS)

    old_med = float(np.median(old_nlls))
    new_med = float(np.median(new_nlls))
    rel_diff = abs(old_med - new_med) / (abs(old_med) + 1e-8)
    passed = rel_diff < TOL
    report(
        f"median NLL  old={old_med:.4f}  new={new_med:.4f}  rel_diff={rel_diff:.3f}",
        passed,
        f"(n_seeds={len(SEEDS)}, n_reps={N_REPS})",
    )


# ---------------------------------------------------------------------------
# Speed benchmarks
# ---------------------------------------------------------------------------

def benchmark(label: str, fn, n_runs: int = 10) -> float:
    """Warm up once, then time n_runs iterations. Returns mean seconds."""
    fn()  # warm-up (important for Numba JIT compilation)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    elapsed = (time.perf_counter() - t0) / n_runs
    print(f"  {label:<55} {elapsed*1000:8.2f} ms/call")
    return elapsed


def run_speed_comparison(
    OldCPU, NewCPU,
    OldCUDA, NewCUDA,
    OldAllModel, NewAllModel,
    OldThreeModel, NewThreeModel,
    n_trials: int,
    device: str | None,
) -> None:
    print("\n── Speed comparison ──")
    cohs = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
    stim = make_stimulus(n_trials, cohs)
    rng  = np.random.default_rng(0)
    data_single = make_synthetic_data(n_trials, cohs, dual_prior=False, rng=rng)
    data_dual   = make_synthetic_data(n_trials, cohs, dual_prior=True,  rng=rng)
    params_all   = None
    params_three = None

    # --- CPU simulator ---
    old_cpu = OldCPU(leak=True, time_dependence=True)
    new_cpu = NewCPU(leak=True, time_dependence=True)
    for sim in (old_cpu, new_cpu):
        sim.time_constant = 0.5

    print("\n  CPU simulator:")
    t_old_cpu = benchmark("  old DriftDiffusionSimulator",      lambda: old_cpu.simulate_trials(stim))
    t_new_cpu = benchmark("  new DriftDiffusionSimulator",      lambda: new_cpu.simulate_trials(stim))
    _print_speedup(t_old_cpu, t_new_cpu)

    # --- CUDA simulator ---
    if device == "cuda" and torch.cuda.is_available():
        old_cuda = OldCUDA(leak=True, time_dependence=True, device="cuda")
        new_cuda = NewCUDA(leak=True, time_dependence=True, device="cuda")
        for sim in (old_cuda, new_cuda):
            sim.time_constant = torch.tensor(0.5, device=sim.device, dtype=torch.float32)

        print("\n  CUDA simulator:")
        t_old_gpu = benchmark("  old DriftDiffusionSimulatorCUDA",  lambda: old_cuda.simulate_trials(stim))
        t_new_gpu = benchmark("  new DriftDiffusionSimulatorCUDA",  lambda: new_cuda.simulate_trials(stim))
        _print_speedup(t_old_gpu, t_new_gpu)

    # --- AllParams objective ---
    old_all = OldAllModel(enable_leak=True, enable_time_dependence=True, device=device)
    new_all = NewAllModel(enable_leak=True, enable_time_dependence=True, device=device)
    params_all = np.array([old_all._param_bounds[k][0] for k in old_all._param_bounds])

    print("\n  AllParams objective function:")
    t_old_all = benchmark(
        "  old DecisionModel._objective_function",
        lambda: old_all._objective_function(params_all, data_single, stim, n_reps=3, seed=42, l1_weight=0.01),
    )
    t_new_all = benchmark(
        "  new AllParamsModel._objective_function",
        lambda: new_all._objective_function(params_all, data_single, stim, n_reps=3, seed=42, l1_weight=0.01),
    )
    _print_speedup(t_old_all, t_new_all)

    # --- ThreeParams objective ---
    old_three = OldThreeModel(enable_leak=True, enable_time_dependence=True, device=device)
    new_three = NewThreeModel(enable_leak=True, enable_time_dependence=True, device=device)
    params_three = np.array([old_three._param_bounds[k][0] for k in old_three._param_bounds])

    print("\n  ThreeParams objective function:")
    t_old_three = benchmark(
        "  old DecisionModel._objective_function",
        lambda: old_three._objective_function(params_three, data_dual, stim, n_reps=3, seed=42, l1_weight=0.01),
    )
    t_new_three = benchmark(
        "  new ThreeParamsModel._objective_function",
        lambda: new_three._objective_function(params_three, data_dual, stim, n_reps=3, seed=42, l1_weight=0.01),
    )
    _print_speedup(t_old_three, t_new_three)


def _print_speedup(t_old: float, t_new: float) -> None:
    ratio = t_old / t_new if t_new > 0 else float("inf")
    direction = "faster" if ratio >= 1 else "slower"
    print(f"  {'speedup: new is':.<55} {abs(ratio):.2f}x {direction}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate DDM refactor equivalence and speed")
    parser.add_argument("--device",   default="cuda", choices=["cuda", "cpu"],
                        help="Device for CUDA tests (default: cuda)")
    parser.add_argument("--n_trials", type=int, default=200,
                        help="Trials per simulation call (default: 200)")
    parser.add_argument("--n_reps",   type=int, default=10,
                        help="Benchmark repetitions (default: 10)")
    args = parser.parse_args()

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else None

    # ---- load modules ----
    here = Path(__file__).parent
    old_all_mod   = load_module("old_all_params",   here / "full_ddm_vary_all_params.py")
    old_three_mod = load_module("old_three_params",  here / "full_ddm_vary_three_params.py")

    sys.path.insert(0, str(here))
    from ddm_base  import DriftDiffusionSimulator  as NewCPU
    from ddm_base  import DriftDiffusionSimulatorCUDA as NewCUDA
    from ddm_base  import LikelihoodCalculator     as NewLC
    from ddm_model import AllParamsModel, ThreeParamsModel

    OldCPU       = old_all_mod.DriftDiffusionSimulator
    OldCUDA      = old_all_mod.DriftDiffusionSimulatorCUDA
    OldLC        = old_all_mod.LikelihoodCalculator
    OldAllModel  = old_all_mod.DecisionModel
    OldThreeModel = old_three_mod.DecisionModel

    print("=" * 65)
    print("  DDM Refactor Validation")
    print(f"  device={device or 'cpu'}  n_trials={args.n_trials}")
    print("=" * 65)

    # ---- correctness tests ----
    test_likelihood_exact(OldLC, NewLC)
    test_cpu_simulator_stats(OldCPU, NewCPU, args.n_trials)

    if device == "cuda":
        test_torch_simulator_exact(OldCUDA, NewCUDA, "cuda", args.n_trials)
    else:
        test_torch_simulator_exact(OldCUDA, NewCUDA, "cpu", args.n_trials)

    test_objective_allparams(OldAllModel, AllParamsModel, OldCPU, args.n_trials, device)
    test_objective_threeparams(OldThreeModel, ThreeParamsModel, OldCPU, args.n_trials, device)

    # ---- speed benchmarks ----
    run_speed_comparison(
        OldCPU, NewCPU,
        OldCUDA, NewCUDA,
        OldAllModel, AllParamsModel,
        OldThreeModel, ThreeParamsModel,
        n_trials=args.n_trials,
        device=device,
    )

    print("\n" + "=" * 65)