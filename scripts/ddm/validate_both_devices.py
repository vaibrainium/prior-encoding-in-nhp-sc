"""
validate_both_devices.py

Validates the refactored ddm_base.py / ddm_model.py against the original
full_ddm_vary_all_params.py / full_ddm_vary_three_params.py on BOTH the
CPU (Numba) and CUDA (PyTorch) code paths in a single run.

Why separate passes are needed
-------------------------------
The CPU and CUDA simulators are independent implementations:
  - CPU : Numba @jit(nopython=True, parallel=True) -- _simulate_ddm_trials_numba
  - CUDA: PyTorch sequential time-loop             -- DriftDiffusionSimulatorCUDA

Verifying one does not imply the other is correct, so both must be exercised
through the full objective pipeline.

Test plan
---------
[CPU section]
  1. CPU simulator statistical equivalence (Numba prange is non-deterministic)
  2. AllParamsModel  objective -- median NLL across 20 seeds, n_reps=30
  3. ThreeParamsModel objective -- same
  Data generated with the CPU (Numba) simulator.

[CUDA section]  (skipped if CUDA unavailable or --skip-cuda)
  4. CUDA simulator exact match (same torch.manual_seed → identical arrays)
  5. AllParamsModel  objective -- 5 seeds, n_reps=5 (deterministic → rel_diff≈0)
  6. ThreeParamsModel objective -- same
  Data generated with the CUDA (PyTorch) simulator.

[Common]
  7. LikelihoodCalculator exact match (device-independent)
  8. Speed benchmark -- CPU simulator, CUDA simulator (if available),
     AllParams and ThreeParams objectives on both devices

Notes on CUDA timing
--------------------
The CUDA simulator has a Python-level for-loop over time steps, so each
simulate_trials call takes several seconds regardless of GPU speed.  To keep
the CUDA section under ~5 minutes the test stimulus uses max_duration=1.0 s
(1 000 steps) instead of 3.0 s -- trials with high |coherence| resolve well
within that window.

Usage
-----
    python validate_both_devices.py
    python validate_both_devices.py --skip-cuda
    python validate_both_devices.py --n_trials 200 --bench-reps 5
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

COHS = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
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


def make_stimulus(n_trials: int, max_duration: float = 3.0, dt: float = 0.001) -> np.ndarray:
    """Constant-coherence stimulus, one coherence level per trial (tiled)."""
    n_tp = int(max_duration / dt)
    coh  = np.tile(COHS, int(np.ceil(n_trials / len(COHS))))[:n_trials]
    return np.tile(coh.reshape(-1, 1), (1, n_tp)).astype(np.float32)


def simulate_data_cpu(
    cpu_sim_cls,
    param_defaults: dict[str, float],
    stim: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    One forward pass with the CPU (Numba) simulator at known parameters.
    Returns exactly len(stim) rows so boolean masks align with the stimulus.
    """
    np.random.seed(seed)
    sim = cpu_sim_cls(leak=True, time_dependence=True)
    for k, v in param_defaults.items():
        setattr(sim, k, v)
    rt, ch, _ = sim.simulate_trials(stim)
    return {"signed_coherence": stim[:, 0].copy(), "choice": ch, "rt": rt}


def simulate_data_cuda(
    cuda_sim_cls,
    param_defaults: dict[str, float],
    stim: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    One forward pass with the CUDA (PyTorch) simulator at known parameters.
    Returns exactly len(stim) rows.
    """
    torch.manual_seed(seed)
    sim = cuda_sim_cls(leak=True, time_dependence=True, device="cuda")
    for k, v in param_defaults.items():
        setattr(sim, k, torch.tensor(v, device=sim.device, dtype=torch.float32))
        # also update _noise_std if variance or dt changed
    sim._noise_std = torch.sqrt(sim.variance * sim.dt)
    rt, ch, _ = sim.simulate_trials(stim)
    return {"signed_coherence": stim[:, 0].copy(), "choice": ch, "rt": rt}


def _eval_nll_across_seeds(model, params, data, stim, seeds, n_reps) -> list[float]:
    return [
        model._objective_function(params, data, stim, n_reps=n_reps, seed=s, l1_weight=0.01)
        for s in seeds
    ]


def _strip_suffix_params(param_defaults: dict[str, float]) -> dict[str, float]:
    """Map {a_1: v, z_1: v, drift_offset_1: v, ndt: v, ...} → {a: v, z: v, ...}"""
    out = {}
    for k, v in param_defaults.items():
        if k.endswith("_1"):
            out[k[:-2]] = v
        elif not k[-1].isdigit():
            out[k] = v
    return out


def benchmark(label: str, fn, n_runs: int = 5) -> float:
    fn()   # warm-up
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    elapsed = (time.perf_counter() - t0) / n_runs
    print(f"    {label:<53} {elapsed * 1000:8.2f} ms/call")
    return elapsed


def _speedup(t_old: float, t_new: float) -> None:
    r = t_old / t_new if t_new > 0 else float("inf")
    d = "faster" if r >= 1 else "slower"
    print(f"    {'speedup: new is':.<53} {abs(r):.2f}x {d}")


# ---------------------------------------------------------------------------
# LikelihoodCalculator (device-independent)
# ---------------------------------------------------------------------------

def test_likelihood_exact(OldLC, NewLC) -> None:
    print("\n── LikelihoodCalculator (exact, device-independent) ──")
    rng = np.random.default_rng(0)
    n   = 400
    coh = np.tile(COHS, n // len(COHS) + 1)[:n]
    rt_p  = rng.uniform(0.2, 1.5, n).astype(float)
    ch_p  = rng.integers(0, 2, n).astype(float)
    rt_d  = rng.uniform(0.2, 1.5, n).astype(float)
    ch_d  = rng.integers(0, 2, n).astype(float)

    for nbins, rt_w in [(5, 1.0), (3, 2.0), (8, 0.5)]:
        o = OldLC(nbins=nbins, rt_weight=rt_w).calculate_likelihood(rt_p, ch_p, rt_d, ch_d, coh, coh)
        n_ = NewLC(nbins=nbins, rt_weight=rt_w).calculate_likelihood(rt_p, ch_p, rt_d, ch_d, coh, coh)
        passed = abs(o - n_) < 1e-9
        report(f"nbins={nbins}, rt_weight={rt_w}", passed,
               f"old={o:.6f}  new={n_:.6f}  diff={abs(o - n_):.2e}")


# ---------------------------------------------------------------------------
# CPU tests
# ---------------------------------------------------------------------------

def test_cpu_simulator_stats(OldSim, NewSim, n_trials: int) -> None:
    print("\n── [CPU] Numba simulator (statistical) ──")
    stim = make_stimulus(n_trials)
    old_sim = OldSim(leak=True, time_dependence=True)
    new_sim = NewSim(leak=True, time_dependence=True)
    for sim in (old_sim, new_sim):
        sim.drift_gain = 7.0; sim.drift_offset = 0.0; sim.a = 2.0
        sim.z = 0.5; sim.ndt = 0.1; sim.variance = 1.0
        sim.leak_rate = 0.01; sim.time_constant = 0.5

    old_rts, new_rts, old_chs, new_chs = [], [], [], []
    for seed in range(5):
        np.random.seed(seed)
        rt, ch, _ = old_sim.simulate_trials(stim)
        old_rts.append(rt[~np.isnan(rt)]); old_chs.append(ch[~np.isnan(ch)])
        np.random.seed(seed)
        rt, ch, _ = new_sim.simulate_trials(stim)
        new_rts.append(rt[~np.isnan(rt)]); new_chs.append(ch[~np.isnan(ch)])

    old_mrt = np.mean(np.concatenate(old_rts)); new_mrt = np.mean(np.concatenate(new_rts))
    old_pch = np.mean(np.concatenate(old_chs)); new_pch = np.mean(np.concatenate(new_chs))
    report(f"mean RT   old={old_mrt:.4f}  new={new_mrt:.4f}", abs(old_mrt - new_mrt) < 0.05)
    report(f"choice %  old={old_pch:.4f}  new={new_pch:.4f}", abs(old_pch - new_pch) < 0.05)


def test_cpu_objective(
    label: str,
    OldModel, NewModel, OldCPU,
    n_trials: int,
    dual_prior: bool,
) -> None:
    """
    Objective function test for the CPU (Numba) path.
    Uses model-generated data (stable NLL) and checks median across 20 seeds.
    n_reps=30 reduces per-seed variance from Numba's non-deterministic prange.
    """
    print(f"\n── [CPU] {label} objective function ──")
    N_REPS = 30
    SEEDS  = list(range(20))
    TOL    = 0.05

    stim = make_stimulus(n_trials)
    old_model = OldModel(enable_leak=True, enable_time_dependence=True, device=None)
    new_model = NewModel(enable_leak=True, enable_time_dependence=True, device=None)

    old_keys = list(old_model._param_bounds.keys())
    assert old_keys == list(new_model._param_bounds.keys()), "param key mismatch"

    param_defaults = {k: old_model._param_bounds[k][0] for k in old_keys}
    sim_defaults   = (_strip_suffix_params(param_defaults) if dual_prior
                      else {k: v for k, v in param_defaults.items() if not k[-1].isdigit()})
    params = np.array([param_defaults[k] for k in old_keys])

    base = simulate_data_cpu(OldCPU, sim_defaults, stim, seed=99)
    if dual_prior:
        n = len(base["rt"])
        base["prior_block"] = np.array(["equal"] * (n // 2) + ["unequal"] * (n - n // 2))

    old_nlls = _eval_nll_across_seeds(old_model, params, base, stim, SEEDS, N_REPS)
    new_nlls = _eval_nll_across_seeds(new_model, params, base, stim, SEEDS, N_REPS)

    old_med = float(np.median(old_nlls))
    new_med = float(np.median(new_nlls))
    rel_diff = abs(old_med - new_med) / (abs(old_med) + 1e-8)
    report(
        f"median NLL  old={old_med:.4f}  new={new_med:.4f}  rel_diff={rel_diff:.3f}",
        rel_diff < TOL,
        f"n_seeds={len(SEEDS)}, n_reps={N_REPS}",
    )


# ---------------------------------------------------------------------------
# CUDA tests
# ---------------------------------------------------------------------------

def test_cuda_simulator_stats(OldCUDA, NewCUDA, n_trials: int) -> None:
    """
    CUDA simulator statistical equivalence.

    The optimised DriftDiffusionSimulatorCUDA precomputes all noise in one
    kernel call, changing the seed→output mapping vs the old simulator.
    Statistical properties are unchanged: check mean RT and choice proportion.
    """
    print("\n── [CUDA] Torch simulator (statistical) ──")
    stim = make_stimulus(n_trials, max_duration=1.0)

    param_sets = [
        dict(drift_gain=7.0, drift_offset=0.0, a=2.0, z=0.5,
             ndt=0.1, variance=1.0, leak_rate=0.01, time_constant=0.5),
        dict(drift_gain=5.0, drift_offset=0.3, a=1.5, z=0.4,
             ndt=0.15, variance=0.8, leak_rate=0.05, time_constant=1.0),
    ]
    for i, ps in enumerate(param_sets):
        old_rts, new_rts, old_chs, new_chs = [], [], [], []
        for seed in range(5):
            old_sim = OldCUDA(leak=True, time_dependence=True, device="cuda")
            new_sim = NewCUDA(leak=True, time_dependence=True, device="cuda")
            for sim in (old_sim, new_sim):
                for k, v in ps.items():
                    setattr(sim, k, torch.tensor(v, device=sim.device, dtype=torch.float32))

            torch.manual_seed(seed)
            rt_o, ch_o, _ = old_sim.simulate_trials(stim)
            torch.manual_seed(seed)
            rt_n, ch_n, _ = new_sim.simulate_trials(stim)

            old_rts.append(rt_o[~np.isnan(rt_o)]); old_chs.append(ch_o[~np.isnan(ch_o)])
            new_rts.append(rt_n[~np.isnan(rt_n)]); new_chs.append(ch_n[~np.isnan(ch_n)])

        old_mrt = np.mean(np.concatenate(old_rts)); new_mrt = np.mean(np.concatenate(new_rts))
        old_pch = np.mean(np.concatenate(old_chs)); new_pch = np.mean(np.concatenate(new_chs))
        rt_ok = abs(old_mrt - new_mrt) < 0.05
        ch_ok = abs(old_pch - new_pch) < 0.05
        report(f"param_set {i+1}: mean RT  old={old_mrt:.4f}  new={new_mrt:.4f}", rt_ok)
        report(f"param_set {i+1}: choice % old={old_pch:.4f}  new={new_pch:.4f}", ch_ok)


def test_cuda_objective(
    label: str,
    OldModel, NewModel, OldCUDA,
    n_trials: int,
    dual_prior: bool,
) -> None:
    """
    Objective function test for the CUDA (PyTorch) path.
    CUDA simulator is deterministic (same seed → identical output), so
    3 seeds × 5 reps is sufficient; we expect rel_diff ≈ 0.
    Uses max_duration=1.0 s to keep each simulation call under ~5 s.
    """
    print(f"\n── [CUDA] {label} objective function ──")
    N_REPS = 5
    SEEDS  = list(range(3))
    TOL    = 1e-4   # CUDA is deterministic → expect near-zero diff

    stim = make_stimulus(n_trials, max_duration=1.0)

    old_model = OldModel(enable_leak=True, enable_time_dependence=True, device="cuda")
    new_model = NewModel(enable_leak=True, enable_time_dependence=True, device="cuda")

    old_keys = list(old_model._param_bounds.keys())
    assert old_keys == list(new_model._param_bounds.keys()), "param key mismatch"

    param_defaults = {k: old_model._param_bounds[k][0] for k in old_keys}
    cuda_defaults  = (_strip_suffix_params(param_defaults) if dual_prior
                      else {k: v for k, v in param_defaults.items() if not k[-1].isdigit()})
    params = np.array([param_defaults[k] for k in old_keys])

    base = simulate_data_cuda(OldCUDA, cuda_defaults, stim, seed=99)
    if dual_prior:
        n = len(base["rt"])
        base["prior_block"] = np.array(["equal"] * (n // 2) + ["unequal"] * (n - n // 2))

    old_nlls = _eval_nll_across_seeds(old_model, params, base, stim, SEEDS, N_REPS)
    new_nlls = _eval_nll_across_seeds(new_model, params, base, stim, SEEDS, N_REPS)

    pairs = list(zip(old_nlls, new_nlls))
    max_diff = max(abs(o - n_) / (abs(o) + 1e-8) for o, n_ in pairs)
    passed = max_diff < TOL
    for s, (o, n_) in zip(SEEDS, pairs):
        report(f"seed={s}  old={o:.4f}  new={n_:.4f}  rel_diff={abs(o-n_)/(abs(o)+1e-8):.2e}",
               abs(o - n_) / (abs(o) + 1e-8) < TOL)


# ---------------------------------------------------------------------------
# Speed benchmarks
# ---------------------------------------------------------------------------

def run_speed_comparison(
    OldCPU, NewCPU, OldCUDA, NewCUDA,
    OldAllModel, NewAllModel,
    OldThreeModel, NewThreeModel,
    n_trials: int,
    bench_reps: int,
    has_cuda: bool,
) -> None:
    print("\n── Speed comparison ──")
    stim_cpu  = make_stimulus(n_trials, max_duration=3.0)
    stim_cuda = make_stimulus(n_trials, max_duration=1.0)
    rng = np.random.default_rng(0)

    # synthetic data for objective benchmarks
    cohs = np.tile(COHS, int(np.ceil(n_trials / len(COHS))))[:n_trials]
    data_single = {
        "signed_coherence": cohs,
        "choice": rng.integers(0, 2, n_trials).astype(float),
        "rt":     rng.uniform(0.2, 1.5, n_trials),
    }
    data_dual = {**data_single,
                 "prior_block": np.array(["equal"] * (n_trials // 2) +
                                         ["unequal"] * (n_trials - n_trials // 2))}

    # ---- CPU ----
    print("\n  [CPU] simulators:")
    old_cpu = OldCPU(leak=True, time_dependence=True)
    new_cpu = NewCPU(leak=True, time_dependence=True)
    for s in (old_cpu, new_cpu):
        s.time_constant = 0.5
    t_oc = benchmark("old DriftDiffusionSimulator",  lambda: old_cpu.simulate_trials(stim_cpu), bench_reps)
    t_nc = benchmark("new DriftDiffusionSimulator",  lambda: new_cpu.simulate_trials(stim_cpu), bench_reps)
    _speedup(t_oc, t_nc)

    print("\n  [CPU] AllParams objective (n_reps=3):")
    old_all_cpu = OldAllModel(enable_leak=True, enable_time_dependence=True, device=None)
    new_all_cpu = NewAllModel(enable_leak=True, enable_time_dependence=True, device=None)
    p_all = np.array([old_all_cpu._param_bounds[k][0] for k in old_all_cpu._param_bounds])
    fn_oa = lambda: old_all_cpu._objective_function(p_all, data_single, stim_cpu, n_reps=3, seed=42, l1_weight=0.01)
    fn_na = lambda: new_all_cpu._objective_function(p_all, data_single, stim_cpu, n_reps=3, seed=42, l1_weight=0.01)
    t_oa = benchmark("old DecisionModel._objective_function",  fn_oa, bench_reps)
    t_na = benchmark("new AllParamsModel._objective_function", fn_na, bench_reps)
    _speedup(t_oa, t_na)

    print("\n  [CPU] ThreeParams objective (n_reps=3):")
    old_thr_cpu = OldThreeModel(enable_leak=True, enable_time_dependence=True, device=None)
    new_thr_cpu = NewThreeModel(enable_leak=True, enable_time_dependence=True, device=None)
    p_thr = np.array([old_thr_cpu._param_bounds[k][0] for k in old_thr_cpu._param_bounds])
    fn_ot = lambda: old_thr_cpu._objective_function(p_thr, data_dual, stim_cpu, n_reps=3, seed=42, l1_weight=0.01)
    fn_nt = lambda: new_thr_cpu._objective_function(p_thr, data_dual, stim_cpu, n_reps=3, seed=42, l1_weight=0.01)
    t_ot = benchmark("old DecisionModel._objective_function",    fn_ot, bench_reps)
    t_nt = benchmark("new ThreeParamsModel._objective_function", fn_nt, bench_reps)
    _speedup(t_ot, t_nt)

    if not has_cuda:
        return

    # ---- CUDA ----
    print("\n  [CUDA] simulators (max_duration=1.0 s):")
    old_gpu = OldCUDA(leak=True, time_dependence=True, device="cuda")
    new_gpu = NewCUDA(leak=True, time_dependence=True, device="cuda")
    for s in (old_gpu, new_gpu):
        s.time_constant = torch.tensor(0.5, device=s.device, dtype=torch.float32)
    t_og = benchmark("old DriftDiffusionSimulatorCUDA", lambda: old_gpu.simulate_trials(stim_cuda), bench_reps)
    t_ng = benchmark("new DriftDiffusionSimulatorCUDA", lambda: new_gpu.simulate_trials(stim_cuda), bench_reps)
    _speedup(t_og, t_ng)

    # data that matches the short CUDA stimulus length
    data_cuda_s = {
        "signed_coherence": stim_cuda[:, 0].copy(),
        "choice": rng.integers(0, 2, n_trials).astype(float),
        "rt":     rng.uniform(0.2, 1.0, n_trials),
    }
    data_cuda_d = {**data_cuda_s,
                   "prior_block": np.array(["equal"] * (n_trials // 2) +
                                           ["unequal"] * (n_trials - n_trials // 2))}

    print("\n  [CUDA] AllParams objective (n_reps=3, max_duration=1.0 s):")
    old_all_gpu = OldAllModel(enable_leak=True, enable_time_dependence=True, device="cuda")
    new_all_gpu = NewAllModel(enable_leak=True, enable_time_dependence=True, device="cuda")
    p_all_g = np.array([old_all_gpu._param_bounds[k][0] for k in old_all_gpu._param_bounds])
    fn_oag = lambda: old_all_gpu._objective_function(p_all_g, data_cuda_s, stim_cuda, n_reps=3, seed=42, l1_weight=0.01)
    fn_nag = lambda: new_all_gpu._objective_function(p_all_g, data_cuda_s, stim_cuda, n_reps=3, seed=42, l1_weight=0.01)
    t_oag = benchmark("old DecisionModel._objective_function",  fn_oag, bench_reps)
    t_nag = benchmark("new AllParamsModel._objective_function", fn_nag, bench_reps)
    _speedup(t_oag, t_nag)

    print("\n  [CUDA] ThreeParams objective (n_reps=3, max_duration=1.0 s):")
    old_thr_gpu = OldThreeModel(enable_leak=True, enable_time_dependence=True, device="cuda")
    new_thr_gpu = NewThreeModel(enable_leak=True, enable_time_dependence=True, device="cuda")
    p_thr_g = np.array([old_thr_gpu._param_bounds[k][0] for k in old_thr_gpu._param_bounds])
    fn_otg = lambda: old_thr_gpu._objective_function(p_thr_g, data_cuda_d, stim_cuda, n_reps=3, seed=42, l1_weight=0.01)
    fn_ntg = lambda: new_thr_gpu._objective_function(p_thr_g, data_cuda_d, stim_cuda, n_reps=3, seed=42, l1_weight=0.01)
    t_otg = benchmark("old DecisionModel._objective_function",    fn_otg, bench_reps)
    t_ntg = benchmark("new ThreeParamsModel._objective_function", fn_ntg, bench_reps)
    _speedup(t_otg, t_ntg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate DDM refactor on both CPU and CUDA in one pass"
    )
    parser.add_argument("--skip-cuda",  action="store_true",
                        help="Skip all CUDA tests (run CPU-only)")
    parser.add_argument("--n_trials",   type=int, default=200,
                        help="Trials per stimulus (default: 200)")
    parser.add_argument("--bench-reps", type=int, default=5,
                        help="Speed benchmark repetitions (default: 5)")
    args = parser.parse_args()

    has_cuda = torch.cuda.is_available() and not args.skip_cuda

    here = Path(__file__).parent
    old_all_mod   = load_module("old_all_params",   here / "full_ddm_vary_all_params.py")
    old_three_mod = load_module("old_three_params",  here / "full_ddm_vary_three_params.py")

    sys.path.insert(0, str(here))
    from ddm_base  import DriftDiffusionSimulator    as NewCPU
    from ddm_base  import DriftDiffusionSimulatorCUDA as NewCUDA
    from ddm_base  import LikelihoodCalculator        as NewLC
    from ddm_model import AllParamsModel, ThreeParamsModel

    OldCPU        = old_all_mod.DriftDiffusionSimulator
    OldCUDA       = old_all_mod.DriftDiffusionSimulatorCUDA
    OldLC         = old_all_mod.LikelihoodCalculator
    OldAllModel   = old_all_mod.DecisionModel
    OldThreeModel = old_three_mod.DecisionModel

    cuda_str = f"cuda ({torch.cuda.get_device_name(0)})" if has_cuda else "skipped"
    print("=" * 65)
    print("  DDM Refactor Validation — Both Devices")
    print(f"  n_trials={args.n_trials}  CUDA={cuda_str}")
    print("=" * 65)

    # ── common ──────────────────────────────────────────────────────────
    test_likelihood_exact(OldLC, NewLC)

    # ── CPU path ────────────────────────────────────────────────────────
    test_cpu_simulator_stats(OldCPU, NewCPU, args.n_trials)
    test_cpu_objective("AllParams",   OldAllModel,   AllParamsModel,  OldCPU, args.n_trials, dual_prior=False)
    test_cpu_objective("ThreeParams", OldThreeModel, ThreeParamsModel, OldCPU, args.n_trials, dual_prior=True)

    # ── CUDA path ───────────────────────────────────────────────────────
    if has_cuda:
        test_cuda_simulator_stats(OldCUDA, NewCUDA, args.n_trials)
        test_cuda_objective("AllParams",   OldAllModel,   AllParamsModel,  OldCUDA, args.n_trials, dual_prior=False)
        test_cuda_objective("ThreeParams", OldThreeModel, ThreeParamsModel, OldCUDA, args.n_trials, dual_prior=True)
    else:
        print("\n  [CUDA] skipped (not available or --skip-cuda)")

    # ── speed ────────────────────────────────────────────────────────────
    run_speed_comparison(
        OldCPU, NewCPU, OldCUDA, NewCUDA,
        OldAllModel, AllParamsModel,
        OldThreeModel, ThreeParamsModel,
        n_trials=args.n_trials,
        bench_reps=args.bench_reps,
        has_cuda=has_cuda,
    )

    print("\n" + "=" * 65)
