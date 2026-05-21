import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import traceback
import json
from datetime import datetime

from config import dir_config
from src.ddm.ddm import DEFAULT_PARAMS, DecisionModel, ParamSpec, validate_params


DT               = 0.001
CAF_WEIGHT       = 1
CAF_BINS = 15
RT_VAR_WEIGHT    = 1.0

class DDMModel(DecisionModel):

    def __init__(self, enable_leak=False, enable_time_constant=False,
                device='cpu', likelihood_params=None):
        super().__init__(device=device, likelihood_params=likelihood_params)
        self.enable_leak = enable_leak
        self.enable_time_constant = enable_time_constant

    @property
    def param_specs(self) -> dict[str, ParamSpec]:
        return {
            "ndt":           ParamSpec(0.2,  (0.1,  1.0)),
            "a":             ParamSpec(2.0,  (0.8,  6.0)),
            "z":             ParamSpec(0.5,  (0.1,  0.9)),
            "drift_gain":    ParamSpec(7.0,  (1.0, 10.0)),
            "drift_offset":  ParamSpec(0.0,  (-5.0, 5.0)),
            "leak_rate":     ParamSpec(0.0,  (0.0,  1.0)),
            "variance":      ParamSpec(1.0,  (0.5,  2.0)),
            "time_constant": ParamSpec(0.0,  (-2.0, 2.0)),
        }

    def _build_params(self, values: np.ndarray, condition_idx: int = None) -> dict:
        p = dict(DEFAULT_PARAMS)

        for key, val in zip(self.param_specs.keys(), values):
            p[key] = val

        p["variance"] = 1.0

        if not self.enable_leak:
            p["leak_rate"] = 0.0
        if not self.enable_time_constant:
            p["time_constant"] = 0.0

        validate_params(p)
        return p

    def _objective_function(self, values, data, stimulus, n_reps, seed, l1_weight):
        try:
            params = self._build_params(values)
        except ValueError:
            return 1e6

        result = self._simulate_condition(stimulus, params, n_reps)
        if result is None:
            return 1e6

        return self.likelihood_calc.calculate_likelihood(
            result["rt"], result["choice"],
            np.asarray(data["rt"]), np.asarray(data["choice"]),
            result["signed_coherence"], np.asarray(data["signed_coherence"]),
        )


def build_grid(behavior_df: pd.DataFrame) -> list[dict]:

    session_ids = np.sort(behavior_df["session_id"].unique())
    prior_blocks = np.sort(behavior_df["prior_block"].unique())

    variants = [
        (False, False),
        (False, True),
        (True,  False),
        (True,  True),
    ]

    return [
        {
            "session_id": session_id,
            "prior_block": prior_block,
            "enable_leak": enable_leak,
            "enable_time_constant": enable_time_constant,
        }
        for enable_leak, enable_time_constant in variants
        for session_id in session_ids
        for prior_block in prior_blocks
    ]


def get_job(grid: list[dict], job_id: int) -> dict:

    if job_id >= len(grid):
        raise ValueError(
            f"job_id {job_id} out of bounds "
            f"(max={len(grid)-1})"
        )

    return grid[job_id]


def get_output_dir(
    ddm_dir: Path,
    enable_leak: bool,
    enable_time_constant: bool,
) -> Path:

    model_name = (
        f"leak-{int(enable_leak)}_"
        f"tc-{int(enable_time_constant)}"
    )

    output_dir = ddm_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def load_behavior_data(ddm_dir: Path) -> pd.DataFrame:

    behavior_path = ddm_dir / "behavior_data.csv"

    if not behavior_path.exists():
        raise FileNotFoundError(f"Missing file: {behavior_path}")

    return pd.read_csv(behavior_path)


def prepare_data(
    behavior_df: pd.DataFrame,
    session_id,
    prior_block,
) -> pd.DataFrame:

    data = behavior_df[
        (behavior_df["session_id"] == session_id) &
        (behavior_df["prior_block"] == prior_block)
    ][["rt", "choice", "signed_coherence"]].copy()

    if data.empty:
        raise ValueError(
            f"No data for session={session_id}, "
            f"prior_block={prior_block}"
        )

    data["choice"] = data["choice"].astype(int)

    return data


def build_stimulus(data: pd.DataFrame) -> np.ndarray:

    stimulus_length = max(100, int(np.clip(data["rt"].max(), 0, 5) * 1000))

    return np.tile(
        data["signed_coherence"].to_numpy()[:, None],
        (1, stimulus_length)
    )


def data_verification(data: pd.DataFrame):

    if data["rt"].isnull().any():
        raise ValueError("Missing RT values")

    if data["choice"].isnull().any():
        raise ValueError("Missing choice values")

    if data["signed_coherence"].isnull().any():
        raise ValueError("Missing signed coherence values")

    # rts should be in seconds
    if data["rt"].max() > 10:
        raise ValueError("RT values out of expected range (0.1s to 10s)")

    # signed coherence should be between -1 and 1
    if data["signed_coherence"].min() < -1 or data["signed_coherence"].max() > 1:
        raise ValueError("Signed coherence values out of expected range (-1 to 1)")

def fit_model(
    data: pd.DataFrame,
    stimulus: np.ndarray,
    enable_leak: bool,
    enable_time_constant: bool,
):

    data_verification(data)

    model = DDMModel(
        enable_leak=enable_leak,
        enable_time_constant=enable_time_constant,
        device='cuda',
        likelihood_params={
            "caf_weight": CAF_WEIGHT,
            "caf_bins": CAF_BINS,
            "rt_var_weight": RT_VAR_WEIGHT,
            }
    )

    result = model.fit(
        data=data,
        stimulus=stimulus,
        n_reps=15,
        max_iterations=500,
        l1_weight=0.01,
        verbose=False,
    )

    return model, result


def save_results(output_dir: Path, session_id, prior_block, model, result, job):
    out_path = (output_dir / f"{session_id}_prior_block_{prior_block}.pkl")

    # if model is on gpu move it to cpu before saving
    if model.device.type == "cuda":
        model.to("cpu")

    with open(out_path, "wb") as f:
        pickle.dump({
                "model": model,
                "results": result,
                "job": job,
        }, f,
        )

    logger.info(f"Saved: {out_path}")


def save_failure(output_dir, session_id, prior_block, job, stage, error):
    fail_path = output_dir / f"{session_id}_prior_block_{prior_block}.FAILED.json"

    payload = {
        "session_id": session_id,
        "prior_block": prior_block,
        "job": job,
        "stage": stage,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "timestamp": datetime.now().isoformat(),
    }

    with open(fail_path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.error(f"[FAILED] {stage} → {fail_path}")


if __name__ == "__main__":
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    # ----------------------------
    # args
    # ----------------------------
    parser = argparse.ArgumentParser(description="Fit DDM model")
    parser.add_argument("--job_id", type=int, required=True)
    args = parser.parse_args()

    # ----------------------------
    # paths
    # ----------------------------
    ddm_dir = Path(dir_config.data.processed) / "ddm"

    # ----------------------------
    # cuda check
    # ----------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    # ----------------------------
    # data + job
    # ----------------------------
    behavior_df = load_behavior_data(ddm_dir)

    grid = build_grid(behavior_df)

    job = get_job(grid, args.job_id)

    session_id = job["session_id"]
    prior_block = job["prior_block"]

    enable_leak = job["enable_leak"]
    enable_time_constant = job["enable_time_constant"]



    logger.info(f"[START] job_id={args.job_id}:  Session ID - {session_id}, Prior Block - {prior_block}, Enable Leak - {enable_leak}, Enable Time Constant - {enable_time_constant}")

    # ----------------------------
    # output dir
    # ----------------------------
    output_dir = get_output_dir(ddm_dir, enable_leak, enable_time_constant)
    logger.info(f"Output directory: {output_dir}")

    # ----------------------------
    # prepare data
    # ----------------------------
    try:
        data = prepare_data(behavior_df, session_id, prior_block)

        stimulus = build_stimulus(data)
    except Exception as e:
        logger.error(f"[FAILED] data_prep: {e}")
        save_failure(output_dir, session_id, prior_block, job, "data_prep", e)
        raise
    # ----------------------------
    # fit
    # ----------------------------
    try:
        model, result = fit_model(data, stimulus, enable_leak, enable_time_constant)
    except Exception as e:
        logger.error(f"[FAILED] fit_model: {e}")
        save_failure(output_dir, session_id, prior_block, job, "fit_model", e)
        raise

    # ----------------------------
    # save
    # ----------------------------
    try:
        save_results(output_dir, session_id, prior_block, model, result, job)
    except Exception as e:
        logger.error(f"[FAILED] save_results: {e}")
        save_failure(output_dir, session_id, prior_block, job, "save_results", e)
        raise
