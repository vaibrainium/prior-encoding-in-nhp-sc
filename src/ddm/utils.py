import numpy as np
import pandas as pd
import pickle
import io
import torch


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

def build_stimulus(data: pd.DataFrame, rt_buffer: float = 1.5, max_seconds: float = 8.0) -> np.ndarray:
    max_rt = float(np.clip(data["rt"].max() * rt_buffer, 0, max_seconds))
    stimulus_length = max(100, int(max_rt * 1000))

    return np.tile(
        data["signed_coherence"].to_numpy()[:, None],
        (1, stimulus_length)
    )

def build_grid(behavior_df: pd.DataFrame) -> list[dict]:

    session_ids = np.sort(behavior_df["session_id"].unique())
    prior_blocks = np.sort(behavior_df["prior_block"].unique())

    variants = [
        {"enable_leak": False, "enable_time_constant": False, "enable_sv": True,  "enable_sz": True},
        {"enable_leak": False, "enable_time_constant": True,  "enable_sv": True,  "enable_sz": True},
        {"enable_leak": True,  "enable_time_constant": False, "enable_sv": True,  "enable_sz": True},
        {"enable_leak": True,  "enable_time_constant": True,  "enable_sv": True,  "enable_sz": True},
        {"enable_leak": False, "enable_time_constant": False, "enable_sv": False, "enable_sz": False},
    ]

    return [
        {
            "session_id": session_id,
            "prior_block": prior_block,
            **variant,
        }
        for variant in variants
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

class CPUUnpickler(pickle.Unpickler):
    """Remaps any CUDA storage to CPU when loading on a CPU-only machine."""
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)

def load_model(path):
    with open(path, "rb") as f:
        return CPUUnpickler(f).load()
