import numpy as np
import pandas as pd
from typing import Tuple


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
