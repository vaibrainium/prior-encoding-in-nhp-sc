import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import dir_config

from src.ddm.ddm import DEFAULT_PARAMS, DecisionModel, ParamSpec, validate_params


class DDMModel(DecisionModel):

    def __init__(self, enable_leak: bool = False, enable_time_constant: bool = False):
        super().__init__()
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
            "leak_rate":     ParamSpec(0.0,  (0.0,  1)),
            "variance":     ParamSpec(1.0,  (0.5,  2.0)),
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
            result["coherence"], np.asarray(data["coherence"]),
        )


if __name__ == "__main__":
    # get session_id from command line arguments
    parser = argparse.ArgumentParser(description="Fit session data")
    parser.add_argument("--session_id", type=int, required=True, help="ID of the session to fit")
    parser.add_argument("--enable_leak", type=bool, default=False, help="Whether to include leak in the model")
    parser.add_argument("--enable_time_constant", type=bool, default=False, help="Whether to include time constant in the model")
    args = parser.parse_args()

    # directory setup
    ddm_dir = Path(dir_config.data.processed) / "ddm"
    model_folder = ddm_dir / f"ddm_leak_{args.enable_leak}_time_constant_{args.enable_time_constant}"
    output_dir = Path(ddm_dir, model_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load session data
    behavior_df = pd.read_csv(Path(dir_config.data.processed) / "behavior.csv")
    session_ids = behavior_df["session_id"].unique()
    idx_prior, idx_session = args.session_id // len(session_ids), args.session_id % len(session_ids)
    session_id = session_ids[idx_session]
    prior_block = behavior_df["prior_block"].unique()[idx_prior]

    # Verify CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU configuration.")

    data = behavior_df[(behavior_df["session_id"] == session_id) & (behavior_df["prior_block"] == prior_block)]
    data = data[["rt", "choice", "coherence"]].reset_index(drop=True)
    data['choice'] = data['choice'].astype(int)

    stimulus_length = int(data["rt"].max() * 1000)
    stimulus = np.tile(data["signed_coherence"].to_numpy().reshape(-1, 1), (1, stimulus_length)) / 100

    model = DDMModel(enable_leak=args.enable_leak, enable_time_constant=args.enable_time_constant)
    result = model.fit(
        data=data,
        stimulus=stimulus,
        n_reps=15,
        max_iterations=500,
        l1_weight=0.01,
        verbose=False
    )

    # Save results
    with open(Path(output_dir, f"{session_id}_prior_block_{prior_block}.pkl"), "wb") as f:
        pickle.dump({"models": model, "results": result, "session_id": session_id}, f)
