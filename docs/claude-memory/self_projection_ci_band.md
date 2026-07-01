---
name: self-projection-ci-band
description: "CI shade on dPCA self-projection thick lines — analytic SEM approach, rationale, and where it lives"
metadata: 
  node_type: memory
  type: project
  originSessionId: 711ee51c-d204-40cf-a8b3-be6cdd319bed
---

Goal: show on each dPCA self-projection axis whether the condition means (e.g. 2 choices) **separate** and **when** — i.e. when SC encodes choice/state/coherence — as a visual companion to the significance bar.

Decision (settled after long deliberation):
- The **thick line** = condition-balanced marginal mean projected on the demixed axis; it already is project-then-average with equal weight per condition (`np.nanmean` over non-key axes), matching the sig test and Kobak. Unchanged.
- The **shade** = analytic SEM propagated from per-cell trial stats in `fit_tw`: `SEM²_cell = s2_cell/n_cell` → condition-balanced collapse `(1/K²)Σ_cells` → projection `Σ_n w_n²·Var`. `w_n = D[marg][:,PC]`. This is the exact closed form of a per-cell trial bootstrap (validated ratio ≈0.96, gap = bootstrap ddof bias), but instant. **Bootstrap dropped** for production.
- Weighting is **condition-balanced (not trial-pooled)** to match the significance mask. Chosen over "across-non-key-condition std" because only the trial-resampling band speaks to whether the means separate.
- Data has **no 1-trial cells** (user confirmed) and `clean_dpca_data` leaves kept timepoints dense → band is exact, no special NaN handling needed.

Where it lives: implemented **notebook-local in `notebooks/4.10-dpca-utils-examples.ipynb`, Example 1 (toRF all-neuron) ONLY** — cells `thick_line_sem` + `plot_self_projection_ci`. Do NOT port into `dpca_plot_utils.py` or apply to awayRF/cell-type/blocks unless asked (see [[feedback-notebook-vs-utils]] and [[feedback-cross-projection-scope]]).

4.10 Example 1 is confirmed consistent with `scripts/dpca/run_dpca_analysis.py` (same sessions/exclusions/condition grid/fit/project/self-sig kwargs; shared sig `.pkl` cache).

Practical constraint: the dataset lives on the **`Z:` network share** (`Z:\BassoLabShare\...\data\processed`), which is **not reachable from background / non-interactive shells** — so figures must be produced by running the cells in the live Jupyter kernel, not via a spawned script. Save outputs to `processed_dir/dpca/` (user currently lacks `dissemination/` access).
