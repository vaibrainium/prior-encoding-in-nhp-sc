# Memory Index

- [Project conda environment](project_conda_env.md) — use `prior-sc` env for all notebooks and scripts; base Anaconda lacks key packages
- [Notebook vs utils](feedback_notebook_vs_utils.md) — keep analysis code in notebooks; never add to or overwrite dpca_utils.py / dpca_plot_utils.py without explicit request
- [Cross-projection scope](feedback_cross_projection_scope.md) — cross-projection only for main all-neuron Example 1; skip for cell-type leave-out, awayRF, and other variants unless asked
- [Self-projection CI band](self_projection_ci_band.md) — analytic-SEM shade on self-projection thick lines (condition-balanced, reflects sig mask); notebook-local in 4.10 Example 1 only; data on Z: share unreachable from background shells
