# Memory Index

- [Project conda environment](project_conda_env.md) — use `prior-sc` env for all notebooks and scripts; base Anaconda lacks key packages
- [Notebook vs utils](feedback_notebook_vs_utils.md) — keep analysis code in notebooks; on mary, don't modify existing utils/notebooks — new files only
- [Decoding 8.20](project_decoding_8_20.md) — SC pseudo-population decoders (abs_coh/choice/hmm_state), paper §5.7–5.9 method, BatchNorm→Dropout→Linear, 800/200 per class, toRF/awayRF separate, cue trimmed data-driven
- [Cross-projection scope](feedback_cross_projection_scope.md) — cross-projection only for main all-neuron toRF + awayRF sessions in 4.10; skip for cell-type/neuron-count leave-out (now in 4.11) unless asked
- [Self-projection CI band](self_projection_ci_band.md) — analytic-SEM shade on self-projection thick lines (condition-balanced, reflects sig mask); notebook-local in 4.10 for both toRF & awayRF all-neuron sessions (4.10 reorganized 2026-07-01 into variance-explained/self-projection/cross-projection sections; cell-type leave-out moved to 4.11); data path from config/dir-config.yaml (Z: on Windows, /Volumes/BassoLabSharedDrives/... on macOS)
