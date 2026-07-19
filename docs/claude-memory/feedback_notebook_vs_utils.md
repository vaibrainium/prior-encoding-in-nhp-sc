---
name: feedback-notebook-vs-utils
description: Keep analysis code in notebooks; do not add to or overwrite dpca_utils.py or dpca_plot_utils.py
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 8517e2d2-1ea7-4dc3-9c13-b34eb3dfdf70
---

Keep exploratory/analysis code in the notebook itself — do not add new functions to `src/utils/dpca_utils.py` or `src/utils/dpca_plot_utils.py` unless explicitly asked.

**Generalized (2026-07-12):** On the `mary` branch, do **not modify any existing util or notebook** — create **new files only**. When reusing helpers from another branch (e.g. `origin/vaib`), port them into a NEW self-contained module rather than editing shared files like `ephys_utils.py` or `__init__.py`.

**Why:** User rejected both a new function added to dpca_plot_utils and a full overwrite of that file, and later stated explicitly: "when on mary's branch, don't change existing utils and notebooks." They want mary's existing files stable (WIP dPCA work + shared utils); notebook cells + new modules are the right place for new analyses.

**How to apply:** When asked to add analysis logic (a new plot, loop, decoder), write it inline as notebook cells, or in a NEW `src/utils/<name>.py`. Only touch an existing util/notebook when the user explicitly says to. See [[project-decoding-8-20]].
