---
name: feedback-notebook-vs-utils
description: Keep analysis code in notebooks; do not add to or overwrite dpca_utils.py or dpca_plot_utils.py
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 8517e2d2-1ea7-4dc3-9c13-b34eb3dfdf70
---

Keep exploratory/analysis code in the notebook itself — do not add new functions to `src/utils/dpca_utils.py` or `src/utils/dpca_plot_utils.py` unless explicitly asked.

**Why:** User rejected both a new function added to dpca_plot_utils and a full overwrite of that file. They want utils to stay stable; notebook cells are the right place for one-off analyses.

**How to apply:** When asked to add analysis logic (e.g., a new plot, a new loop), write it inline as notebook cells. Only touch utils when the user explicitly says "add to utils" or "add to dpca_plot_utils".
