---
name: project-conda-env
description: The correct conda environment for running notebooks and code in this project
metadata: 
  node_type: memory
  type: project
  originSessionId: ba5b50c9-853b-4b70-bf44-f72c0f9a2641
---

Always use the `prior-sc` conda environment for this project.

**Why:** The base Anaconda environment is missing key packages (matplotlib, etc.). The project's environment is `prior-sc` at `C:\Users\mbasso.NPI-NRB507-XPS\.conda\envs\prior-sc`.

**How to apply:** When executing notebooks via nbconvert, use `conda run -n prior-sc jupyter nbconvert ...`. When running Python scripts, use `conda run -n prior-sc python ...`.

**macOS tooling note (2026-07-12):** on this Mac, `conda run -n prior-sc python ...` **swallows stdout** unless you pass `--no-capture-output` (i.e. `conda run --no-capture-output -n prior-sc python ...`). Add `flush=True` / merge stderr when debugging. Env lives at `/Users/HONGYE/opt/anaconda3/envs/prior-sc` here (not the Windows path).
