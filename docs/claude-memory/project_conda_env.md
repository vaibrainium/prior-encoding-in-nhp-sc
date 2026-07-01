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
