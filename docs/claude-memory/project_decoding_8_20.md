---
name: project-decoding-8-20
description: SC pseudo-population decoders (notebook 8.20) — targets, method, paper, key data facts
metadata:
  node_type: memory
  type: project
  originSessionId: 8bad3b19-3286-48d2-827e-9cab749228e7
---

Decoding SC population activity, time-resolved across the 4 epochs (baseline/visual/cue/response),
fit **separately** for toRF vs awayRF prior sessions, **correct trials only** (for now).

**Deliverables (new files only, on `mary`):** `src/utils/decoding_utils.py` (self-contained: `get_neuron_ids`, `prepare_trial_info`, `get_session_trial_data`) and `notebooks/8.20-decoding-logistic.ipynb`. Vaib's PR (`origin/vaib`, notebooks 8.01/8.10/8.11) was evaluated: 8.11 had the right sampling but wrong model (no BatchNorm) + no RT trim + toRF-only; 8.01 leaked train/test. We took the good parts.

**Targets:** `abs_coherence` (evidence strength, multiclass), `choice` (toRF=1/awayRF=0, binary), `hmm_state` (biased=1/unbiased=0, binary). Signed coherence dropped. Choice is decoded as direction so it stays orthogonal to abs_coherence on correct trials.

**Method (paper: Zhang, Jutras, Dede, Walker, Buffalo, Fairhall, bioRxiv 2025.12.31.697231, §5.7–5.9):** decoder = single linear layer `BatchNorm1d → Dropout(0.5) → Linear`, cross-entropy (softmax in loss), argmax at test, Adam lr=1e-3 ~200 epochs. Input = 50 ms **smoothed firing rate** (mean of `convolved_spike_trains` per bin). Per session, per class: real-trial 80/20 split THEN bootstrap to **800 train / 200 test per class** (user chose 800/200; paper used 1000/200). Sessions stacked neuron-wise; 8 repeats → mean±SEM. Class-balance only (no nuisance matching). No feature selection in v1. One split reused across bins for clean cross-time (Goal 2, not yet built).

**Key verified data facts (2026-07-12):** 24 toRF / 19 awayRF sessions (excl `210210_GP_JP`,`241209_GP_TZ`); `prob_toRF∈{50,70}`toRF / `{50,30}`awayRF. Choice coding is **already correct** after `extract_hmm_state_trial_info`'s awayRF flip (`choice=1`=toRF: biased-block 0%-coh mean=0.635 toRF / 0.356 awayRF) — assert, don't re-flip. RT (correct): min 156 ms (toRF)/164 (awayRF) is an outlier; p1≈240, median≈460. RT is coherence-dependent. **Cue epoch trim = window-only, keep ALL trials** (computed data-driven: keep 50 ms bins with no NaN across all trials/neurons → ~5 bins to ~150 ms); no coherence-selection bias. Only cue is RT-truncated; response epoch carries the decision/movement period. `neuron_metadata.classification` has a `trash` class (excluded). `ephys_neuron_wise.pkl` is 3.1 GB (slow over the shared drive; cache locally when iterating).

See [[feedback-notebook-vs-utils]], [[project-conda-env]].
