---
name: feedback-cross-projection-scope
description: "Cross-projection analysis is only for the main all-neuron Example 1; skip it for cell-type leave-out, awayRF, and other variants until explicitly requested"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7d8b4ec3-b825-434b-8013-18dcefbef431
---

Cross-projection is ONLY for: (1) Example 1 all-neuron toRF (notebook 4.10), and (2) awayRF sessions. Do NOT add or modify cross-projection cells for: no-undefined (Example 2 in 4.11), cell-type leave-out, or any other variant.

**Why:** User explicitly rejected cross-projection edits to 4.11 three times: "no need to look at cross-projection in these analysis yet", "for excluding cell types no need for cross-period projection", "no cross-projection for no_undefined analyses".

**How to apply:** Leave cell e64f959e in 4.11 completely untouched. For new analysis sections (cell-type leave-out etc.), include ONLY self-projection. awayRF → full analysis (self + cross projection with sig masks).
