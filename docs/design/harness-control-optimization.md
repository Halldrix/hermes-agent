# Harness Optimization — Reflection/Control Slot Localization

Status: design proposal
Author: Halldrix
Supersedes: N/A

## Context

The harness-optimization value from arXiv:2609.02889 (HARNESSEVO) is localized, not uniformly distributed across the harness slots.

## Verified findings

- Setup: frozen 7B backbone, harness split into 4 separately evolvable slots (role, task-strategy, tool/format-rules, reflection/control), same reflective optimizer, iso-budget, leave-one-in / leave-one-out attribution.
- ALFWorld: full HARNESSEVO 0.657 vs 0.642 stock harness vs 0.642 flat-string evolution (no significant overall gain), but nearly all value localized in reflection/control slot (+0.119 leave-one-in gain); other slots individually null.
- Budget-splitting trap: 64 rollouts across 4 slots = 16 per slot, below the optimizer's effective search floor; every slot freezes at its empty seed. Concentrating the budget on the high-credit control slot recovers the gain: 0.761 with half the split budget.
- WebShop: all slots freeze empty and all methods tie — genuine absence of recurrent verbalizable control failures, not budget starvation.

## Recommendation

Evolve the reflection/control slot first under a fixed budget; expand to other slots only with leave-one-in evidence. Uniform budget splitting is actively harmful. Credit assignment must precede structured agent-evolution.

Refs: arXiv:2609.02889
