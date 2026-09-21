# Procedural-Family Skill Consolidation (SkillGLoW)

Status: design proposal
Author: Halldrix
Supersedes: N/A

## Problem
Self-improving agents keep skills as one global document (collapses into generic discipline) or a flat pool of per-task entries (inflates, bound to the writing instance). Both fail on long-horizon streams with diverse tasks.

## Proposal — Global-Local Weave
The unit of reuse is the solving procedure shared by a cluster of related tasks. Local skills written from a task's own execution are aggregated into procedural families and compressed into de-instantiated global priors. Instance detail is regenerated per task, not stored. A commit gate admits a prior only when real execution shows no degradation of the deployed library.

## Evidence (arXiv:2609.02217)
+17.2 points (hard) over no-skill baseline on average across 4 benchmarks (math reasoning, terminal automation, software repair, embodied control) x 3 models; positive gains in all 12 continual runs; 18.0 with local regeneration. One prior per family, 3.6x more compact than per-task pool; leads a published single-document optimizer on 15 of 21 cells; unseen ALFWorld transfer 73.9% to 83.9%.

## Mapping to Hermes
Skills library + curator + scheduled consolidation job. One prior per procedural family instead of unbounded per-task entries.
