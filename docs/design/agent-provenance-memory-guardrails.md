# Agent Provenance + Memory Guardrails — Design Proposal

Status: design proposal (not yet implemented)
Author: Halldrix
Supersedes: N/A

## Scope

Typed provenance guardrails for persistent agent memory in Hermes, combining arXiv:2609.02127 (Stored Is Not Supported) and arXiv:2609.02265 (CAPTURE).

## Background

- 02127: typed provenance graph (origin, dependency lineage, epistemic role, validity, disclosure scope) plus resolver (authorized projections, one evidential status, conflict/staleness/withholding flags, protected decision witness) and a generate-verify-revise mediator. Conformance suite: 24 cases; typed mediation blocked all 19 unsafe opportunities unqualified, preserved all 5 supported controls; flat rules released 19/19 unsafe, source-tag rule 18/19.
- 02265 CAPTURE: 480 held-out episodes / 96 users; 71.5% win vs 69.3% supervised baseline vs 66.1% heuristic; fixed-policy poisoning limited to 11.5% while accepting 83.5% of genuine preference updates; adaptive attacker (released weights) rises to 24.7%. Mechanisms: belief tracker, multi-timescale ledger, uncertainty-triggered clarification, counterfactual auditing of cited memories. Recency plus provenance rules alone are insufficient.
- Mapping to Hermes: persistent memory layer covers memory and skills persistence discipline; autobiographical state is built by reflection, retrieval, and consolidation. Principle: stored is not supported.

## Guardrail design

The persistent memory layer must apply typed provenance before any consolidation writes. Each memory entry carries origin, lineage, epistemic role, validity, and disclosure scope. The resolver produces one evidential status per projection and raises conflict, staleness, or withholding flags. The mediator applies generate-verify-revise before committing. Recency plus provenance rules are necessary but not sufficient; multi-timescale ledgers and counterfactual auditing of cited memories are required.
